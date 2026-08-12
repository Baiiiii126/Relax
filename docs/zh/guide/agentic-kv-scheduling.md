# Agentic KV 调度

两个相互独立、默认关闭的特性，用于降低 agentic rollout 期间的 KV cache 压力：**session KV lifecycle** 在 session 结束时立即释放其 KV，**program-aware admission** 在 request 边界上限制活跃工作集大小。

## 概述

在 agentic rollout 中，一个 session 是长生命周期的多轮 program，而不是单次 request。两轮之间 agent 在执行 tool——可能耗时数秒到数分钟——而该 session 的 KV prefix 一直驻留在 engine 里。并发 session 一多，KV pool 就会打满，engine 退化为 eviction 和 recompute，新到达的 request 排在这些并没有真正在 decode 的工作后面。

Relax 从 session 生命周期的两端分别处理这个问题：

| 特性 | Flag | 作用位置 | 效果 |
|---|---|---|---|
| Session KV lifecycle | `--agentic-session-lifecycle` | session 结束时 | 立即释放 session 的 KV，而不是等 LRU eviction |
| Program-aware admission | `--agentic-program-admission` | 每次 request 开始前 | 限制集群同时承诺出去的 KV 总量 |

两个特性都**默认关闭**、**彼此独立**（可以只开其中一个）、并且都**fail-open**——任何信号缺失、过期或失败都会回退到原有行为。两者都不改变生成结果：完整的 replay payload（`input_ids`）始终会被发送，所以即使 cache 是冷的也能正确服务。

::: tip
这两个特性优化的是*调度*开销，不是模型质量。当 rollout 是 KV-bound 时才需要开启——engine `token_usage` 高、频繁 eviction、GPU 没打满但 request 在排队。
:::

## Session KV Lifecycle

### 做了什么

开启后，SGLang backend adapter 会在每次 `generate` 调用时把 engine session id 作为顶层 `session_id` 字段发送，从而在 server 的 session-radix cache 上标记该叶子节点。当 session 进入终态时——`finalize_and_discard` 或 `discard_session`——Relax 会先 abort 或静默掉在途 request，然后发出幂等的 `/close_session`，立即释放该 session 的 KV。

### 路由

sgl-router **不会**代理 `/close_session`。因此 Relax 直接向每个 engine base URL 扇出，方式与 request abort 一致。engine 的 DP controller 会把释放广播到所有 DP rank；不持有该 session 的 rank 会 no-op。

```
                    ┌──────────────────┐
   generate ───────►│   sgl-router     │───────► engine (placement decided here)
   (session_id)     └──────────────────┘
                    ┌──────────────────┐
   /close_session ─X│   sgl-router     │   not proxied
                    └──────────────────┘
                             │
   /close_session ───────────┴──────────────► every engine base URL directly
                                              (DP controller broadcasts to all DP ranks)
```

### 前置条件

SGLang server 必须带 `--sglang-enable-session-radix-cache` 启动。否则 `session_id` 字段会被直接忽略、close 调用也是 no-op——只开 Relax 这一侧的 flag 没有任何收益。

::: warning
`--sglang-enable-session-radix-cache` 和 `--sglang-radix-eviction-policy` 不是 Relax 定义的参数。它们是 SGLang `ServerArgs` 通过 `--sglang-` 前缀自动暴露出来的（见 `relax/backends/sglang/arguments.py`），因此只有当你安装的 SGLang 提供这两个参数时它们才存在。SGLang 0.5.15 中两者均存在。
:::

### 失败行为

close 只是一个 KV 释放优化。失败和超时会被记录并吞掉，绝不阻塞逻辑终态；受影响的 session 退回 LRU eviction。可以关注这些 warning：

```
close_session skipped: cannot list SGLang workers: ...
close_session skipped: no SGLang worker urls available.
close_session: 3/8 call(s) failed; affected sessions fall back to LRU eviction.
```

### 配置项

| Flag | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--agentic-session-lifecycle` | flag | `False` | 开启该特性 |
| `--agentic-session-close-timeout-ms` | int | `500` | close 扇出的单次调用超时；`0` 表示不设上限 |
| `--agentic-session-close-max-retries` | int | `2` | 单次调用的重试次数，仅针对瞬时 HTTP 错误；必须 `>= 1` |

## Program-Aware Admission

### 做了什么

在每个 request 边界上——`InflightRequest` 已构建并入队之后、**尚未**获取 SGLang permit 也尚未调用 `generate()` 之前——admission 返回三种决策之一：

| 动作 | 行为 |
|---|---|
| **Bypass** | 保持原有路径不变，不持有 lease |
| **Admit** | 持有一个集群级 execution-token lease，request 结束时释放 |
| **Defer** | 把该 request 挂在 shard 侧，不占用 permit；由 resume pump 稍后重启 |

无论怎样配置，两条不变式始终成立：

- Admission **绝不选择 worker**。最终 placement 仍由 SGLang router 决定。
- Admission **绝不中断在途 decode**。它只拦截尚未开始的工作。

### 架构

```
┌──────────────────────┐   reserve / release   ┌────────────────────────────────┐
│  AgenticSessionShard │ ────────────────────► │  AdmissionBudgetCoordinator    │
│  (16 Ray actors)     │ ◄──────────────────── │  single Ray actor, one writer  │
│                      │        grant          │  wraps the BudgetState ledger  │
└──────────┬───────────┘                       └───────────────┬────────────────┘
           │                                                   │ poll /metrics
           │ generate() only when not deferred                 │ every reconcile tick
           ▼                                                   ▼
┌──────────────────────┐                       ┌────────────────────────────────┐
│     sgl-router       │ ────────────────────► │        SGLang engines          │
└──────────────────────┘                       └────────────────────────────────┘
```

| 组件 | 职责 | 实现位置 |
|---|---|---|
| **决策逻辑** | 纯粹的 Bypass/Admit/Defer 策略与 token ledger，不含 Ray 和 I/O | `relax/agentic/session/admission.py` |
| **Coordinator** | 单写者 Ray actor、`/metrics` 轮询、lease TTL 回收 | `relax/agentic/session/admission_coordinator.py` |
| **Shard 集成** | 特征采集、defer gate、resume pump、每 shard 计数器 | `relax/agentic/session/service.py` |

ledger 刻意与 Ray 解耦，使得策略和预算记账可以在 CPU 上做确定性单测（`tests/test_agentic_rollout.py`）。所有与时间相关的状态都通过注入的 monotonic `now` 传入。

### Reservation 的计算

一次 reservation = **精确的 prompt token 数 + 有界的 decode 估计**。decode 估计取三者中最紧的：sampling `max_new_tokens`、`--agentic-admission-expected-decode-cap`、`--rollout-max-response-len`，因此永远不会无界。

### 决策顺序

prelude 先执行，不消耗任何 RPC：

1. 特性未开启 → **Bypass**
2. 没有 session id → **Bypass**（`missing_identity`）
3. scope 不允许 → **Bypass**（`capability_unavailable`）
4. protected 工作 → **Bypass**（`fairness_reserve`）——protected 工作永不被饿死
5. session 已被标记为压力状态 → **Defer**（`pressure_guard`）

否则 shard 会咨询 coordinator，由它决定授予还是拒绝：

| 拒绝原因 | ledger 判定条件 | 调用方行为 |
|---|---|---|
| `degraded` | 没有快照，或最近一次快照超过 3 个 reconcile 间隔 | **Bypass**（fail open） |
| `pressure_guard` | 最坏情况 engine `token_usage` ≥ 压力阈值，且请求未 aged | **Defer** |
| `capacity_exhausted` | `reserved + tokens` 超过上限；非 aged 请求的上限不含 emergency reserve | **Defer** |

ceiling = `健康 engine 的 max_total_num_tokens 之和 × headroom`。

### Lease

Lease 按 `(epoch, dispatch_id, decision_id)` 幂等，因此重试的 reserve 会返回同一个 lease，而不会重复扣减预算。TTL 负责回收因 shard 死亡而搁浅的 lease。当健康 worker 集合或 serving weight version 发生变化时，coordinator 会 bump epoch 并丢弃所有旧 epoch 的 lease。

### 防饥饿

被 defer 的 session 由每 shard 的 pump 按最早优先顺序恢复，tick 间隔为 `--agentic-admission-reconcile-interval-s`：

- 在最大等待时间以内，只有当 coordinator 报告其队头 request 有足够空间时才恢复。
- 如果 ledger 处于 degraded，被 defer 的 session 立即恢复——重新 admit 时会走 bypass。
- 超过 `--agentic-admission-max-wait-s` 后，session 会被**强制恢复**（aged）。aged 请求跳过 pressure guard 并可动用 emergency reserve；且一个 aged 请求如果仍会被 defer，则改为 bypass，从而保证一定能推进。
- 每个 shard 每个 tick 的强制恢复次数由 `--agentic-admission-forced-resume-cap` 限制。

### 前置条件

coordinator 通过 SGLang router 发现 engine（`/workers`，失败则回退 `/list_workers`），并抓取每个 engine 的 Prometheus `/metrics`。如果 router 地址未设置或不可达，就没有快照，ledger 会报告 `degraded`，所有请求都走 bypass。

::: tip
由于 coordinator 读取的是 engine 侧的 gauge，TP 复制会造成影响。SGLang 会为每个 rank 各发一份 `sglang:max_total_num_tokens` 之类的指标，且值完全相同，因此 Relax 用 **max** 而非 sum 聚合。若用 sum，容量会被放大、usage 会被缩小，倍数都是 TP 度，结果就是一个已经打满的 engine 看起来几乎空闲。
:::

### 配置项

| Flag | 类型 | 默认值 | 约束 | 说明 |
|---|---|---|---|---|
| `--agentic-program-admission` | flag | `False` | — | 开启该特性 |
| `--agentic-admission-headroom` | float | `0.90` | `(0, 1]` | 可用作 ceiling 的 KV 总容量比例 |
| `--agentic-admission-pressure-threshold` | float | `0.92` | `(0, 1]` | 单 worker `token_usage` 达到该值时非 aged 请求 defer |
| `--agentic-admission-expected-decode-cap` | int | `--rollout-max-response-len` | `> 0` | 单次 reservation 的 decode token 上界 |
| `--agentic-admission-max-wait-s` | float | `30.0` | `>= 0` | 强制恢复前的最长 defer 时间 |
| `--agentic-admission-reconcile-interval-s` | float | `2.0` | `> 0` | coordinator reconcile 与 resume pump 的 tick 间隔 |
| `--agentic-admission-lease-ttl-s` | float | `600.0` | `> 0` | lease TTL，用于回收死亡 shard 的 lease |
| `--agentic-admission-reserve-timeout-ms` | int | `100` | `> 0` | reserve RPC 超时；超时则该请求 bypass |
| `--agentic-admission-emergency-reserve-frac` | float | `0.05` | `[0, 1)` | 为 aged 请求预留的 ceiling 比例 |
| `--agentic-admission-forced-resume-cap` | int | `8` | `>= 1` | 每 shard 每 tick 的最大强制恢复数 |
| `--agentic-admission-scope` | str | `train` | `train` \| `all` | 只作用于 train，还是 train + eval |

上述数值约束仅在设置了 `--agentic-program-admission` 时才会校验。

## 快速开始

在已有的 agentic rollout 启动脚本中加入：

```bash
AGENTIC_ARGS=(
   --use-agentic-rollout
   # ... 已有的 agent flags ...

   # Session KV lifecycle：依赖 server 侧的 session radix cache
   --sglang-enable-session-radix-cache
   --sglang-radix-eviction-policy priority
   --agentic-session-lifecycle

   # Program-aware admission：默认值是一个合理的起点
   --agentic-program-admission
   --agentic-admission-headroom 0.90
   --agentic-admission-pressure-threshold 0.92
)
```

完整示例见 `examples/mini_swe_agent/run_mini_swe_agent.sh`。

## 监控指标

两个特性都按 rollout step 上报一次，与已有的 `rollout/` 和 `perf/` 指标并列。所有 series 共用 `agentic_kv/` 前缀，因此按首段路径分组的 tracker——ClearML 会以第一个 `/` 把 key 拆成 `(title, series)`——会把它们渲染成一个面板，而不是三个。

| 指标 | 含义 |
|---|---|
| `agentic_kv/session/lifecycle_enabled` | 开启 session lifecycle 时为 `1.0`，否则不上报 |
| `agentic_kv/admission/admit` / `defer` / `bypass` | 每 step 的决策计数（至少有一次决策时才上报） |
| `agentic_kv/admission/defer_rate` | `defer / (admit + defer + bypass)` |
| `agentic_kv/admission/degraded_rate` | 因信号 degraded 而 fail-open 的决策占比 |
| `agentic_kv/admission/forced_resume` | 本 step 被强制恢复的 aged session 数 |
| `agentic_kv/admission/reserve_error` | 抛异常并 fail-open 到 bypass 的 reserve RPC 数 |
| `agentic_kv/admission/defer_wait_ms_mean` | 在 gate 后平均停留时间 |
| `agentic_kv/admission/state_ready` / `state_in_flight` / `state_acting` / `state_deferred` | 实时 session 状态分布 |
| `agentic_kv/budget/ceiling` / `agentic_kv/budget/reserved` | 集群 execution-token 上限与当前预约量 |
| `agentic_kv/budget/reserved_utilization` | `reserved / ceiling` |
| `agentic_kv/budget/kv_token_usage_mean` / `agentic_kv/budget/kv_token_usage_max` | 窗口内 engine KV usage 的均值与峰值 |
| `agentic_kv/budget/epoch` | coordinator epoch，worker 集合或 weight version 变化时递增 |
| `agentic_kv/budget/degraded` | ledger 没有新鲜快照时为 `1.0` |

::: warning
`agentic_kv/budget/kv_token_usage_*` 采样自一个每 step 排空一次的滑动窗口，因为在打日志那一刻做瞬时读取时 rollout 已经排空，会低估真实峰值。engine 侧的释放收益——pool 大小、强制 eviction、释放的 token 数——不在此表中，需要从 engine 自身的 Prometheus `/metrics` 读取。
:::

## 故障排除

| 现象 | 可能原因 | 检查项 |
|---|---|---|
| 全部走 bypass，`agentic_kv/admission/defer` 始终为 0 | ledger degraded | `agentic_kv/budget/degraded == 1.0`。router 地址未配置或不可达，或 `/metrics` 抓取失败。日志中查找 `Admission coordinator reconcile failed`。 |
| `agentic_kv/admission/defer_rate` 高但 `agentic_kv/budget/reserved_utilization` 低 | 是 pressure guard 在触发，而非容量不足 | engine `token_usage` 已达到阈值。调高 `--agentic-admission-pressure-threshold`，或降低并发。 |
| `defer_rate` 高且 `reserved_utilization` 接近 1.0 | 确实是容量受限 | 调高 `--agentic-admission-headroom`，或减少并发 session 数。 |
| `agentic_kv/admission/forced_resume` 每 step 持续上升 | session 经常撞上 aging 截止时间 | 预算相对于负载过紧，gate 退化成了单纯的延迟。放宽 headroom 或降低并发。 |
| 开了 session lifecycle 但 KV 不下降 | server 侧 cache 未开启 | 确认 engine 启动时带了 `--sglang-enable-session-radix-cache`。 |
| 出现 `close_session skipped: ...` warning | worker 发现失败 | router 不可达。相关 session 退回 LRU eviction，生成结果不受影响。 |

## 下一步

- 阅读 [Agentic Rollout](./agentic-rollout.md) 了解这两个特性所挂载的 session 生命周期。
- 阅读 [性能调优](./performance-tuning.md) 了解更完整的 rollout 吞吐检查清单。
- 阅读 [OOM 排查](./oom-troubleshooting.md) 了解 KV 压力演变成 OOM 时的处理方式。
