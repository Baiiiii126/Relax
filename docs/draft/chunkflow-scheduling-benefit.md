# ChunkFlow step-scheduling 收益对齐验证：实验方案

## 0. 目的

回答四个问题，并用实验把它们钉死：

1. **理论收益多少** —— 一次 step 里，ChunkFlow 的调度能省掉多少流水线气泡？
2. **实测收益多少** —— 同样的一次 step，墙钟时间 / MFU 实际改善多少？
3. **是否对齐** —— 实测与理论差多少？
4. **差异的原因** —— 差额落在哪一项成本上？

本方案交付**可执行的实验设计 + 一个纯 CPU 的理论计算器**，不包含任何 GPU 执行。

## 1. 理论模型

### 1.1 先说一个结构性事实（决定了一切）

`relax/backends/megatron/chunkflow/pipeline_schedules.py:222` 的主循环是：

```python
for mb_idx in range(_num_microbatches(num_microbatches)):
    parent = next_parent()
    ... warmup / steady / cooldown ...     # 344-399 行
```

也就是说 **ChunkFlow 为每一个 parent microbatch 单独跑一次完整的流水线**（warmup → 1F1B steady → cooldown），气泡 `(P-1)` 是**按 microbatch 付**的，不是整步摊销一次。`telemetry.py` 的气泡代理正是按这个结构算的（按 plan 加权）。

### 1.2 气泡公式

记 `M` = 一个 step 的 microbatch 数，`U` = 一个 step 拆出的 execution unit 总数，`P` = 流水线级数（`PP`）。

| 调度 | 一次 step 的结构 | 气泡因子 |
| --- | --- | --- |
| baseline（标准 Megatron 1F1B） | 一次流水线过 `M` 个 microbatch | `1 + (P-1)/M` |
| chunkflow | `M` 次流水线，共 `U` 个 unit | `≈ 1 + (P-1)·M/U` |

（两式分别对应 `telemetry.py:141` 与 `telemetry.py:142-144`；`telemetry.py:178-180` 输出的
`equal_unit_fill_drain_{baseline,chunkflow,delta}_proxy` 就是它们。）

**理论收益**

```
theory_x = [1 + (P-1)/M] / [1 + (P-1)·M/U]
```

**判据**：`theory_x > 1  ⟺  M² < U  ⟺  M < √U`

这个判据很关键：收益对 `M` 是**二次敏感**的。

- `M=1`（一个长序列，即 ChunkFlow 的目标场景）：baseline 气泡 100%，`theory_x ≈ P/(1+(P-1)/U)`。
- `M` 增大到 `√U` 附近：收益归零。
- `M > √U`：**ChunkFlow 反而更慢**（每次流水线重启的成本压过省下的气泡）。

### 1.3 理论收益是**上界**

`theory_x` 只算气泡，它隐含假设「一个 unit 的耗时 ∝ 它的 token 行数」。真实运行还要付：

| 成本 | telemetry 键 | 含义 |
| --- | --- | --- |
| 填充碎片 | `fragmentation_amplification` | unit 按 128 对齐填充，行数放大 |
| 重算 | `token_forward_recompute_multiplier` / `attention_forward_recompute_multiplier` | `keep_last_k` 触发的 Fr 重算 |
| 注意力矩形化 | `kernel_rect_pairs / logical_attention_pairs` | 因果三角被切成矩形，注意力 FLOPs 放大 |
| 算子启动 | `launch_amplification` | `num_units / num_microbatches`，每 unit 的固定开销 |
| 点对点通信 | `p2p_messages` / `p2p_row_transfers` | 更多、更小的 p2p 消息 |

**Level-2 修正预测**（把这些已建模的成本乘进去）：

```
T_pred ≈ T_compute · frag · tok_recomp · [1 + a·(rect_attn − 1)] · (1 + (P-1)·M/U)  +  β · N_units
```

其中 `a` = 注意力占算力的比例，`β` = 每个 unit 的固定开销（kernel launch + p2p 握手）—— 这两项 telemetry 给不出，**需要用 A/B 实测数据拟合**。拟合后剩下的残差就是「模型还没解释的部分」。

## 2. 实验设计

### 2.1 并行布局（先说，它排除了「8 卡全用」的直觉方案）

两条硬约束：

- **`DP` 必须 = 1。** telemetry 只从 `pp=last / tp=0 / dp=0` 这一个 rank 记录（`model.py:1548-1552`），而**实际执行的 `M` 是 DP 维度的 MAX**（`data.py:934-938`）。`DP>1` 时 telemetry 里的 `M` 会小于真实 `M` → 理论收益被系统性高估，理论/实测直接对不齐。
- **`PP` 必须整除层数。** Qwen3-4B 是 **36 层**（`scripts/models/qwen3-4B.sh`），故 `PP ∈ {1,2,3,4,6,9,12,18,36}` —— **`PP=8` 非法**（36/8=4.5）。

8 卡机器上的摆法：

| 方案 | 用卡 | DP | 说明 |
| --- | --- | --- | --- |
| `PP=2` × `TP=2`，`DP=1` | 4 | 1 | 仅用于先做 PP correctness smoke，不是最终 8 卡性能结果。 |
| `PP=6`，`DP=1`（可选） | 6 | 1 | `P` 更大 → 气泡更大 → 信号更强；但 PP=6 未经 chunkflow 验证。 |
| **`PP=4` × `TP=2`，`DP=1`（主实验）** | 8 | 1 | 用满 8 卡且保住 `DP=1`；TP 的通信开销计入真实收益。 |

> **不要**用 `PP=4 × DP=2`：`M` 被 DP-MAX 放大而 telemetry 只记 rank0，是本次实验最隐蔽的坑。

### 2.2 固定项

| 项 | 值 |
| --- | --- |
| 模型 | Qwen3-4B（`--hf-checkpoint`） |
| 并行 | `PP=4`，`TP=2`，`CP=EP=DP=1` |
| 数据 | 先跑 16K-32K tail-only 受控集，再跑 0-32K 真实长尾集 |
| 其它 | `--seed 1234`、`--attention-backend flash`、`--no-rope-fusion`、`--calculate-per-token-loss` |

### 2.3 变量（两个旋钮）

**`M` 由数据装箱决定，不是固定算式**（`relax/utils/data/data.py:116` `get_minimum_num_micro_batch_size`，调用点 `relax/backends/megatron/data.py:921-944`）。两臂拿到**同一个 `M`**（`model.py:1219-1228` 单次 `forward_backward_func(num_microbatches=M)`）。

- **`GBS`（主旋钮）**：每步样本数。`M ≤ GBS`，所以 `M=30` 需要 `GBS ≥ 30`。硬约束 `GBS ≤ 数据集大小`（`bootstrap.py:69`）。
- **`MAX_TOKENS_PER_GPU`（次旋钮）**：装箱容量。调小 → 每 microbatch 装更少样本 → `M` 变大。
  ⚠️ 容量**低于最长样本**时那条长样本自成一个大 microbatch（合法，但改变显存画像）。用计算器的 `WARNING` 行自查。
- **`CHUNK_SIZE`**：`0` = baseline 臂（走标准 Megatron 调度）；`>0` = chunkflow 臂。
- **可选对照**：`PACK_POLICY ∈ {staggered(主), lockstep}`。

本方案**固定 `MAX_TOKENS_PER_GPU=SEQ_LENGTH=32768`**，用不同 `GBS` 扫描。`M` 必须以运行时
`train/chunkflow/num_microbatches` 为准，不能只由 `GBS` 猜测。

| `GBS` | `M` | 说明 |
| --- | --- | --- |
| 1 | 1 | ChunkFlow 的目标场景；PP=4 下理论收益最大 |
| 4 | 由 telemetry 确认 | 收益衰减区 |
| 16 | 由 telemetry 确认 | 接近交叉区 |

### 2.4 步数（关键，别踩）

`run-qwen3-4B-8xgpu-chunkflow.sh` 默认 `NUM_ROLLOUT=1`，而 SFT 每个 rollout 恰好是 **1 个 optimizer step**（`data.py:849-852`）—— 也就是说**默认配置只跑 1 步，测不出任何性能**（无预热、单样本）。

`NUM_ROLLOUT=N` 就是 N 个 optimizer step（每步消费 `GBS` 个样本，数据集跨 epoch 循环）。默认
tail-only 矩阵分别使用 35 / 20 / 10 步；正式统计用 `--drop-first 5` 丢掉预热，并保留至少 30 个
测量 step 的主结果。大 GBS 点用于趋势复核，不与主结果共享置信度声明。

### 2.5 矩阵与跑法

每个点：baseline（`chunk=0`）跑一次，chunkflow 每个 `CHUNK_SIZE` 各跑一次。用驱动脚本一条命令跑完：

```bash
# 先干跑一遍看命令，不提交
DRY_RUN=1 bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh

# 真正跑（需要 MODEL_DIR / DATA_DIR）
MODEL_DIR=/path/to/models DATA_DIR=/path/to/data \
  bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh
```

默认 tail-only 矩阵（8 个 run，`POINTS` 可覆盖）：

| `GBS` | `M` | steps | `CHUNK_SIZE` |
| --- | --- | --- | --- |
| 1 | 由 telemetry 确认 | 35 | 0(baseline), 512, 1024, 2048, 4096 |
| 4 | 由 telemetry 确认 | 20 | 0(baseline), 1024 |
| 16 | 由 telemetry 确认 | 10 | 0(baseline), 1024 |

单个点也可以手跑：

```bash
GBS=1 MAX_TOKENS_PER_GPU=32768 NUM_ROLLOUT=20 \
  bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh baseline
GBS=1 MAX_TOKENS_PER_GPU=32768 NUM_ROLLOUT=20 CHUNK_SIZE=512 \
  bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh chunkflow
```

`GBS` / `MAX_TOKENS_PER_GPU` / `NUM_ROLLOUT` / `CHUNK_SIZE` / `PACK_POLICY` / `KEEP_LAST_K` 都是脚本里可覆盖的环境变量（默认值与改动前一致：`GBS=1`、`MAX_TOKENS_PER_GPU=SEQ_LENGTH`、`NUM_ROLLOUT=1`）。`EXP_NAME` 由驱动脚本按 `gbs/cap/chunk/steps` 生成，便于回查。

## 3. 采集清单

**实测**（两臂都采，ClearML）：

| 键 | 用途 |
| --- | --- |
| `perf/actor_train_time` | 调度收益主指标：纯训练时间 |
| `perf/step_time` | 端到端次指标，包含 train wait |
| `perf/actor_train_tflops` / `perf/mfu/actor_train` | 算力利用率 |
| `perf/step_token_per_s` | 吞吐 |
| `perf/wait_time_ratio` | 排除 rollout/wait 干扰 |

（定义见 `relax/utils/training/train_metric_utils.py:23-91`。）

**理论**（chunkflow 臂，`train/chunkflow/*`，`model.py:1568`）：重点键见 §1.3 的表，外加
`num_microbatches`、`num_units`、`equal_unit_fill_drain_{baseline,chunkflow,delta}_proxy`。

> **按 step 采集，不要只看均值。** 两臂用同一份数据、同一顺序、同一 `seed`，所以第 `i` 步在两臂里是**同一批样本** —— 逐 step 配对（`actor_train_time` 与 `equal_unit_fill_drain_*` 同一个 step index）比取平均更干净，也能看出 `U` 随样本长度的散布。

## 4. 对齐与归因方法

1. **先验证气泡模型本身**（不涉及 chunkflow）
   用 baseline 臂跨 `M` 的实测 `step_time` 拟合 `T(M) = T_compute·(1 + (P-1)/M)`。
   实测点应落在曲线上；偏离说明有气泡之外的固定成本 —— 这一步是整套归因的地基。

2. **理论 vs 实测 speedup 散点**
   逐 step 配对（见 §3 的提示）：`x = theory_x`（来自 `train/chunkflow/equal_unit_fill_drain_*`），
   `y = actor_train_time_baseline / actor_train_time_chunkflow`（同一步 index）。
   `y = x` 线即「完全对齐」。按 `M` 着色，看偏差是否随 `M` 系统性放大。

3. **Level-2 修正 + 残差归因**
   把 §1.3 的成本乘数乘进预测，用 A/B 数据拟合 `a`（注意力算力占比）与 `β`（每 unit 固定开销）。
   残差显著缩小 → 差异已被 telemetry 解释；残差仍大 → 指向未建模项（优先怀疑 `launch_amplification` 与 `p2p_*`）。

4. **判读产出**
   ① 理论收益是否可达；② 瓶颈是 frag / recompute / launch / rect-attn 中的哪一项；
   ③ 给定数据分布下推荐的 `CHUNK_SIZE` 与可用 `M` 上限（即 `√U` 判据的实测验证）。

## 5. 工具：理论计算器 + 扫描驱动

两个脚本配套：**计算器**算理论侧，**驱动**跑实测侧。

### 5.1 `chunkflow_benefit_model.py` —— 理论收益 + 成本分解

纯 Python，复用 chunkflow 自己的 `scheduler` / `units` / `telemetry`，**不需要 torch**，Mac 上即可跑。它算出的量与真实运行记录的 `train/chunkflow/*` 是同一个量、同一套算法。

```bash
# 长序列场景（ChunkFlow 的目标）：一条 ~28K 样本，PP=4
python3 scripts/training/sft/tools/chunkflow_benefit_model.py \
  --synthetic --sort desc --gbs 1,4,16,64 --max-tokens-per-gpu 32768 \
  --chunk-sizes 0,512,1024,2048

# 用自己的真实长度分布（JSON int 数组，或 JSONL 行带 total_lengths/token_count/length）
python3 scripts/training/sft/tools/chunkflow_benefit_model.py \
  --lengths lengths.json --pp-size 4 --gbs 64 --max-tokens-per-gpu 32768
```

**合成分布下的参考输出**（PP=4，`staggered`，`--sort desc`）：

| GBS | M | chunk | U | bubble_base | bubble_cf | theory_x | launch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 512 | 56 | 4.000 | 1.054 | **3.797** | 56.0 |
| 4 | 4 | 512 | 197 | 1.750 | 1.061 | 1.650 | 49.2 |
| 16 | 11 | 512 | 537 | 1.273 | 1.061 | 1.199 | 48.8 |
| 64 | 17 | 2048 | 284 | 1.176 | 1.179 | 0.998 | 16.7 |

读法：`M=1` 且序列够长时理论收益最大（≈3.8x）；`M` 涨到 `√U` 附近（此处 `U=284`，`√U≈17`）收益归零，与 §1.2 判据一致。

> 注意 `launch` 列同步涨到几十倍 —— 这正是理论收益会被吃掉的地方，也是 Level-2 要量化的东西。

### 5.2 `run_chunkflow_benefit_sweep.sh` —— 实测扫描驱动

按 §2.5 的矩阵依次提交 baseline / chunkflow 各臂，每个 run 一个可回查的 `EXP_NAME`。支持 `DRY_RUN=1` 干跑、`POINTS` 覆盖矩阵（格式 `gbs:capacity:steps:chunk,chunk,...`）。

```bash
DRY_RUN=1 bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh   # 先看命令
MODEL_DIR=... DATA_DIR=... bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh
```

`DATA_PROFILE=tail` 是默认主实验；`DATA_PROFILE=longtail` 跑真实分布泛化；
`DATA_PROFILE=smoke` 只做三步接线检查。sweep 必须同步串行执行，不能设置 `RAY_NO_WAIT=1`。
PP executor 尚未完成 GPU 端到端验证，因此正式实验前先用
`TP_SIZE=2 PP_SIZE=2 DATA_PROFILE=smoke` 做 4 卡 loss / grad-norm correctness smoke；通过后再使用默认
`TP_SIZE=2 PP_SIZE=4` 跑 8 卡性能矩阵。

### 5.3 `chunkflow_align.py` —— 实测收益与置信区间

```bash
python3 scripts/training/sft/tools/chunkflow_align.py \
  --baseline baseline.log --chunkflow chunkflow.log --drop-first 5
```

默认以 `perf/actor_train_time` 计算 `sum(T_baseline) / sum(T_chunkflow)`，并给出配对 bootstrap 95% CI。
如需端到端口径，追加 `--metric step_time`。成本代理分别报告，不直接相乘，避免重复计价。

单点失败不会中断整轮（末尾汇总失败数并以非零码退出）。

## 6. QS 提交

QS 侧只需要配三样，其余都在仓库脚本里：

| 项 | 值 |
| --- | --- |
| 镜像 | 含 Relax + Megatron + SGLang 的训练镜像 |
| 资源 | `{"sft": [1, 0], "actor": [1, 8]}`（脚本内已定，**8 卡**；原因见 §2.1） |
| 环境 | `MODEL_DIR`（含 `Qwen3-4B/`）、`DATA_DIR`（含 `chunkflow-longtail/train.parquet`） |

**第一步，生成受控长序列数据**（主实验，只需一次）：

```bash
python3 scripts/training/sft/tools/build_chunkflow_longtail_sft.py \
  --tokenizer "${MODEL_DIR}/Qwen3-4B" \
  --output "${DATA_DIR}/chunkflow-tail/train.parquet" \
  --synthetic-only --short-count 0 --medium-count 0 --long-count 0 --tail-count 64
```

它稳定生成 64 条 16K-32K 样本，适合先验证调度收益上界，不受下载和样本分布漂移影响。

**第二步，生成真实长尾泛化集**（Infinity-Instruct + LongWriter-6k）：

```bash
python3 scripts/training/sft/tools/build_chunkflow_longtail_sft.py \
  --tokenizer "${MODEL_DIR}/Qwen3-4B" \
  --output "${DATA_DIR}/chunkflow-longtail/train.parquet"
```

构造器会把源 revision、输出 SHA256 和桶统计写到同目录 `.stats.json`。不要用 `--allow-partial`，
否则尾桶不足会静默改变实验分布；无网络时可追加 `--synthetic-only` 作为可复现替代。

**第三步，跑扫描**：

```bash
MODEL_DIR=... DATA_DIR=... PROMPT_DATA=${DATA_DIR}/chunkflow-tail/train.parquet \
  DATA_PROFILE=tail bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh

MODEL_DIR=... DATA_DIR=... PROMPT_DATA=${DATA_DIR}/chunkflow-longtail/train.parquet \
  DATA_PROFILE=longtail bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh
```

先 `DRY_RUN=1` 干跑确认命令与资源占用，再正式提交。**不硬编码任何集群路径或密钥。**

## 7. 验证（怎么确认本方案与工具本身是对的）

- 本地跑计算器，产出理论曲线与分解表（秒级，无 GPU）。
- `pytest tests/backends/megatron/test_chunkflow_telemetry.py` —— 确认指标口径与代码一致。
- 自洽性：`M=1` 时 `bubble_base` 应恰为 `P`；`M→∞` 时 `theory_x→1`；`M > √U` 时 `theory_x < 1`。
- 计算器内置校验：`_first_fit_partitions` 与 `relax.utils.data.data.get_minimum_num_micro_batch_size` 在 relax 可导入时逐点比对，不一致直接报错（防止 M 口径漂移）。

## 8. 一个顺带的重要观察

§1.1 的结构性事实（每个 microbatch 重启一次流水线）意味着 ChunkFlow 当前把气泡成本乘了 `M`。如果改成**让所有 microbatch 的 unit 汇入同一条流水线**（一次 fill/drain，`M` 从公式里消失），理论收益就不再受 `√U` 判据限制。

这**不在本方案范围内**（本方案只做「现状收益是否对齐」的验证），但它是本实验最可能指向的下一步。建议在 §4 的判读里明确记录：实测收益随 `M` 的衰减是否与 `√U` 判据吻合 —— 若吻合，即为该优化方向的直接证据。

## 附录：关键代码位置

| 内容 | 位置 |
| --- | --- |
| 气泡代理 + 成本乘数 | `relax/backends/megatron/chunkflow/telemetry.py:128-183` |
| 每 microbatch 一次流水线 | `relax/backends/megatron/chunkflow/pipeline_schedules.py:222,344-399` |
| M 的装箱决定 | `relax/utils/data/data.py:116`，`relax/backends/megatron/data.py:921-944` |
| 调度分派 | `relax/backends/megatron/model.py:1176-1228` |
| telemetry 落日志 | `relax/backends/megatron/model.py:1561-1568` |
| 实测 perf 指标 | `relax/utils/training/train_metric_utils.py:23-91` |
| 启动脚本 | `scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh` |
| 理论计算器 | `scripts/training/sft/tools/chunkflow_benefit_model.py` |
| 扫描驱动 | `scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh` |
| 步数语义（每 rollout 1 step） | `relax/backends/megatron/data.py:849-852` |
| 数据生成 | `scripts/training/sft/tools/build_chunkflow_longtail_sft.py` |
