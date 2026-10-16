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

记 `M` = 一个 step 的 microbatch 数，`U` = 一个 step 拆出的 execution unit 总数，`P` = 流水线级数（`PP`）。第 `i` 个 parent microbatch 有 `u_i` 个 unit、`R_i` 行填充后的 unit tokens。

| 调度 | 一次 step 的结构 | 气泡因子 |
| --- | --- | --- |
| baseline（标准 Megatron 1F1B） | 一次流水线过 `M` 个 microbatch | `1 + (P-1)/M` |
| chunkflow | `M` 次流水线，共 `U` 个 unit | `1 + (P-1)·Σ(R_i/u_i)/ΣR_i` |

（两式分别对应 `telemetry.py:141` 与 `telemetry.py:142-144`；`telemetry.py:178-180` 输出的
`equal_unit_fill_drain_{baseline,chunkflow,delta}_proxy` 就是它们。）

**气泡代理比值**

```
theory_x = [1 + (P-1)/M] / [1 + (P-1)·Σ(R_i/u_i)/ΣR_i]
```

当 `P>1` 时，**代理收益的精确判据**是 `Σ(R_i/u_i)/ΣR_i < 1/M`。只有各 unit 的填充行数近似相同，即 `R_i ∝ u_i` 时，才能简化为 `theory_x > 1 ⇔ M² < U ⇔ M < √U`。`P=1` 时两个气泡代理恒为 1。

简化判据说明在等 unit 成本模型下，收益对 `M` 是**二次敏感**的；正式实验应以逐 step telemetry 的实际代理值判断。

- `M=1`（一个长序列，即 ChunkFlow 的目标场景）：baseline 气泡因子为 `P`，`theory_x ≈ P/(1+(P-1)/U)`。`P=4` 时，相对无气泡理想执行时间的附加气泡为 300%。
- 等 unit 成本时，`M` 增大到 `√U` 附近：代理收益归零。
- 等 unit 成本且 `M > √U` 时，气泡代理预测 ChunkFlow 更慢；实际速度还取决于计算、通信与内存成本。

### 1.3 气泡代理不是实际速度上界

`theory_x` 只算理想化的 fill/drain，隐含各 unit 和各 stage 耗时可由 token 行数近似。它是实验对齐的参照值，不是实际 step speedup 的数学上界；分块可能改变算子效率或负载平衡。真实运行还要付：

| 成本 | telemetry 键 | 含义 |
| --- | --- | --- |
| 填充碎片 | `fragmentation_amplification` | unit 按 128 对齐填充，行数放大 |
| 重算 | `token_forward_recompute_multiplier` / `attention_forward_recompute_multiplier` | `keep_last_k` 触发的 Fr 重算 |
| 注意力矩形化 | `kernel_rect_pairs / logical_attention_pairs` | 因果三角被切成矩形，注意力 FLOPs 放大 |
| 算子启动 | `launch_amplification` | `num_units / num_microbatches`，每 unit 的固定开销 |
| 点对点通信 | `p2p_messages` / `p2p_row_transfers` | 更多、更小的 p2p 消息 |

`pipeline_schedules.py` 的依赖延迟 `C` 会增加 warmup 深度。单条长序列拆为 `U` 个依赖 unit 时，`C=U-1`，早期 stage 会先完成全部前向再开始反向。这在等耗时模型中不必然改变 fill/drain 比值，但会改变激活峰值与实际耗时分布；现有气泡代理没有刻画这部分。

**Level-2 修正预测**（把这些已建模的成本乘进去）：

```
T_pred ≈ T_compute · frag · tok_recomp · [1 + a·(rect_attn − 1)] · B_cf  +  β · N_units
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

本方案将 `MAX_TOKENS_PER_GPU` 和 `SEQ_LENGTH` 均固定为 **32768**，用不同 `GBS` 扫描。`M` 必须以运行时
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
   固定同一批样本和总 token 数，仅改变装箱容量以得到不同 `M`，用 baseline 臂的 `actor_train_time` 检查 `T(M) ≈ T_compute·(1 + (P-1)/M)`。
   跨 `GBS` 的运行不能直接用同一个 `T_compute` 拟合，因为总工作量也随之变化。装箱变化本身也会改变 kernel 形状，因此偏离曲线只能提示额外成本，不能单独证明气泡模型错误。

2. **理论 vs 实测 speedup 散点**
   逐 step 配对（见 §3 的提示）：`x = theory_x`（来自 `train/chunkflow/equal_unit_fill_drain_*`），
   `y = actor_train_time_baseline / actor_train_time_chunkflow`（同一步 index）。
   `y = x` 线即「完全对齐」。按 `M` 着色，看偏差是否随 `M` 系统性放大。

3. **Level-2 修正 + 残差归因**
   把 §1.3 的成本乘数乘进预测，用 A/B 数据拟合 `a`（注意力算力占比）与 `β`（每 unit 固定开销）。
   残差显著缩小 → 差异已被 telemetry 解释；残差仍大 → 指向未建模项（优先怀疑 `launch_amplification` 与 `p2p_*`）。

4. **判读产出**
   ① 理论收益是否可达；② 瓶颈是 frag / recompute / launch / rect-attn 中的哪一项；
   ③ 给定数据分布下推荐的 `CHUNK_SIZE`，以及加权气泡代理和实测收益各自的 `M` 交叉区。

## 5. 工具：理论计算器 + 扫描驱动

两个脚本配套：**计算器**算理论侧，**驱动**跑实测侧。

### 5.1 `chunkflow_benefit_model.py` —— 理论收益 + 成本分解

纯 Python，复用 chunkflow 自己的 `scheduler` / `units` / `telemetry`，**不需要 torch**，Mac 上即可跑。计算器用 first-fit 决定 `M`，再按训练逻辑进行长度均衡分组；文本 `CP=1` 场景还会按 `TP × data_pad_size_multiplier` 补齐每个 parent。输入长度与配置仍需匹配真实运行，正式对齐以运行日志的 `train/chunkflow/*` 为准。
计算器默认 `--tp-size 2 --pad-multiplier 128`，与主实验脚本一致；更改训练配置时同步覆盖这两个参数。

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
| 1 | 1 | 512 | 57 | 4.000 | 1.053 | **3.800** | 57.0 |
| 4 | 4 | 512 | 201 | 1.750 | 1.060 | 1.651 | 50.2 |
| 16 | 11 | 512 | 548 | 1.273 | 1.060 | 1.200 | 49.8 |
| 64 | 17 | 2048 | 288 | 1.176 | 1.177 | 0.999 | 16.9 |

读法：`M=1` 且序列够长时气泡代理比值最大（≈3.8x）；最后一行 `M=17`、`U=288` 的代理比值接近 1。`√U≈17` 在这里是等 unit 成本的近似解释，准确结果以加权代理为准。

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

默认以 `perf/actor_train_time` 计算 `sum(T_baseline) / sum(T_chunkflow)`，并给出配对 bootstrap 95% CI。理论汇总也按总时间口径计算：`sum(T_baseline) / sum(T_baseline/theory_x)`；报告两者之差对应的额外秒数。
如需端到端口径，追加 `--metric step_time`。成本代理分别报告，不直接相乘，避免重复计价。
单对 A/B 只能提出开销假设；要确认差异原因，需要比较不同 `CHUNK_SIZE` 的成对运行，观察额外秒数与 launch、padding、矩形注意力等代理如何一起变化。

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

它稳定生成 64 条 16K-32K 样本，适合先验证理想化调度代理是否对应实测收益，不受下载和样本分布漂移影响。

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
- 自洽性：`M=1` 时 `bubble_base` 应恰为 `P`；`P=1` 时两个气泡代理都为 1；`P>1` 时用 §1.2 的加权判据检查代理收益方向。固定每个 parent 的 unit 数并让 `M` 增大，`theory_x` 不一定趋近 1。
- 计算器内置校验：`_first_fit_partitions` 与 `relax.utils.data.data.get_minimum_num_micro_batch_size` 在 relax 可导入时逐点比对，不一致直接报错（防止 M 口径漂移）。

## 8. 一个顺带的重要观察

§1.1 的结构性事实（每个 microbatch 重启一次流水线）意味着 ChunkFlow 当前要为 `M` 个 parent 分别支付 fill/drain。如果改成**让所有 microbatch 的 unit 汇入同一条流水线**（一次 fill/drain），等 unit 成本模型里的 `M²<U` 限制就会消失。

这**不在本方案范围内**（本方案只做「现状收益是否对齐」的验证），但它是本实验可能指向的下一步。建议在 §4 的判读里记录实测收益随 `M` 的衰减，并与逐 step 加权气泡代理比较。

## 9. 逐 step 归因：预测是否解释实测波动

先保留执行实现，统计预测与实测的关系，再选择优化方向。每个配对 step 定义
`x = 100 × (theory_x - 1)`、`y = 100 × (actual_x - 1)`，单位都是加速比变化的百分数，
不与耗时减少百分比混用。

```bash
python3 scripts/training/sft/tools/chunkflow_align.py \
  --baseline "$baseline" --chunkflow "$chunkflow" \
  --theory-source proxy --drop-first 5 --output-prefix analysis/gbs4-chunk1024
```

`--theory-source proxy` 使用原有气泡公式，已有日志可回溯。脚本输出：

| 计数 | 预测正收益 | 预测负收益 |
| --- | --- | --- |
| 实际正收益 | 同方向 | 反方向 |
| 实际负收益 | 反方向 | 同方向 |

对默认 9 个 tail 实验，可自动选择 6 组 A/B 对比，并输出与上表相同的中文 Markdown 表：

```bash
python3 scripts/training/sft/tools/chunkflow_align.py \
  --sweep-log-dir log --theory-source proxy --drop-first 5 \
  --output-prefix analysis/chunkflow-attribution
cat analysis/chunkflow-attribution.md
```

完整日志去掉各组前 5 步后，共有 140 次 A/B step 比较（120+15+5）。中性点单独报告，
四格计数之和等于非中性比较数。自动选取每个配置最新 baseline 及之后的 CF 日志，
会打印实际使用的路径；若新一轮还未跑齐则报错，不与更早的 CF 日志静默混配。
自定义矩阵或指定历史轮次时，使用显式 `--baseline` / `--chunkflow`。

同时计算方向一致率、Pearson、Spearman 和平均绝对预测误差。`--tolerance-pct 1` 可将
绝对加速变化不超过 1% 的点归入中性，不计入正负方向表；相关系数仍使用全部点。
预测或实测恒定、或样本不足时相关系数输出 undefined，不能解释为零相关。
输出文件包括每步 `.csv`、统计 `.json`、带零轴和 `y=x` 参考线的散点 `.svg`。

重复传入成对的 `--baseline` / `--chunkflow` 可合并多个配置，预热过滤逐配置进行。
报告同时保留分配置和合并统计。合并相关性可能仅来自配置差异；复用同一个 baseline
也会让跨配置观测相关，因此不为合并结果生成独立 step bootstrap 置信区间。

### 调度时间轴预测

新的 `schedule_model.py` 与 PP 执行器共用前后向顺序及依赖延迟。CPU 模拟器按每个
stage 的执行顺序建立 DAG，FIFO 匹配阻塞通信的 shape/data 两阶段，求关键路径 makespan。
假设：每 128 行的 F 成本为 1、B 为 2、Fr 为 1，各 stage 相同，网络耗时为 0；
计入 unit padding 和计划内重算。它没有拟合本次实测时间，也不计真实 kernel 效率、
TP 通信、KV 复制、optimizer 和最终梯度同步，因此是声明假设后的时间轴模型，不是论文实现一致性的证明。

若事先知道一个 step 的各样本 token 长度，可在训练之前离线输出预测和调度图：

```bash
# step_lengths.json 是该 step 按原顺序排列的样本长度 JSON 数组
python3 scripts/training/sft/tools/chunkflow_benefit_model.py \
  --lengths step_lengths.json --gbs 4 --max-tokens-per-gpu 32768 \
  --chunk-sizes 1024 --pp-size 4 --tp-size 2 --timeline-dir analysis/timelines
```

表里的 `theory_x` 仍是旧气泡代理，`schedule_x` 才是两个模拟 makespan 的比值；
SVG 使用相同时间轴并排展示 baseline/CF，JSON 保存每个计算事件的开始与结束位置。

后续诊断运行可用 `RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=1` 启用运行时预测汇总。
计划在各 parent 执行前捕获，完整 step 模拟在 step 末汇总并上传 ClearML，
指标名为 `train/chunkflow/schedule_model_speedup`。这会增加 CPU 分析开销，默认关闭；
`schedule_model_analysis_time_s` 记录最后一个 PP stage 的模拟器 CPU 时间，
不代表所有 rank 的完整诊断开销，不应直接从实测时间中扣除。
有此指标的日志可使用 `--theory-source schedule` 做归因；缺少时脚本报错，不会静默
替换成气泡代理。现有实验可先用 proxy 分析，无需立即重跑整个矩阵。

高度正相关说明预测有解释力，但还要检查收益方向和预测误差；例如始终差一倍的预测
也可能高度相关。低相关应进一步检查样本配对、预测取值范围、额外开销和配置分层，
不能仅据此断言实现与论文不一致。本轮只增加测量和统计，不实现自动开关策略。

### 聚焦已有最佳候选，自动生成归因表

`run_chunkflow_attribution.sh` 默认 GBS=8，固定 TP2/PP4、capacity=32768、staggered、keep-last-k=0，
顺序运行 baseline 和 chunk=2048。2048 是当前实测最好的候选，已有置信区间仍跨过 1，不能保证正收益。
两组默认各运行 510 步，去掉前 10 对预热 step，保留 500 对。
GBS 可用环境变量调整，例如 `GBS=4` 或 `GBS=16`。默认 8 让短、中序列在同一 step 内混合打包；
它是候选配置，不代表最优值，microbatch 数 M 应以实际日志为准。

默认采用模拟的混合长度分布，先生成数据：

```bash
MODEL_DIR=/path/to/models DATA_DIR=/path/to/data \
  bash scripts/training/sft/tools/build_chunkflow_mixed_sft.sh
```

共 5000 条样本，按样本数量分配：0–2K 占 70%，2K–8K 占 20%，8K–16K 占 8%，
16K–32K 占 2%。最后 100 条长尾中有 25 条明确位于 28K–32K（最高 32256 tokens），
其余 75 条位于 16K–28K。长度按完整 chat template 渲染后的 prompt + answer 计算。
这是模拟分布，不是业务数据统计；长样本占用的 token 和计算量比例会高于样本数量比例。
生成器固定 seed、打乱顺序，并保存 `train.stats.json`，其中记录实际分桶计数、
28K–32K 样本数、长度范围和文件校验值。

```bash
MODEL_DIR=/path/to/models DATA_DIR=/path/to/data \
  bash scripts/training/sft/tools/run_chunkflow_attribution.sh
```

默认数据文件为 `$DATA_DIR/chunkflow-mixed/train.parquet`，也可以显式设置 `PROMPT_DATA`。
通过 `STEPS` 和 `DROP_FIRST` 调整总步数与预热步数。
设置 `DRY_RUN=1` 只打印配置和命令；不需要先跑 smoke。
必须取消 `RAY_NO_WAIT`，以保证两组顺序执行。

每次运行创建独立的 `analysis/chunkflow-attribution-时间-进程号/` 目录；
也可用 `REPORT_DIR` 指定尚不存在的目录。该目录包含：

- `baseline.log`、`chunk2048.log`：本次两组完整训练日志。
- `attribution.md`：四格计数表、Pearson、Spearman、实际加速及区间。
- `attribution.csv`：逐 step 的预测、实测收益及差额。
- `attribution.svg`：预测与实测收益散点图。
- `attribution.json`、`summary.txt`：结构化统计与分析终端输出。

新入口启用 `RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=1`，报告明确使用 `schedule` 预测，
不会回退到旧的 `proxy`。缺少预测、两组 token 数不同、配对步数不足或训练失败都会终止，
避免把不完整实验当成完整归因表。此目录应使用新入口的自动报告；旧的 `--sweep-log-dir` 匹配旧矩阵。

模拟器 CPU 开销可能进入训练计时；参照上文 `schedule_model_analysis_time_s` 的限制解释结果。
默认每组需要 4080 条样本，低于数据集的 5000 条；增加 GBS 或步数时应同时检查数据量，
避免重复读取。每步的 token 数与长度构成会不同，不能把所有计时视为独立同分布观测。
尤其是 28K–32K 只有 25 条，分桶结论应同时报告样本数。
本实验可检查收益方向和相关性，不能单凭相关性确定具体性能瓶颈。

## 10. 已有结果的深入诊断（无需训练）

```bash
python3 scripts/training/sft/tools/chunkflow_diagnose.py \
  --csv /path/to/experiment/attribution.csv \
  --output-prefix /path/to/experiment/diagnosis --tolerance-pct 0
python3 scripts/training/sft/tools/chunkflow_model_audit.py \
  --output /path/to/experiment/model-audit.json
```

诊断工具使用已去除预热的 CSV，不再次丢弃 step。每个 case 分开报告总体加速、
Pearson/Spearman、收益方向相关性、实际提速的 AUROC、预测收益分档、整步 token 分档和 M 分组。
M 分组还给出线性控制整步 token 数后的 Pearson，避免把共同长度趋势全归于调度预测能力。
预测方向恒正时方向相关性未定义，不等于连续收益相关性为零。

默认中性带为正负 1 个收益百分点；设为 0 可复现原归因表的方向定义。
默认用长度为 10 的连续 step block 做 bootstrap；可用 `--block-steps 5` 或 `20` 检查敏感性。
它只考虑局部依赖，不代替重复 A/B 实验，也不能排除先后运行造成的机器状态变化。
分档不能用于在同一批数据选择阈值后宣称泛化，成本代理的残差相关性也不能证明因果。
整步 token 数不能判断是否含超长样本，进一步的序列级归因需要额外输入记录。

模型核对工具使用论文第 3–5 节的四条长度示例，并报告当前模型与论文气泡比例的差异，
另外包含单长 parent 的正收益对照和多 parent 不切分的负收益对照。
它核对的是当前模型，不是独立实现的论文调度 oracle；时间轴数字不一致时，应先逐事件检查
执行顺序、依赖、通信语义和重计算位置，不能直接据此断定论文错误。

## 11. 现有 8 卡的分阶段实验

入口为 `scripts/training/sft/tools/run_chunkflow_8gpu.sh`。沿用 Qwen3-4B 和已有
mixed 数据，固定 TP=2、PP=4、DP=1、CP=1、capacity=32768、staggered、K=0。
这里 K=0 表示保留全部 chunk 激活。此配置用于验证当前实现的收益，不等同于论文配置。

| 阶段 | 顺序 | 默认步数 | seed | 目的 |
| --- | --- | --- | --- | --- |
| `screen` | baseline、chunk4096、chunk8192 | 每组 110，去掉前 10 | 1234 | 单独验证 chunk 粒度影响 |
| `confirm` | baseline/候选、候选/baseline、baseline/候选 | 每组 510，去掉前 10 | 每轮 1235、1236、1237，轮内一致 | 用新 seed 复验候选 |
| `profile` | baseline、候选 | 每组 14，采集 step 10–12 | 1234 | 采样 CPU/CUDA 操作，定位开销 |

默认 GBS=8，可用 `GBS` 覆盖。一次只改变一个因素；首轮保持模型、数据、GBS 和 K 不变。
若改变 GBS，要重新跑同 GBS 的 baseline，并检查数据量与样本重复情况。
`confirm` 和 `profile` 必须显式设置 `CHUNK_SIZE`，脚本不会自动选择或进入下一阶段。

### 第一阶段

在仓库根目录设置已有的 `MODEL_DIR`、`DATA_DIR`；默认读取
`${DATA_DIR}/chunkflow-mixed/train.parquet`，也可显式设置 `PROMPT_DATA`。
使用独占且空闲的 Ray 集群：已有 `ray-job.sh` 会清理旧训练和 Serve 状态，不应与其他任务并行。

```bash
unset RAY_NO_WAIT RELAX_ENTRYPOINT_MODE WORKING_DIR CHUNK_SIZE CHUNKFLOW_PROFILE
unset TP_SIZE PP_SIZE SEQ_LENGTH MAX_TOKENS_PER_GPU PACK_POLICY KEEP_LAST_K
unset REPEATS PROFILE_START PROFILE_END DRY_RUN
export REPORT_DIR="analysis/chunkflow-8gpu-screen-$(date +%Y%m%d-%H%M%S)"
GBS=8 STEPS=110 DROP_FIRST=10 TRAIN_SEED=1234 CHUNK_SIZES="4096 8192" \
  bash scripts/training/sft/tools/run_chunkflow_8gpu.sh screen
cat "${REPORT_DIR}/summary.md"
```

可用 `DRY_RUN=1` 只打印配置，不创建目录、不提交训练。没有 GPU smoke 阶段。
每组在当前代码与数据上顺序运行；旧结果目录不可覆盖。中途失败会停止后续运行，
保留已有日志。`manifest.json` 的 `status=complete` 才表示整组完成。
脚本校验完整 step ID、有限正计时、逐 step token 数、数据和相关代码的校验和。
相同文件与 seed 加 token 数检查用于发现明显配对错误，不能代替 loss/梯度数值对齐验证。

### 看结果后再运行下一阶段

下面以 4096 为示例，实际按 screen 报告选择；分别运行，不能把三条命令当成必跑清单。

```bash
# 若出现收益，复验一个候选。先 unset REPORT_DIR，让脚本为新阶段建立新目录。
unset REPORT_DIR CHUNK_SIZES STEPS TRAIN_SEED
CHUNK_SIZE=4096 bash scripts/training/sft/tools/run_chunkflow_8gpu.sh confirm

# 若仍然变慢，单独采样。也可以选报告中的代表 step，例如 PROFILE_START=27 PROFILE_END=29。
CHUNK_SIZE=4096 bash scripts/training/sft/tools/run_chunkflow_8gpu.sh profile
```

`summary.md` 包含每个候选的总体加速比、95% block bootstrap 区间、step_time 加速比、
四格表和相关系数。逐步 CSV、分组诊断、散点图保存在 `repeatN/chunkSIZE/`。
筛选时共享的 baseline 不构成独立样本；区间不能消除机器状态差异。
`manifest.json` 保存有效配置、代码版本、相关代码哈希、数据哈希、各组日志路径和 ClearML 链接。

profile 阶段不生成收益报告，避免把 profiler 开销当成真实收益。
时间轴由训练 worker 写到其工作目录下的 `traces/<RUN_NAME>/train_trace/*.pt.trace.json.gz`，
每组确切的相对目录在 `profile.md` 和 `manifest.json` 中；不会自动复制到报告目录。

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
