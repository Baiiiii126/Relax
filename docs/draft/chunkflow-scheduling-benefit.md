# ChunkFlow 核心实现与运行

## 代码结构

核心实现位于 `relax/backends/megatron/chunkflow/`：

| 模块 | 职责 |
| --- | --- |
| `scheduler.py`、`units.py` | 切分、装箱及执行单元排序 |
| `schedules.py`、`pipeline_schedules.py` | 单阶段与流水线前后向 |
| `attention_ops.py`、`flash_backends.py` | 分块注意力和 FA2/FA3 适配 |
| `state_store.py`、`runtime.py` | KV 缓存、生命周期及运行时指标 |
| `unit_batch.py`、`rope.py` | 训练批次切片、目标与位置对齐 |
| `integration.py`、`gdn.py` | 注意力安装与 GDN 跨块状态、梯度衔接 |
| `schedule_model.py`、`telemetry.py` | 调度预测及计划指标 |

启用 `--use-chunkflow` 后，每个训练 step 都使用 ChunkFlow。
核心训练不依赖离线分析或批量实验脚本。

## 手动训练入口

保留 `scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh`，支持
`baseline`、`chunkflow` 两种模式；通过现有 `ray-job.sh` 提交。
以下示例会实际提交训练，应在自己的独占空闲 8 卡环境中运行。
数据须已准备好，含 `question`、`answer` 列；脚本不生成数据。

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/train.parquet
export TP_SIZE=2 PP_SIZE=4 GBS=8
export SEQ_LENGTH=32768 MAX_TOKENS_PER_GPU=32768
export NUM_ROLLOUT=100 TRAIN_SEED=1235
export CHUNK_SIZE=4096 PACK_POLICY=staggered KEEP_LAST_K=0
export RELAX_CHUNKFLOW_FLASH_BACKEND=fa3
export RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY=1

# 选择一种模式执行。
bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh chunkflow
```

对照模式将最后的 `chunkflow` 换为 `baseline`。
默认 TP2/PP4、GBS8、chunk4096、FA3、100 步、seed1235，一次只提交一组。
同步结束后检查完成步数、安装层数及后端日志；这不代替 loss/梯度的数值对齐验证。
`INPUT_KEY`、`LABEL_KEY` 可覆盖数据列名，默认 `question`、`answer`。
`RAY_NO_WAIT=1` 只提交任务，不检查训练完成；默认同步等待。
FA3 需要训练 worker 镜像中提供兼容实现；可显式设 `RELAX_CHUNKFLOW_FLASH_BACKEND=fa2`。

## 正确性约束

- `staggered` 为当前训练入口默认策略。多依赖段的 `lockstep` RoPE 尚不支持。
- `KEEP_LAST_K=0` 保留全部 chunk 激活；正数启用部分 Fp/Fr 重计算。
- 同一依赖组必须完成前向，再逆序反向；KV 叶子梯度通过 hook 回填，GDN 状态梯度通过 autograd 桥接回填。
- RoPE 使用序列内偏移，不能加 packed batch 行偏移；入口禁用 RoPE fusion。

## 指标与计时

调度模型使用声明的相对成本：F 与填充行数成正比，B 为 F 的两倍，网络延迟为零。
它预测调度收益，不是 GPU 耗时预测，也不是实测加速上界。

`RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY=1` 将详细计划指标留在本地日志，
常规训练与性能指标继续上报 ClearML。脚本默认设为 1，避免诊断指标同步上报影响计时。
`RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=1` 默认启用普通 ChunkFlow 的理论调度指标；
将其设为 0 可关闭这部分模拟。预测仅用于诊断，不控制是否执行 ChunkFlow。

`actor_train_time` 统计训练路径；`step_time` 还包含数据等待等环节，
不包含整个作业的启动和模型加载。加速比用相同输入的总耗时比计算，
而非逐步加速比的算术平均。Profiler 计时不用于判定收益。

## 测试

纯 CPU 切分、计划指标与调度预测检查：

```bash
python3 tests/backends/megatron/test_chunkflow_cpu.py
```

其余核心正确性测试位于 `tests/backends/megatron/test_chunkflow_*.py`，
其中注意力、梯度和模型集成测试需要相应的 PyTorch、Megatron 或 GPU 环境。

## Qwen3.6 GDN 对齐（第一阶段）

GDN 适配复用 Megatron 原始投影、门控和归一化，只包装卷积与递推的状态接口。
每条长序列分别缓存卷积末状态和递推末状态；后续 chunk 的状态梯度按逆序反向传回。
Fp/Fr 与 GDN 自身的 checkpoint 重放复用相同的缓存叶子。尾块和卷积对齐 padding 不写入真实序列的状态。

当前要求 CP=1、packed THD、`staggered`、非 deterministic 模式，关闭 GDN 预处理融合。
Megatron 须提供 `GatedDeltaNet._forward_compute`，FLA 的两个算子须支持初始状态和末状态梯度。
本次核对的训练环境为 FLA 0.4.2。

混合模型的 full-layer recompute 尚未支持，安装时会报错；可用 selective GDN 重计算或 ChunkFlow 的 Fp/Fr。
原有 Qwen3.6 训练脚本包含 CP/VPP/full recompute 配置，不能直接加 `--use-chunkflow` 当作已适配入口。

在已同步代码的训练容器、仓库根目录运行：

```bash
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
export PYTHONPATH="${MEGATRON}:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m pytest tests/backends/megatron/test_chunkflow_gdn.py -q -rs
```

CPU 检查用独立 Torch 卷积和递推计算比较整段与分块的输出、输入梯度、参数梯度，
覆盖仅末块有 loss、重计算、多序列隔离和 padding。fixture 模拟 Megatron 的 GDN 模块入口；
这些状态桥接检查不依赖真实 Megatron 实现。训练仍需要镜像提供兼容的 Megatron，
其路径由训练入口设置；直接运行 pytest 时需按上面命令设置。
CUDA 检查使用真实 FLA 卷积与 GDN 算子、bf16 小型投影测试壳，覆盖 Fp/Fr 与仅末块 loss；
按张量最大幅值设置 2% 相对/绝对容差。未检测到 GPU 时显式跳过。
通过该检查仅代表 GDN 状态接口对齐；整模型、MoE、TP/PP 与优化器 step 仍须分别验证。

### Qwen3.6-35B-A3B 8 卡对照

入口为 `scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh`。
固定 TP2/PP4/CP1/EP1、bf16、selective GDN 重计算和优化器 CPU offload；
默认 GBS8、100 步、32K 长度上限、chunk4096、保留全部 chunk 激活。
两组均按 2048 token 分块计算 logits，降低大词表带来的显存峰值。
这些配置已做静态及模拟提交检查，尚未完成真实 8 卡模型训练验证，显存是否足够需以实际卡型和数据为准。

先完成上面的 FLA CUDA 对齐检查，再在现有 Ray 环境中逐组运行：

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/train.parquet
export TRAIN_SEED=1235 NUM_ROLLOUT=100

bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh baseline
bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh chunkflow
```

默认读取 `${MODEL_DIR}/Qwen3.6-35B-A3B`；`HF_CHECKPOINT` 可直接指定 checkpoint 目录。
数据使用 `question` / `answer` 列，支持 `INPUT_KEY` / `LABEL_KEY` 覆盖。
须包含实际分词后超过 4096 token 的样本，才能验证跨 chunk 状态；超过 `SEQ_LENGTH` 的样本会被跳过。
`PROMPT_DATA` 未设置时读取 `${DATA_DIR}/chunkflow-mixed/train.parquet`。

每次只运行一组，不带 profiler。日志位于 `log/${RUN_NAME}.log`，ClearML 默认项目为
`Relax/sft/chunkflow-qwen36`。同步模式会检查完成步数以及 GDN、普通注意力两类安装日志；
`RAY_NO_WAIT=1` 仅提交，不检查最终完成状态。FA3 不可用时可设置 `RELAX_CHUNKFLOW_FLASH_BACKEND=fa2`。
调度预测默认关闭；需要时设置 `RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=1`，其成本假设仍未按 GDN 校准。
