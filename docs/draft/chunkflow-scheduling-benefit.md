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
| `integration.py` | 安装注意力替换与恢复 baseline |
| `schedule_model.py`、`telemetry.py`、`gating.py` | 调度预测、计划指标及逐步门控 |

`relax/backends/megatron/model.py` 选择每步执行路径。
核心训练不依赖离线分析或批量实验脚本。

## 手动训练入口

保留 `scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh`，支持
`baseline`、`chunkflow`、`adaptive` 三种模式；通过现有 `ray-job.sh` 提交。
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
逐步门控将其换为 `adaptive`，并设置 `GATE_GAIN_PCT`。
默认 TP2/PP4、GBS8、chunk4096、FA3、100 步、seed1235，一次只提交一组。
同步结束后检查完成步数、安装层数、后端及门控日志；这不代替 loss/梯度的数值对齐验证。
`INPUT_KEY`、`LABEL_KEY` 可覆盖数据列名，默认 `question`、`answer`。
`RAY_NO_WAIT=1` 只提交任务，不检查训练完成；默认同步等待。
FA3 需要训练 worker 镜像中提供兼容实现；可显式设 `RELAX_CHUNKFLOW_FLASH_BACKEND=fa2`。

## 正确性约束

- `staggered` 为当前训练入口默认策略。多依赖段的 `lockstep` RoPE 尚不支持。
- `KEEP_LAST_K=0` 保留全部 chunk 激活；正数启用部分 Fp/Fr 重计算。
- 同一依赖组必须完成前向，再逆序反向；缓存叶子的梯度通过 hook 回填原计算图。
- RoPE 使用序列内偏移，不能加 packed batch 行偏移；入口禁用 RoPE fusion。
- 门控目前仅支持 text SFT、DP=CP=1、无 VPP、THD、staggered、K=0。
- `GATE_GAIN_PCT=140` 表示预测速度提升严格大于 140%，即加速比大于 2.4×。
  决策在当前步前后向开始前作出，所有 rank 先确认输入和决策一致。

## 指标与计时

调度模型使用声明的相对成本：F 与填充行数成正比，B 为 F 的两倍，网络延迟为零。
它预测调度收益，不是 GPU 耗时预测，也不是实测加速上界。

`RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY=1` 将详细计划指标留在本地日志，
常规训练与性能指标继续上报 ClearML。脚本默认设为 1，避免诊断指标同步上报影响计时。
`RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=1` 默认启用普通 ChunkFlow 的理论调度指标；
将其设为 0 可关闭这部分模拟，但不会关闭 adaptive 的决策预测。

`actor_train_time` 统计训练路径；`step_time` 还包含数据等待等环节，
不包含整个作业的启动和模型加载。加速比用相同输入的总耗时比计算，
而非逐步加速比的算术平均。Profiler 计时不用于判定收益。

## 测试

纯 CPU 计划、门控和 FA3 适配检查：

```bash
python3 tests/backends/megatron/test_chunkflow_cpu.py
```

其余核心正确性测试位于 `tests/backends/megatron/test_chunkflow_*.py`，
其中注意力、梯度和模型集成测试需要相应的 PyTorch、Megatron 或 GPU 环境。
