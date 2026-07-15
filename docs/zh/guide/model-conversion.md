# 模型 Checkpoint 转换

在训练完成后把 Relax checkpoint 转换为 Hugging Face 格式，并可在导出过程中直接量化为 FP8。

## 概述

Relax 使用 Megatron torch distributed checkpoint（DCP）格式保存训练 checkpoint。在部署或发布训练后的模型前，可使用 `scripts/tools/convert_torch_dist_to_hf_bridge.py` 将其导出为 Hugging Face safetensors。

这是 checkpoint 后置处理流程，不会改变训练阶段使用的精度或执行模式。

| 输入 | 输出 | 工具 |
| --- | --- | --- |
| Megatron DCP | 标准 HF safetensors | `convert_torch_dist_to_hf_bridge.py` |
| Megatron DCP | FP8 HF safetensors | `convert_torch_dist_to_hf_bridge.py --fp8` |
| BF16/FP16/FP32 HF safetensors | FP8 HF safetensors | `convert_hf_to_fp8.py` |

## 前置条件

- 在 Relax 仓库根目录执行命令。
- 当前环境必须能够导入 Megatron-LM 和 Megatron Bridge。
- `--origin-hf-dir` 必须指向原始 HF 模型目录。Bridge 使用它读取模型结构；流式 FP8 导出还要求目录中包含 safetensors 权重，以获得预期 HF key 映射。
- FP8 转换默认使用 CUDA，因此需要支持 CUDA 的 PyTorch 环境。

`convert_torch_dist_to_hf_bridge.py` 会自动把 Relax 仓库根目录加入 `sys.path` 和 `PYTHONPATH`，无需手动配置 Relax 路径。

## 将 Megatron DCP 导出为 HF

```bash
python scripts/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/torch_dist_checkpoint \
  --origin-hf-dir /path/to/original_hf_model \
  --output-dir /path/to/output_hf
```

| 参数 | 说明 |
| --- | --- |
| `--input-dir` | Megatron DCP checkpoint 根目录或单个 checkpoint 目录。 |
| `--origin-hf-dir` | 用于读取模型结构和权重映射的原始 HF 模型目录。 |
| `--output-dir` | 输出 HF checkpoint 目录。 |
| `-f`, `--force` | 允许输出目录已存在。 |

如果原始 HF 目录中存在 `tokenizer_config.json`、`vocab.json` 和 `merges.txt`，脚本也会复制这些文件。如果原始配置开启了 MTP，但 DCP checkpoint 没有 MTP 权重，导出脚本会检测并在导出时关闭 MTP。

## 流式导出 FP8

添加 `--fp8` 后，每个 HF tensor 会在 Megatron Bridge 导出时立即量化，不会写出中间 BF16 HF checkpoint。

```bash
python scripts/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/torch_dist_checkpoint \
  --origin-hf-dir /path/to/original_hf_model \
  --output-dir /path/to/output_fp8 \
  --fp8 \
  --fp8-strategy block \
  --fp8-block-size 128 128 \
  --fp8-device cuda \
  --fp8-max-shard-size-mb 4096
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--fp8` | `false` | 在 Bridge 导出过程中启用 FP8 转换。 |
| `--fp8-strategy` | `block` | 量化策略：`block`、`channel` 或 `tensor`。 |
| `--fp8-block-size` | `128 128` | `block` 策略的 block 形状；不能与 `channel` 或 `tensor` 一起使用。 |
| `--fp8-device` | `cuda` | 逐 tensor 或 expert slice 执行量化的设备。 |
| `--fp8-max-shard-size-mb` | `4096` | 输出 shard 的目标大小，单位 MiB；单个转换 tensor group 可以超过该值。 |

流式 hook 位于 Bridge HF tensor generator 和 safetensors writer 之间。使用 `--fp8-device cuda` 时，GPU 显存峰值大致是一份二维权重或单个 expert slice，加上量化 workspace。Bridge 仍会在 CPU 上构建并加载完整 BF16 Megatron 模型，因此源 checkpoint 的加载过程本身不是流式的。

writer 会先在临时目录中暂存权重 shard 和 `model.safetensors.index.json`，再替换输出文件。如果替换期间出现可捕获异常或 `KeyboardInterrupt`，writer 会尝试恢复旧权重和索引。这不是跨文件原子提交，无法防护 `SIGKILL` 或掉电。

::: warning 输出目录
开启 `--fp8` 时，`--output-dir` 必须与 `--origin-hf-dir` 不同，包括最终解析到同一目录的路径。
:::

::: warning Scale 格式
流式转换写出标准 FP32 scale tensor。当前没有实现打包的 UE8M0 scale，因此这条路径不提供 `--scale-fmt`。
:::

### FP8 输出布局

- 量化后的 `*.safetensors` shard 和 `model.safetensors.index.json`。
- 包含自动生成 `quantization_config` 的 `config.json`。
- block 量化写出 `.weight_scale_inv`，channel 和 tensor 量化写出 `.weight_scale`。
- embedding、norm、router、`lm_head`、visual 模块和部分 gate 等不适合量化的权重保持原 dtype，并记录到量化配置中。
- fused MoE expert tensor 会在转换过程中拆成逐 expert 的 HF 权重。

## 将已有 HF Checkpoint 转换为 FP8

如果输入已经是 BF16、FP16 或 FP32 HF safetensors checkpoint，使用离线转换脚本：

```bash
python scripts/tools/convert_hf_to_fp8.py \
  --model-dir /path/to/input_hf \
  --save-dir /path/to/output_fp8 \
  --strategy block \
  --block-size 128 128 \
  --max-workers 1
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-dir` | — | 输入 HF safetensors 目录。 |
| `--save-dir` | — | 输出目录。 |
| `--strategy` | `block` | 量化策略：`block`、`channel` 或 `tensor`。 |
| `--block-size` | — | 使用 `block` 时必须提供两个正整数。 |
| `--max-workers` | `1` | 并发处理的输入 shard 数量。 |
| `--scale-fmt` | `None` | 仅作兼容元数据。`ue8m0` 不会打包或改变 FP32 scale tensor。 |

离线转换器会保留一个输入 shard 的全部转换结果，直到该 shard 写盘。增大 `--max-workers` 会增加 GPU 显存占用；显存有限时保持为 `1`。

## 启动转换后的 FP8 模型

生成的 `config.json` 可以让 SGLang 自动识别 FP8，不必显式指定 `--quantization fp8`。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m sglang.launch_server \
  --model-path /path/to/output_fp8 \
  --tp-size 8 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --mem-fraction-static 0.85
```

项目 Docker 镜像当前使用 `lmsysorg/sglang:v0.5.12.post1-cu129`。如果启动时 OOM，可把 `--mem-fraction-static` 下调到 `0.8` 或 `0.75`。

## 故障排除

### 输出目录已存在

选择新的目录或添加 `--force`。FP8 导出时，不要把输出指向原始 HF 目录。

### CUDA 不可用

当 `torch.cuda.is_available()` 为 false 时，导出脚本会拒绝 CUDA `--fp8-device`。请在 CUDA 环境中执行，或显式选择其他受支持的设备。

### 单个 shard 超过目标大小

`--fp8-max-shard-size-mb` 是目标值而不是硬限制。单个 tensor group，尤其是 fused expert group，不会跨 writer group 拆分，因此可能产生更大的 shard。

## 下一步

- [快速上手](./quick-start.md) — 训练并导出模型。
- [外部模型接入](./external-model-integration.md) — 为新模型架构添加 Bridge 映射。
- [Distributed Checkpoint](./distributed-checkpoint.md) — 了解 Relax checkpoint 的存储和同步。
