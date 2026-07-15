# Model Checkpoint Conversion

Convert a trained Relax checkpoint to Hugging Face format as a post-training step, with optional FP8 quantization during export.

## Overview

Relax saves Megatron training checkpoints in torch distributed checkpoint (DCP) format. Before serving or publishing a trained model, use `scripts/tools/convert_torch_dist_to_hf_bridge.py` to export it to Hugging Face safetensors.

This is a checkpoint post-processing workflow and does not change the precision or execution mode used during training.

| Source | Output | Tool |
| --- | --- | --- |
| Megatron DCP | Standard HF safetensors | `convert_torch_dist_to_hf_bridge.py` |
| Megatron DCP | FP8 HF safetensors | `convert_torch_dist_to_hf_bridge.py --fp8` |
| BF16/FP16/FP32 HF safetensors | FP8 HF safetensors | `convert_hf_to_fp8.py` |

## Prerequisites

- Run commands from the Relax repository root.
- Megatron-LM and Megatron Bridge must be importable in the current environment.
- `--origin-hf-dir` must point to the original HF model directory. Bridge uses it for the architecture; streaming FP8 export additionally requires safetensors weights for the expected HF key map.
- FP8 conversion defaults to CUDA and therefore requires a CUDA-enabled PyTorch environment.

`convert_torch_dist_to_hf_bridge.py` automatically prepends the Relax repository root to `sys.path` and `PYTHONPATH`; no manual Relax path setup is required.

## Export Megatron DCP to HF

```bash
python scripts/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/torch_dist_checkpoint \
  --origin-hf-dir /path/to/original_hf_model \
  --output-dir /path/to/output_hf
```

| Flag | Description |
| --- | --- |
| `--input-dir` | Megatron DCP checkpoint root or a single checkpoint directory. |
| `--origin-hf-dir` | Original HF model directory used for model structure and weight mapping. |
| `--output-dir` | Output HF checkpoint directory. |
| `-f`, `--force` | Allow an existing output directory. |

The script also copies `tokenizer_config.json`, `vocab.json`, and `merges.txt` from the original HF directory when present. If the original configuration enables MTP but the DCP checkpoint has no MTP weights, the exporter detects this and disables MTP for export.

## Streaming FP8 Export

Add `--fp8` to quantize each HF tensor as Megatron Bridge exports it. No intermediate BF16 HF checkpoint is written.

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

| Flag | Default | Description |
| --- | --- | --- |
| `--fp8` | `false` | Enable FP8 conversion during Bridge export. |
| `--fp8-strategy` | `block` | Quantization strategy: `block`, `channel`, or `tensor`. |
| `--fp8-block-size` | `128 128` | Block shape for `block`; invalid with `channel` or `tensor`. |
| `--fp8-device` | `cuda` | Device used to quantize one tensor or expert slice at a time. |
| `--fp8-max-shard-size-mb` | `4096` | Target output shard size in MiB. One converted tensor group may exceed it. |

The streaming hook runs between the Bridge HF tensor generator and the safetensors writer. With `--fp8-device cuda`, GPU memory is approximately bounded by one two-dimensional weight, or one expert slice, plus quantization workspace. Bridge still constructs and loads the complete BF16 Megatron model on CPU, so loading the source checkpoint itself is not streaming.

The writer stages weight shards and `model.safetensors.index.json` in a temporary directory before replacing the output files. If a caught exception or `KeyboardInterrupt` occurs during replacement, it attempts to restore the previous weight files and index. This is not a cross-file atomic commit and cannot protect against `SIGKILL` or power loss.

::: warning Output directory
When `--fp8` is enabled, `--output-dir` must differ from `--origin-hf-dir`, including paths that resolve to the same directory.
:::

::: warning Scale format
Streaming conversion writes standard FP32 scale tensors. Packed UE8M0 scales are not implemented, so this path does not provide `--scale-fmt`.
:::

### FP8 output layout

- Quantized `*.safetensors` shards and `model.safetensors.index.json`.
- `config.json` with a generated `quantization_config`.
- Block quantization writes `.weight_scale_inv`; channel and tensor quantization write `.weight_scale`.
- Non-quantizable weights such as embeddings, norms, routers, `lm_head`, visual modules, and selected gates remain in their original dtype and are recorded in the quantization configuration.
- Fused MoE expert tensors are split into per-expert HF weights during conversion.

## Convert an Existing HF Checkpoint to FP8

Use the offline converter when the source is already a BF16, FP16, or FP32 HF safetensors checkpoint:

```bash
python scripts/tools/convert_hf_to_fp8.py \
  --model-dir /path/to/input_hf \
  --save-dir /path/to/output_fp8 \
  --strategy block \
  --block-size 128 128 \
  --max-workers 1
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model-dir` | — | Source HF safetensors directory. |
| `--save-dir` | — | Output directory. |
| `--strategy` | `block` | Quantization strategy: `block`, `channel`, or `tensor`. |
| `--block-size` | — | Exactly two positive integers when using `block`. |
| `--max-workers` | `1` | Number of source shards processed concurrently. |
| `--scale-fmt` | `None` | Compatibility metadata only. `ue8m0` does not pack or change the FP32 scale tensors. |

The offline converter keeps all converted tensors for one source shard until that shard is saved. Increasing `--max-workers` therefore increases GPU memory use; keep it at `1` when memory is limited.

## Serve the Converted FP8 Model

The generated `config.json` lets SGLang detect FP8 automatically, so `--quantization fp8` does not need to be specified explicitly.

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

The project Docker image currently uses `lmsysorg/sglang:v0.5.12.post1-cu129`. If startup runs out of memory, lower `--mem-fraction-static` to `0.8` or `0.75`.

## Troubleshooting

### Output directory already exists

Choose a new directory or pass `--force`. With FP8 export, do not point the output at the original HF directory.

### CUDA is unavailable

The exporter rejects a CUDA `--fp8-device` when `torch.cuda.is_available()` is false. Run the conversion in a CUDA environment or select another supported device explicitly.

### A shard exceeds the target size

`--fp8-max-shard-size-mb` is a target rather than a hard limit. A single tensor group, especially a fused expert group, is never split across writer groups and may produce a larger shard.

## Next Steps

- [Quick Start](./quick-start.md) — Train and export a model.
- [External Model Integration](./external-model-integration.md) — Add Bridge mappings for a new architecture.
- [Distributed Checkpoint](./distributed-checkpoint.md) — Understand Relax checkpoint storage and synchronization.
