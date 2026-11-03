# MoE expert tiles training experiment

This opt-in backend keeps sequence chunks at 8192. It changes expert GEMMs and,
optionally, forward dispatch scheduling. It does not change routing, top-k,
the original weighted activation, optimizer parameters or the ChunkFlow schedule.

- `EXPERT_BACKEND=tiles256` (default): fixed 256-row tiles for expert forward
  and input-gradient GEMMs. Weight-gradient reduction tiles remain unchanged.
  A GPU tile-prefix plan skips repeated scans of empty experts. Its two int32
  arrays use the same storage bytes as the previous int64 row offsets, and each
  forward retains its own plan until backward.
- `EXPERT_BACKEND=mixed`: GPU counts select full 256-row tiles, then 128/64-row
  tails in one grouped kernel. There is no fixed capacity padding or dropped token.
- Original TE Parameters and Megatron FP32 `main_grad` buffers remain in place.
  Device pointer tables avoid concatenating expert weights. GPU counts replace
  TE's Python expert split list. Parameter validation is cached against identity,
  storage and layout; pointer rebinding and dynamic gradient flags remain checked.
- `DISPATCH_OVERLAP=0` (default): one FC1 for all sources within each chunk,
  avoiding separate local/remote tail tiles. `DISPATCH_OVERLAP=1` attempts to run
  EP2 local-source FC1 while remote hidden states and
  FP64 probabilities are dispatched. A CUDA event joins before remote-source FC1.
  Activation, FC2 and backward each use the complete canonical expert/source order.
  This adds one FC1 launch. It is not a network tile-ready transport and does not
  reduce remote payload bytes. `DISPATCH_OVERLAP=0` retains the original dispatcher.
- Eligible ETP1 dropless, fused, non-graph dispatchers pack the EP input/output
  splits into one D2H copy and keep expert counts on the GPU. The original one
  event wait before all-to-all is retained. Each invocation owns its CPU split
  buffer for backward. Other configurations retain original metadata handling;
  installation logs report how many layers use compact metadata. Legacy metadata
  mode `a` retains its original handling; mode `b` is rejected because it does not
  return GPU expert counts.

Supported training path: bias-free TEGroupedMLP, BF16 operands, FP32 main gradients,
ETP1, dropless all-to-all, standard Megatron Parameters. Local-first additionally
requires EP2, FP64 router, permutation fusion and no shared-expert overlap.
FP8/FP4, FSDP, delayed weight gradients and activation offload are rejected.
CPU optimizer offload stays enabled in the recipe. Transport CUDA Graph capture is
unsupported. Incompatible requested paths fail instead of silently falling back.

## Validate and run

Use the existing environment and model/data paths. No dependency upgrades are needed.

```bash
export MEGATRON=/root/Megatron-LM
export PYTHONPATH="$PWD:$MEGATRON${PYTHONPATH:+:$PYTHONPATH}"
# Export HF_CHECKPOINT and PROMPT_DATA to the existing model directory and LongAlign file.
test -f "$HF_CHECKPOINT/config.json"
test -f "$PROMPT_DATA"

python3 -m pytest -q tests/backends/megatron/test_expert_adapter_cpu.py
python3 -m pytest -q tests/tools/test_expert_counts_replay.py tests/tools/test_summarize_qwen36_comparison.py
python3 -m pytest -q tests/tools/test_expert_kernel_cache.py
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/backends/megatron/test_expert_tiles.py

NUM_ROLLOUT=100 bash scripts/training/sft/run-qwen36-35B-A3B-8xgpu-expert-tiles.sh run
```

The launcher defaults to one eight-GPU `expert_chunkflow` job, with live terminal
logs. `EXPERIMENT_ARMS=expert_chunkflow` selects this explicitly. Use
`NUM_ROLLOUT=10` for a short diagnostic run or `NUM_ROLLOUT=100` for the full run.
Single-arm runs retain loss/timing logs and memory records without generating
a paired comparison. The offline cache test verifies that varying routed token
counts reuse compiled kernels; it does not execute GPU kernels.

Set `EXPERIMENT_ARMS=production` to run the following pair sequentially:

| Arm              | ChunkFlow | Expert backend | Local-first dispatch |
| ---------------- | --------- | -------------- | -------------------- |
| baseline         | off       | original TE    | off                  |
| expert_chunkflow | 8192      | tiles256       | off                  |

This pair measures the combined production gain against the original baseline.
Set `EXPERIMENT_ARMS=all` to additionally run `chunkflow` and `expert_baseline`
when separate attribution of backend and ChunkFlow gains is needed.

All arms use TP2/PP4/EP2/ETP1, GBS8, sequence32768, seed1235, BF16, FP64 router,
GDN recomputation on, and CPU optimizer offload/overlap on. Changing
`GDN_RECOMPUTE` applies to every arm; disabling it previously caused the baseline
to OOM on eight H20s. There is no automatic fallback and no loss/performance/memory
acceptance threshold. Failed or incomplete jobs are not valid speed measurements.
The ChunkFlow execution patch defaults on: nonblocking sequence metadata copies,
metadata reuse and removal of redundant GDN slices. Set
`RELAX_CHUNKFLOW_EXECUTION_OPT=0` to disable it in the selected ChunkFlow arms.

To compare the old local-first scheduling with the new default, explicitly enable
overlap. Keep the same backend and execution patch in both experiments:

```bash
EXPERIMENT_ARMS=optimized DISPATCH_OVERLAP=1 NUM_ROLLOUT=100 \
  bash scripts/training/sft/run-qwen36-35B-A3B-8xgpu-expert-tiles.sh run
```

## Inspect or replay

```bash
exp_dir="$(cat analysis/expert-tiles-latest-path.txt)"
cat "$exp_dir/comparison.md"
# Available after EXPERIMENT_ARMS=production or all:
# cat "$exp_dir/production-diagnosis.md"
```

Each pair also writes every loss/timing point to `*paired.csv`, and per-rank
allocator/device peaks to `*memory.csv`. Timing windows start at 0/10/30/50;
use the complete steady-state window, not the best window. Device memory includes
startup and is sampled every250 ms. Missing memory data means unmeasured, not zero.
With `EXPERIMENT_ARMS=optimized` or `all`, `experts-diagnosis.md` additionally
reports ChunkFlow gain with the same new backend.

Capture actual expert counts and a short timeline separately from speed runs:

```bash
NUM_ROLLOUT=40 bash scripts/training/sft/run-qwen36-35B-A3B-8xgpu-expert-tiles.sh profile
exp_dir="$(cat analysis/expert-tiles-latest-path.txt)"
find "$exp_dir" -name 'expert_counts_*.jsonl'
# Select one file printed above and a zero-based record within it.
python3 scripts/tools/benchmark_expert_tiles.py --counts-file "$COUNTS_FILE" \
  --counts-record 0 --block-m 256 --warmup 10 --steps 100 \
  --output-dir "analysis/expert-replay-$(date +%Y%m%d-%H%M%S)"
```

Replay uses real counts with synthetic operands and the standalone scan kernel;
it measures computation only, not the adapter's indexed plan or metadata path.
Inspect the timeline to establish whether communication and local FC1 actually
overlap on the installed NCCL/CUDA configuration. Multiple streams alone do not
prove overlap. Extra temporary communication storage and allocator lifetimes can
increase memory; only the full training comparison can establish a net speed or
memory benefit. Loss alignment and production improvement remain measurements,
not properties guaranteed by these patches.

The two-GPU tests cover empty experts, tails, original-parameter gradient hooks,
FP64 probability transport, zero splits, nondefault streams, reverse backward,
and a real TEGroupedMLP/MCore DDP/master-optimizer unit. They do not replace the
eight-GPU model test or validate the production CPU optimizer offload path.
