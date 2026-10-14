# ChunkFlow loss alignment memo

## Goal

Make PP=4 ChunkFlow produce the same deterministic step loss and gradients as
the non-ChunkFlow baseline before enabling performance features.

## 2026-09-23 findings

1. The unit slicer used the packed batch row offset as part of the RoPE
   position. Baseline packed sequences restart RoPE at zero, so only
   `Segment.prefix` is valid.
2. ChunkFlow was installed on the bare model, but schedules attached a second
   runtime to the Float16/DDP wrapper. Attention stores and schedule cleanup
   therefore operated on different objects.
3. A dependent group's complete StateStore was released after its highest
   chunk backward. Fp/Fr recomputation of earlier chunks still needs that
   store; release is valid only after the index-0 backward.
4. A single cumulative `cu_seqlens_*_padded` cannot encode independent
   non-zero RoPE offsets for several segments in one lockstep unit. Until an
   explicit per-segment RoPE path exists, lockstep dependent units must fail
   closed rather than silently train with wrong positions.
5. Unit losses are additive only on the current `--calculate-per-token-loss`
   path. Sample-mean loss needs original sample identity and denominators to be
   preserved across chunks.
6. The per-token normalizer was still not additive for prompt-only chunks.
   `get_cp_local_num_tokens()` clamps every sample's mask sum to at least one,
   but a ChunkFlow unit is a slice rather than an independent sample. Applying
   that clamp to every unit inflated the step denominator and made both the
   reported loss and normalized gradients smaller than the baseline. Unit
   batches now carry their exact (unclamped) mask sum through
   `__num_tokens_override__`.

## 2026-09-23 validation

- Isolated GPU unit suite: 485 passed, 42 skipped.
- Full ChunkFlow test suite, including the PP schedule simulator: passed.
- The PP schedule simulator now covers the single-unit, PP=4 control case
  instead of skipping every `num_units < pipeline_size` case.
- PP=4 one-step baseline completed on Qwen3-4B with the fixed sample
  `sample_idx=626`, `len=353`, `learn_tokens=200`:
  `loss=1.8637005615234374`, `grad_norm=74.20682525634766`.
- Control and forced-chunk runs remain to be executed serially against this
  baseline. The remote execution request was declined before the control run,
  so no alignment claim is made yet.

## Validation ladder

The standalone Torch forward/backward oracle has already aligned and has been
removed. Continue validation in Relax:

1. Capture one deterministic baseline step: small model, Chinese samples,
   dropout 0, PP=4, fixed order. Save input tensors, logits/loss, parameter
   gradients, and post-step weights.
2. Enable forced chunking with `staggered` and `keep_last_k=0`; compare forward,
   then backward, then one optimizer step.
3. Enable Fp/Fr recomputation (`keep_last_k=1`) and repeat.
4. Implement explicit per-segment RoPE positions, add a multi-segment oracle,
   then re-enable `lockstep`.
5. Restore TP, sequence parallel, dynamic batching, bf16, and fused kernels one
   at a time. PP=4 is held constant in both arms throughout.

Run the two deterministic arms with the same script:

```bash
bash scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh baseline
bash scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh chunkflow
```

If `chunkflow` differs from `baseline`, inspect input identity, prefix RoPE,
q-short/kv-long causal attention, and unit loss slicing.

## Current launch verdict

UNSAFE until the code fixes pass unit tests and a clean GPU golden-sample run.
The 4B script defaults to the final 8-card topology (`TP=2`, `PP=4`, `DP=1`),
but accepts `TP_SIZE=2 PP_SIZE=2` for a 4-card correctness smoke. Both arms
disable fused RoPE and keep `staggered` with `keep_last_k=0`. Relax currently
forces Megatron training to bf16 in its argument defaults; changing that
requires a separately approved argument-parser change.
