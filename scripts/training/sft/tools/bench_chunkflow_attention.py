"""Attribute chunkflow's attention overhead, on the real Qwen3-4B TP=2 shapes.

Measures, on one packed THD batch shaped like the SFT A/B dataset:
  1. one flash_attn_varlen_func call over the whole batch  (what the baseline does)
  2. ChunkedCoreAttention over the same batch              (what chunkflow does)
  3. the individual costs chunkflow adds: GQA head widening and the per-chunk
     K/V concatenation, so the slowdown can be attributed rather than guessed.

Run on a free GPU:
  CUDA_VISIBLE_DEVICES=4 python tools/bench_chunkflow_attention.py
"""

from __future__ import annotations

import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, "/root/workspace/Relax")

from flash_attn import flash_attn_varlen_func

from relax.backends.megatron.chunkflow.integration import ChunkedCoreAttention


# Qwen3-4B under TP=2 (num_attention_heads 32 / num_query_groups 8 halve).
HEADS, KV_HEADS, HEAD_DIM = 16, 4, 128
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Length histogram of $DATA_DIR/gsm8k-mixed, bucketed at 512 -- the mix matters
# because chunkflow's cost scales with chunk count, not just total tokens.
LENGTH_BUCKETS = [(320, 271), (768, 292), (1536, 96), (2048, 159), (2560, 111), (2764, 45)]
CHUNK_SIZE = 512


def build_batch(repeat: int = 1):
    lengths = []
    for length, count in LENGTH_BUCKETS:
        lengths += [length] * (count * repeat)
    lengths = torch.tensor(lengths)
    total = int(lengths.sum())
    bounds = [0] + torch.cumsum(lengths, 0).tolist()

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    q = torch.randn(total, HEADS, HEAD_DIM, generator=gen, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total, KV_HEADS, HEAD_DIM, generator=gen, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total, KV_HEADS, HEAD_DIM, generator=gen, device=DEVICE, dtype=DTYPE)
    return q, k, v, bounds, lengths, total


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000  # ms


def main() -> None:
    q, k, v, bounds, lengths, total = build_batch()
    cu = torch.tensor(bounds, device=DEVICE, dtype=torch.int32)
    max_len = int(lengths.max())
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor(bounds, device=DEVICE, dtype=torch.int32))

    chunks = [(n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in lengths.tolist()]
    print(f"batch: {len(lengths)} sequences, {total} tokens")
    print(f"  chunks/sequence at chunk_size={CHUNK_SIZE}: mean {statistics.mean(chunks):.2f}, max {max(chunks)}")
    print(f"  total attention calls: baseline 1 (fused), chunkflow {sum(chunks)}\n")

    k_wide = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
    v_wide = v.repeat_interleave(HEADS // KV_HEADS, dim=1)

    def baseline():
        return flash_attn_varlen_func(q, k_wide, v_wide, cu, cu, max_len, max_len, causal=True)

    attn = ChunkedCoreAttention(config=SimpleNamespace(context_parallel_size=1), layer_number=1, chunk_size=CHUNK_SIZE)

    def chunkflow():
        return attn(q, k, v, packed_seq_params=packed)

    def widen_only():
        return k.repeat_interleave(HEADS // KV_HEADS, dim=1), v.repeat_interleave(HEADS // KV_HEADS, dim=1)

    def chunkflow_no_widen():
        return attn(q, k_wide, v_wide, packed_seq_params=packed)

    t_base = timeit(baseline)
    t_widen = timeit(widen_only)
    t_chunk = timeit(chunkflow)
    t_chunk_nowiden = timeit(chunkflow_no_widen)

    # Where does the 12x go? chunkflow makes ~2600 python-level calls where the
    # baseline makes one, so separate the per-call plumbing (StateStore, list
    # building, torch.cat) from the kernel time.
    from relax.backends.megatron.chunkflow.state_store import ChunkState, StateStore

    def cat_loop():
        """The per-chunk plumbing only -- no attention kernel at all.

        One StateStore per sequence: the store enforces ascending chunk ids
        starting at 0, and chunkflow's own integration does the same.
        """
        outs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end <= start:
                continue
            store = StateStore()
            for chunk_id, cs in enumerate(range(start, end, CHUNK_SIZE)):
                ce = min(cs + CHUNK_SIZE, end)
                store.put(chunk_id, ChunkState({"k": k[cs:ce], "v": v[cs:ce]}))
                states = store.lookup(chunk_id)
                outs.append(torch.cat([s.tensors["k"] for s in states], dim=0))
        return outs

    t_cat = timeit(cat_loop, warmup=1, iters=3)

    print(f"{'path':<44}{'ms':>10}{'vs baseline':>14}")
    print("-" * 68)
    print(f"{'baseline: 1 fused varlen call':<44}{t_base:>10.3f}{'-':>14}")
    print(f"{'chunkflow: full (incl. GQA widening)':<44}{t_chunk:>10.3f}{t_chunk / t_base:>13.2f}x")
    print(f"{'  chunkflow without widening':<44}{t_chunk_nowiden:>10.3f}{t_chunk_nowiden / t_base:>13.2f}x")
    print(f"{'  GQA widening alone':<44}{t_widen:>10.3f}{t_widen / t_base:>13.2f}x")
    print(f"{'  per-chunk plumbing: store + cat':<44}{t_cat:>10.3f}{t_cat / t_base:>13.2f}x")
    print()
    print(f"plumbing (store + cat), no kernel   {t_cat:.3f} ms   = {100 * t_cat / t_chunk_nowiden:.0f}% of chunkflow")
    print(f"kernels after plumbing              {t_chunk_nowiden - t_cat:.3f} ms")
    print(f"baseline kernels                    {t_base:.3f} ms")
    print(f"GQA widening                        {t_chunk - t_chunk_nowiden:+.3f} ms")
    print(f"total overhead vs baseline          {t_chunk - t_base:+.3f} ms  ({100 * (t_chunk - t_base) / t_base:+.1f}%)")

    print("\nchunk_size sweep (same batch, forward only):")
    print(f"{'chunk':>7}{'chunks/seq':>12}{'calls':>8}{'ms':>10}{'vs baseline':>13}")
    for cs in (256, 512, 1024, 2048, 4096):
        a = ChunkedCoreAttention(config=SimpleNamespace(context_parallel_size=1), layer_number=1, chunk_size=cs)
        n_calls = sum((int(n) + cs - 1) // cs for n in lengths)
        t = timeit(lambda a=a: a(q, k, v, packed_seq_params=packed), warmup=2, iters=5)
        per_seq = statistics.mean([(int(n) + cs - 1) // cs for n in lengths])
        print(f"{cs:>7}{per_seq:>12.2f}{n_calls:>8}{t:>10.3f}{t / t_base:>12.2f}x")

    # Decisive control: the SAME kernel, called once per sequence instead of once
    # per batch, with no chunkflow code in the loop at all. If this is already
    # several times slower than the fused call, the cost is per-call launch
    # overhead (and the fix is batching chunks across sequences), not chunkflow's
    # python.
    def baseline_per_sequence():
        outs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            n = end - start
            cu1 = torch.tensor([0, n], device=DEVICE, dtype=torch.int32)
            outs.append(
                flash_attn_varlen_func(
                    q[start:end], k_wide[start:end], v_wide[start:end], cu1, cu1, n, n, causal=True
                )
            )
        return outs

    t_per_seq = timeit(baseline_per_sequence, warmup=2, iters=5)
    print(f"\n{'one flash call per sequence (no chunkflow)':<44}{t_per_seq:>10.3f}{t_per_seq / t_base:>12.2f}x")
    print(f"  calls: {len(lengths)} vs 1 fused -- pure launch/occupancy cost")
    print(f"  per call: {1000 * t_per_seq / len(lengths):.1f} us")


if __name__ == "__main__":
    main()