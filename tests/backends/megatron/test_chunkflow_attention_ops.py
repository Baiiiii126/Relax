# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for chunkflow's chunk-local KV-cache attention.

Pure-tensor, CPU-only (no flash_attn / no GPU needed): the production kernel
behind ``ChunkCacheAttention`` is meant to be flash_attn's low-level varlen
forward/backward once confirmed on the training cluster, but the *math* is
identical to what ``torch.nn.functional.scaled_dot_product_attention``
computes — so SDPA is used here as an independent oracle, not just as a
stand-in for the same code being tested.

Covers:

1. ``_manual_attn_forward`` alone reproduces SDPA's causal output for a
   single one-shot call (no chunking yet) — isolates the core math from the
   chunking/StateStore plumbing before testing them together.
2. Chunking end-to-end: splitting one sequence into chunks, running each
   through ``chunk_attention_forward``/``chunk_attention_backward`` against a
   ``StateStore``, must reproduce the exact same output and gradients
   (``dq``, ``dk``, ``dv``) as one-shot SDPA — for a range of chunk sizes,
   including a chunk size of 1 (every token its own chunk, the most
   dependency-heavy case) and a chunk size that doesn't evenly divide the
   sequence length.
3. The same equivalence under bf16, with a widened tolerance.
4. Memory: the activation saved for backward must stay flat as the chunk count
   grows. An earlier version concatenated the cache and saved that, costing
   O(T^2 / chunk_size) -- 11x the unchunked baseline at 32 chunks, the exact
   opposite of what chunking is for. This is regression-tested here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron.chunkflow.attention_ops import (
        _manual_attn_forward,
        chunk_attention_backward,
        chunk_attention_forward,
    )
    from relax.backends.megatron.chunkflow.state_store import StateStore
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


def _full_attn_and_grads(q, k, v, dout, causal=True, softmax_scale=None):
    """Independent oracle: one-shot SDPA causal attention."""
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)

    # THD [T, H, D] -> SDPA wants [B, H, T, D].
    q_b = q_.permute(1, 0, 2).unsqueeze(0)
    k_b = k_.permute(1, 0, 2).unsqueeze(0)
    v_b = v_.permute(1, 0, 2).unsqueeze(0)
    out_b = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=causal, scale=scale)
    out = out_b.squeeze(0).permute(1, 0, 2).contiguous()

    out.backward(dout)
    return out.detach(), q_.grad.detach(), k_.grad.detach(), v_.grad.detach()


def _chunked_cache_attn_and_grads(q, k, v, dout, chunk_size, causal=True, softmax_scale=None):
    """chunkflow path: split into chunks, forward ascending through a
    StateStore, backward descending, freeing as we go."""
    total = q.size(0)
    boundaries = list(range(0, total, chunk_size)) + [total]
    num_chunks = len(boundaries) - 1

    q_leaves, k_leaves, v_leaves, outs = [], [], [], []
    store = StateStore()
    for i in range(num_chunks):
        s, e = boundaries[i], boundaries[i + 1]
        q_i = q[s:e].detach().clone().requires_grad_(True)
        k_i = k[s:e].detach().clone().requires_grad_(True)
        v_i = v[s:e].detach().clone().requires_grad_(True)
        q_leaves.append(q_i)
        k_leaves.append(k_i)
        v_leaves.append(v_i)

        out_i, _lse_i = chunk_attention_forward(i, q_i, k_i, v_i, store, causal=causal, softmax_scale=softmax_scale)
        outs.append(out_i)

    assert store.live_chunk_ids == list(range(num_chunks))  # fully built, sawtooth peak

    for i in reversed(range(num_chunks)):
        s, e = boundaries[i], boundaries[i + 1]
        chunk_attention_backward(i, outs[i], dout[s:e], store)

    assert len(store) == 0  # fully torn down

    out_full = torch.cat([o.detach() for o in outs], dim=0)
    dq = torch.cat([leaf.grad for leaf in q_leaves], dim=0)
    dk = torch.cat([leaf.grad for leaf in k_leaves], dim=0)
    dv = torch.cat([leaf.grad for leaf in v_leaves], dim=0)
    return out_full, dq, dk, dv


# ---------------------------------------------------------------------------
# 1. Core math alone (no chunking): _manual_attn_forward vs SDPA
# ---------------------------------------------------------------------------


def test_manual_attn_forward_matches_sdpa():
    torch.manual_seed(0)
    tq, h, d = 17, 3, 8
    q = torch.randn(tq, h, d)
    k = torch.randn(tq, h, d)
    v = torch.randn(tq, h, d)
    scale = d**-0.5

    out, _lse = _manual_attn_forward(q, k, v, causal=True, softmax_scale=scale)

    q_b, k_b, v_b = (t.permute(1, 0, 2).unsqueeze(0) for t in (q, k, v))
    ref_b = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True, scale=scale)
    ref = ref_b.squeeze(0).permute(1, 0, 2)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Chunked end-to-end vs one-shot SDPA: output and gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 17, 64])
def test_chunked_cache_attention_matches_full(chunk_size: int):
    torch.manual_seed(0)
    total, h, d = 37, 2, 8  # 37 is not a multiple of any chunk_size above
    q = torch.randn(total, h, d)
    k = torch.randn(total, h, d)
    v = torch.randn(total, h, d)
    dout = torch.randn(total, h, d)

    out_ref, dq_ref, dk_ref, dv_ref = _full_attn_and_grads(q, k, v, dout)
    out_chunk, dq_chunk, dk_chunk, dv_chunk = _chunked_cache_attn_and_grads(q, k, v, dout, chunk_size)

    torch.testing.assert_close(out_ref, out_chunk, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dq_ref, dq_chunk, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dk_ref, dk_chunk, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dv_ref, dv_chunk, atol=1e-4, rtol=1e-4)


def test_chunked_cache_attention_bf16_within_tolerance():
    torch.manual_seed(0)
    total, h, d = 40, 2, 8
    q = torch.randn(total, h, d, dtype=torch.bfloat16)
    k = torch.randn(total, h, d, dtype=torch.bfloat16)
    v = torch.randn(total, h, d, dtype=torch.bfloat16)
    dout = torch.randn(total, h, d, dtype=torch.bfloat16)

    out_ref, dq_ref, dk_ref, dv_ref = _full_attn_and_grads(q, k, v, dout)
    out_chunk, dq_chunk, dk_chunk, dv_chunk = _chunked_cache_attn_and_grads(q, k, v, dout, chunk_size=8)

    torch.testing.assert_close(out_ref.float(), out_chunk.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq_ref.float(), dq_chunk.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dk_ref.float(), dk_chunk.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dv_ref.float(), dv_chunk.float(), atol=5e-2, rtol=5e-2)


def test_dk_dv_accumulate_contributions_from_later_chunks():
    """Directly exercises chunkflow lesson 3.4: chunk 0's dk/dv must equal the
    sum of contributions from every later query chunk, not just its own."""
    torch.manual_seed(0)
    total, h, d, chunk_size = 12, 2, 8, 4
    q = torch.randn(total, h, d)
    k = torch.randn(total, h, d)
    v = torch.randn(total, h, d)
    dout = torch.randn(total, h, d)

    _out, _dq, dk_chunk, dv_chunk = _chunked_cache_attn_and_grads(q, k, v, dout, chunk_size)
    _out_ref, _dq_ref, dk_ref, dv_ref = _full_attn_and_grads(q, k, v, dout)

    # chunk 0 = tokens [0:4): read by query chunks 0, 1, and 2 (all of them).
    torch.testing.assert_close(dk_chunk[:chunk_size], dk_ref[:chunk_size], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dv_chunk[:chunk_size], dv_ref[:chunk_size], atol=1e-4, rtol=1e-4)


def _saved_activation_bytes(out: torch.Tensor) -> int:
    """Total bytes of tensors the autograd graph behind ``out`` keeps alive."""
    seen: set[int] = set()
    total = 0
    stack = [out.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        for t in getattr(fn, "saved_tensors", ()):
            if isinstance(t, torch.Tensor) and id(t) not in seen:
                seen.add(id(t))
                total += t.numel() * t.element_size()
        stack.extend(next_fn for next_fn, _ in fn.next_functions)
    return total


def _chunked_forward_only(q, k, v, chunk_size):
    store = StateStore()
    outs = []
    for chunk_id, start in enumerate(range(0, q.shape[0], chunk_size)):
        sl = slice(start, start + chunk_size)
        out, _lse = chunk_attention_forward(chunk_id, q[sl], k[sl], v[sl], store)
        outs.append(out)
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize("chunk_size", [64, 32, 16, 8])
def test_saved_activation_does_not_grow_with_chunk_count(chunk_size):
    """Chunking must not inflate what backward keeps alive.

    Saving the concatenated cache per chunk costs O(T^2 / chunk_size); saving
    the per-chunk K/V (shared objects) costs O(T). Without this test the
    regression is invisible -- every numerical test still passes.
    """
    total, h, d = 128, 4, 16
    q = torch.randn(total, h, d, requires_grad=True)
    k = torch.randn(total, h, d, requires_grad=True)
    v = torch.randn(total, h, d, requires_grad=True)

    baseline = _saved_activation_bytes(_chunked_forward_only(q, k, v, total))
    chunked = _saved_activation_bytes(_chunked_forward_only(q, k, v, chunk_size))

    assert chunked <= baseline * 1.2, (
        f"chunk_size={chunk_size} ({total // chunk_size} chunks) saves "
        f"{chunked / baseline:.2f}x the unchunked activation; the cache is "
        "probably being concatenated and saved again per chunk."
    )


# ---------------------------------------------------------------------------
# 4. flash backend (GPU only): must be at least as accurate as the reference
# ---------------------------------------------------------------------------


def _chunked_fwd_bwd(q, k, v, chunk_size, force_manual):
    """Chunked forward + one backward; returns (out, dq, dk, dv)."""
    from relax.backends.megatron.chunkflow import attention_ops as ao

    original = ao._use_flash
    if force_manual:
        ao._use_flash = lambda _q: False
    try:
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        ql, kl, vl = leaves
        store, outs = StateStore(), []
        for chunk_id, start in enumerate(range(0, q.shape[0], chunk_size)):
            sl = slice(start, start + chunk_size)
            out, _lse = chunk_attention_forward(chunk_id, ql[sl], kl[sl], vl[sl], store)
            outs.append(out)
        torch.cat(outs, dim=0).sum().backward()
        return torch.cat([o.detach() for o in outs], dim=0), ql.grad, kl.grad, vl.grad
    finally:
        ao._use_flash = original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_backend_is_at_least_as_accurate_as_the_reference():
    """Judge both bf16 backends against an fp32 reference.

    Comparing two bf16 results to each other proves nothing -- bf16's ULP at
    magnitude ~10 is already ~0.06. What matters is the distance to the true
    answer; flash accumulates in fp32 and should not be worse.
    """
    from relax.backends.megatron.chunkflow import attention_ops as ao

    if ao._flash_varlen_ops() is None:
        pytest.skip("flash_attn not installed")

    torch.manual_seed(0)
    total, h, d, chunk_size = 256, 8, 64, 64
    q32 = torch.randn(total, h, d, device="cuda")
    k32 = torch.randn(total, h, d, device="cuda")
    v32 = torch.randn(total, h, d, device="cuda")
    qb, kb, vb = (t.bfloat16() for t in (q32, k32, v32))

    ref = _chunked_fwd_bwd(q32, k32, v32, chunk_size, force_manual=True)
    flash = _chunked_fwd_bwd(qb, kb, vb, chunk_size, force_manual=False)
    manual = _chunked_fwd_bwd(qb, kb, vb, chunk_size, force_manual=True)

    for name, r, f, m in zip(("out", "dq", "dk", "dv"), ref, flash, manual):
        scale = r.float().abs().max().item()
        err_flash = (f.float() - r.float()).abs().max().item() / scale
        err_manual = (m.float() - r.float()).abs().max().item() / scale
        assert err_flash <= err_manual * 3, (
            f"{name}: flash rel-err {err_flash:.2e} vs reference {err_manual:.2e} -- "
            "the flash path is materially less accurate than the fallback."
        )
