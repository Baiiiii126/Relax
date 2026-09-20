# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Chunk-local, KV-cache-aware causal attention for chunkflow.

Shapes follow the repo's THD convention: ``[T, H, D]`` (sequence, heads,
head-dim) — see e.g. ``relax/models/gemma4/attention.py``.

Within one chunk's forward, ``q`` is the *new* chunk's queries (length
``Tq``) and ``k_all``/``v_all`` are the full accumulated key/value up to and
including this chunk (length ``Tk >= Tq``) — the "q short, kv long" shape a
KV cache produces. The op returns ``(out, lse)``: ``lse`` is not consumed
anywhere in this module (it never flows forward into the rest of the
network), but is threaded through because (a) it is the sufficient statistic
needed to merge across KV blocks incrementally, and (b) real flash-attention
kernels need it to avoid materializing the full attention-probability matrix
in their backward.

Two backends sit behind ``ChunkCacheAttention``:

* flash_attn's low-level varlen forward/backward on CUDA with fp16/bf16 —
  the production path. The low-level pair is used (not
  ``flash_attn_varlen_func``) because only it exposes ``softmax_lse`` and
  lets us drive backward ourselves. Its ``causal=True`` aligns the mask to
  the bottom right when ``seqlen_q < seqlen_k``, which is exactly the
  "q short, kv long" semantics ``_causal_mask`` encodes.
* ``_manual_attn_forward``/``_manual_attn_backward``, a plain-torch
  reference implementing the identical math, used on CPU/fp32 so this module
  stays testable without flash_attn or a GPU. It materialises the
  ``[H, Tq, Tk]`` score matrix, so it is a correctness reference, not a
  production path.

Gradient accumulation across chunks is *not* done by hand here: every chunk
reads the same K/V tensors out of one ``StateStore``, so calling
``.backward()`` once per chunk, in descending order, lets ordinary autograd
``+=`` accumulation do the summing (chunkflow lesson 3.4) — what is manual is
*driving* those per-chunk backward calls in the right order and freeing the
StateStore entry right after each one (lesson 3.5/3.6), instead of a single
``.backward()`` over the whole sequence.

Memory: the op saves the per-chunk K/V, never their concatenation, so the
activation kept for backward is O(T) rather than O(T^2 / chunk_size). See
``ChunkCacheAttention`` for why, and the memory regression test in
``tests/backends/megatron/test_chunkflow_attention_ops.py``.
"""

from __future__ import annotations

import torch

from relax.backends.megatron.chunkflow.state_store import ChunkState, StateStore


def _manual_attn_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference causal-attention forward returning ``(out, lse)``.

    ``q``: ``[Tq, H, D]``. ``k``/``v``: ``[Tk, H, D]`` with ``Tk >= Tq``; when
    ``causal``, ``q``'s tokens are treated as the last ``Tq`` positions of the
    ``Tk``-token sequence (offset causal mask), matching the KV-cache shape.
    """
    tq, h, _d = q.shape
    tk = k.shape[0]
    q_h = q.permute(1, 0, 2)  # [H, Tq, D]
    k_h = k.permute(1, 0, 2)  # [H, Tk, D]
    v_h = v.permute(1, 0, 2)  # [H, Tk, D]

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * softmax_scale  # [H, Tq, Tk]
    if causal:
        scores = scores.masked_fill(_causal_mask(tq, tk, q.device), float("-inf"))

    lse_h = torch.logsumexp(scores, dim=-1)  # [H, Tq]
    probs = torch.exp(scores - lse_h.unsqueeze(-1))  # [H, Tq, Tk]
    out_h = torch.matmul(probs, v_h)  # [H, Tq, D]

    out = out_h.permute(1, 0, 2).contiguous()  # [Tq, H, D]
    lse = lse_h.permute(1, 0).contiguous()  # [Tq, H]
    del h
    return out, lse


def _manual_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference causal-attention backward, recomputing attention probabilities
    from the saved ``lse`` (no attention-probability matrix needs to survive
    from forward to backward — same trick real flash attention uses)."""
    tq, _h, _d = q.shape
    tk = k.shape[0]
    q_h = q.permute(1, 0, 2)
    k_h = k.permute(1, 0, 2)
    v_h = v.permute(1, 0, 2)

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * softmax_scale
    if causal:
        scores = scores.masked_fill(_causal_mask(tq, tk, q.device), float("-inf"))
    lse_h = lse.permute(1, 0)  # [H, Tq]
    probs = torch.exp(scores - lse_h.unsqueeze(-1))  # [H, Tq, Tk]

    dout_h = dout.permute(1, 0, 2)  # [H, Tq, D]
    dv_h = torch.matmul(probs.transpose(-1, -2), dout_h)  # [H, Tk, D]
    dprobs = torch.matmul(dout_h, v_h.transpose(-1, -2))  # [H, Tq, Tk]
    dscores = probs * (dprobs - (dprobs * probs).sum(dim=-1, keepdim=True))  # softmax backward
    draw = dscores * softmax_scale  # scores = (q @ k^T) * softmax_scale

    dq_h = torch.matmul(draw, k_h)  # [H, Tq, D]
    dk_h = torch.matmul(draw.transpose(-1, -2), q_h)  # [H, Tk, D]

    dq = dq_h.permute(1, 0, 2).contiguous()
    dk = dk_h.permute(1, 0, 2).contiguous()
    dv = dv_h.permute(1, 0, 2).contiguous()
    return dq, dk, dv


def _causal_mask(tq: int, tk: int, device: torch.device) -> torch.Tensor:
    """``[Tq, Tk]`` boolean mask, ``True`` where attention is forbidden.

    ``q``'s tokens are the last ``Tq`` positions of a ``Tk``-token sequence.
    """
    offset = tk - tq
    q_pos = torch.arange(tq, device=device).unsqueeze(1) + offset  # [Tq, 1]
    k_pos = torch.arange(tk, device=device).unsqueeze(0)  # [1, Tk]
    return k_pos > q_pos


def _flash_varlen_ops():
    """flash_attn's low-level varlen forward/backward, or None.

    The low-level pair is used rather than ``flash_attn_varlen_func`` because
    only these expose ``softmax_lse`` and let us drive backward ourselves.
    ``causal=True`` with ``seqlen_q < seqlen_k`` aligns the mask to the bottom
    right, which is exactly chunkflow's "q short, kv long" semantics — verified
    against the offset mask in ``_causal_mask``.
    """
    try:
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward,
            _flash_attn_varlen_forward,
        )
    except ImportError:
        return None
    return _flash_attn_varlen_forward, _flash_attn_varlen_backward


def _use_flash(q: torch.Tensor) -> bool:
    """Flash kernels need CUDA and fp16/bf16; CPU tests fall back to the
    reference implementation."""
    return q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and _flash_varlen_ops() is not None


def _cu_seqlens(total: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, total], dtype=torch.int32, device=device)


def _flash_fwd(q, k, v, causal, softmax_scale):
    fwd, _bwd = _flash_varlen_ops()
    out, lse, _s, _rng = fwd(
        q,
        k,
        v,
        _cu_seqlens(q.shape[0], q.device),
        _cu_seqlens(k.shape[0], k.device),
        q.shape[0],
        k.shape[0],
        0.0,  # dropout_p
        softmax_scale,
        causal,
    )[:4]
    return out, lse


def _flash_bwd(dout, q, k, v, out, lse, causal, softmax_scale):
    _fwd, bwd = _flash_varlen_ops()
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        _cu_seqlens(q.shape[0], q.device),
        _cu_seqlens(k.shape[0], k.device),
        q.shape[0],
        k.shape[0],
        0.0,  # dropout_p
        softmax_scale,
        causal,
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # deterministic
    )
    return dq, dk, dv


class ChunkCacheAttention(torch.autograd.Function):
    """Chunk-internal attention op: ``q`` (this chunk) attends over the K/V of
    chunks ``0..i`` (cache-so-far, "q short, kv long").

    Takes the PER-CHUNK k/v rather than a pre-concatenated cache, and saves
    those for backward instead of the concatenation. Every chunk sees the same
    k/v tensor objects (they come from one ``StateStore``), so saving them N
    times costs one reference each, not one copy each: total saved activation
    is O(T) instead of the O(T^2 / chunk_size) a saved ``torch.cat`` would
    cost. Backward re-concatenates — one transient copy, freed immediately,
    the same compute-for-memory trade real flash attention makes.

    Mirrors the ``SparseMLA`` autograd.Function template in
    ``relax/models/glm_moe_dsa/ops/sparse_mla.py``.
    """

    @staticmethod
    def forward(ctx, q, causal, softmax_scale, num_chunks, *kv_chunks):
        ks, vs = kv_chunks[:num_chunks], kv_chunks[num_chunks:]
        k_all = torch.cat(ks, dim=0)
        v_all = torch.cat(vs, dim=0)
        flash = _use_flash(q)
        if flash:
            out, lse = _flash_fwd(q, k_all, v_all, causal, softmax_scale)
        else:
            out, lse = _manual_attn_forward(q, k_all, v_all, causal, softmax_scale)
        # k_all/v_all deliberately NOT saved: they are concatenations, so saving
        # them would cost a fresh copy per chunk. The per-chunk tensors below are
        # shared across every chunk's ctx, so they cost one reference each.
        # `out` is saved only on the flash path, whose backward needs it.
        ctx.save_for_backward(q, lse, *((out,) if flash else ()), *kv_chunks)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.num_chunks = num_chunks
        ctx.flash = flash
        ctx.chunk_lengths = [k.shape[0] for k in ks]
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        # `_dlse` (grad w.r.t. lse) is unused: lse never flows forward into
        # the rest of the network, so nothing downstream produces a real
        # gradient for it.
        saved = list(ctx.saved_tensors)
        q, lse = saved.pop(0), saved.pop(0)
        out = saved.pop(0) if ctx.flash else None
        kv_chunks = saved
        n = ctx.num_chunks
        k_all = torch.cat(kv_chunks[:n], dim=0)  # transient, freed on return
        v_all = torch.cat(kv_chunks[n:], dim=0)
        if ctx.flash:
            dq, dk_all, dv_all = _flash_bwd(
                dout.contiguous(), q, k_all, v_all, out, lse, ctx.causal, ctx.softmax_scale
            )
        else:
            dq, dk_all, dv_all = _manual_attn_backward(dout, q, k_all, v_all, lse, ctx.causal, ctx.softmax_scale)
        dks = torch.split(dk_all, ctx.chunk_lengths, dim=0)
        dvs = torch.split(dv_all, ctx.chunk_lengths, dim=0)
        return (dq, None, None, None, *dks, *dvs)


def chunk_attention_forward(
    chunk_id: int,
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    store: StateStore,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward for one chunk: store ``(new_k, new_v)`` under ``chunk_id``
    (增量存储), then attend ``q`` over every chunk's K/V up to and including this
    one (查找 + concat).

    Must be called with ``chunk_id`` ascending from 0 — enforced by
    ``StateStore.put``.
    """
    store.put(chunk_id, ChunkState({"k": new_k, "v": new_v}))
    states = store.lookup(chunk_id)
    ks = [s.tensors["k"] for s in states]
    vs = [s.tensors["v"] for s in states]
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    # Pass the chunks themselves, not a concatenation: the op saves these for
    # backward, and they are shared across chunks, so nothing is copied.
    return ChunkCacheAttention.apply(q, causal, scale, len(ks), *ks, *vs)


def chunk_attention_backward(chunk_id: int, out: torch.Tensor, dout: torch.Tensor, store: StateStore) -> None:
    """Backward for one chunk: back-propagate through ``out`` (accumulating,
    via ordinary autograd, into every K/V leaf this chunk read — including
    chunks stored under smaller ids), then free this chunk (减量删除).

    Must be called with ``chunk_id`` descending from the highest chunk —
    enforced by ``StateStore.free``. Pairing backward + free in one call makes
    it impossible to free a chunk without having backpropagated through it
    first (or vice versa). Each chunk's forward built its own private ``cat``
    node over the leaves it read, so one plain ``.backward()`` per chunk is
    enough — no ``retain_graph`` needed.
    """
    out.backward(dout)
    store.free(chunk_id)
