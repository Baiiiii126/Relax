# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Chunk-local causal attention over a KV cache for chunkflow.

THD ``[T, H, D]`` shapes. Within one chunk, ``q`` is the chunk's queries
(``Tq``) and ``k_all``/``v_all`` are the accumulated K/V up to it (``Tk >=
Tq``). Two backends: flash_attn's low-level varlen forward/backward on CUDA
fp16/bf16 (production), and a plain-torch reference on CPU/fp32 (for tests).

The op saves per-chunk K/V (shared across chunks), never their concatenation,
so saved activation is O(T) rather than O(T^2 / chunk_size). The per-chunk
backward calls run in descending order; autograd accumulates dk/dv into the
shared leaves.
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
    k, v, _repeat = _expand_kv_heads(k, v, h)
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
    tq, h, _d = q.shape
    tk = k.shape[0]
    kv_heads = k.shape[1]
    k, v, repeat = _expand_kv_heads(k, v, h)
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
    if repeat > 1:
        dk = dk.reshape(tk, kv_heads, repeat, _d).sum(dim=2)
        dv = dv.reshape(tk, kv_heads, repeat, _d).sum(dim=2)
    return dq, dk, dv


def _expand_kv_heads(k: torch.Tensor, v: torch.Tensor, query_heads: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Expand compact GQA K/V only for the plain-torch reference path."""
    kv_heads = k.shape[1]
    if kv_heads == query_heads:
        return k, v, 1
    if query_heads % kv_heads != 0:
        raise ValueError(f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})")
    repeat = query_heads // kv_heads
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1), repeat


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


def _split_sizes_by_cu(cu: torch.Tensor) -> list[int]:
    return [int(cu[i + 1] - cu[i]) for i in range(cu.numel() - 1)]


class UnitCacheAttention(torch.autograd.Function):
    """One varlen call for a whole execution unit.

    A unit packs several sequences, each attending over its own K/V: its
    group's prefix plus its own chunk. ``cu_seqlens_q`` gives the query lengths
    and ``cu_seqlens_kv`` the K/V lengths -- the two differ for any chunk with
    a prefix, which is what lets one call serve them all instead of one call
    per chunk. Call count, not FLOPs, dominates chunked attention's cost.

    An optional final ``(max_q, max_kv)`` tuple supplies CPU length metadata.
    The production caller already has these lengths before creating CUDA
    cu_seqlens; carrying them through backward avoids GPU scalar reads.
    """

    @staticmethod
    def forward(ctx, q, causal, softmax_scale, cu_q, cu_kv, *kv):
        ctx.has_max_seqlens = bool(kv and isinstance(kv[-1], tuple))
        max_seqlens = kv[-1] if ctx.has_max_seqlens else None
        if ctx.has_max_seqlens:
            kv = kv[:-1]
            if len(max_seqlens) != 2 or any(type(length) is not int or length <= 0 for length in max_seqlens):
                raise ValueError("max_seqlens must contain two positive Python integers")
        n = len(kv) // 2
        k_all = torch.cat(kv[:n], dim=0)
        v_all = torch.cat(kv[n:], dim=0)
        flash = _use_flash(q)
        if flash:
            if max_seqlens is None:
                # Compatibility for callers without CPU metadata: read once
                # in forward and reuse the result in backward.
                max_seqlens = (_max_seqlen(cu_q), _max_seqlen(cu_kv))
            out, lse = _flash_varlen(q, k_all, v_all, cu_q, cu_kv, causal, softmax_scale, max_seqlens=max_seqlens)
        else:
            out, lse = _manual_varlen_forward(q, k_all, v_all, cu_q, cu_kv, causal, softmax_scale)

        ctx.save_for_backward(q, lse, cu_q, cu_kv, *((out,) if flash else ()), *kv)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.flash = flash
        ctx.n = n
        ctx.kv_lengths = [k.shape[0] for k in kv[:n]]
        ctx.max_seqlens = max_seqlens
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        saved = list(ctx.saved_tensors)
        q, lse, cu_q, cu_kv = saved[:4]
        rest = saved[4:]
        out = rest.pop(0) if ctx.flash else None
        n = ctx.n
        k_all = torch.cat(rest[:n], dim=0)
        v_all = torch.cat(rest[n:], dim=0)
        if ctx.flash:
            dq, dk_all, dv_all = _flash_varlen_backward(
                dout.contiguous(),
                q,
                k_all,
                v_all,
                out,
                lse,
                cu_q,
                cu_kv,
                ctx.causal,
                ctx.softmax_scale,
                max_seqlens=ctx.max_seqlens,
            )
        else:
            dq, dk_all, dv_all = _manual_varlen_backward(
                dout, q, k_all, v_all, lse, cu_q, cu_kv, ctx.causal, ctx.softmax_scale
            )
        dks = torch.split(dk_all, ctx.kv_lengths, dim=0)
        dvs = torch.split(dv_all, ctx.kv_lengths, dim=0)
        gradients = (dq, None, None, None, None, *dks, *dvs)
        return (*gradients, None) if ctx.has_max_seqlens else gradients


def _manual_varlen_forward(q, k, v, cu_q, cu_kv, causal, softmax_scale):
    """Per-sequence causal attention over a shared packed K/V, one pass."""
    outs, lses = [], []
    qs = _split_sizes_by_cu(cu_q)
    ks = _split_sizes_by_cu(cu_kv)
    qo = ko = 0
    for ql, kl in zip(qs, ks):
        out, lse = _manual_attn_forward(
            q[qo : qo + ql], k[ko : ko + kl], v[ko : ko + kl], causal=causal, softmax_scale=softmax_scale
        )
        outs.append(out)
        lses.append(lse)
        qo += ql
        ko += kl
    return torch.cat(outs, dim=0), torch.cat(lses, dim=0)


def _manual_varlen_backward(dout, q, k, v, lse, cu_q, cu_kv, causal, softmax_scale):
    qs = _split_sizes_by_cu(cu_q)
    ks = _split_sizes_by_cu(cu_kv)
    dqs, dks, dvs = [], [], []
    qo = ko = lo = do = 0
    for ql, kl in zip(qs, ks):
        dq, dk, dv = _manual_attn_backward(
            dout[do : do + ql],
            q[qo : qo + ql],
            k[ko : ko + kl],
            v[ko : ko + kl],
            lse[lo : lo + ql],
            causal=causal,
            softmax_scale=softmax_scale,
        )
        dqs.append(dq)
        dks.append(dk)
        dvs.append(dv)
        qo += ql
        ko += kl
        lo += ql
        do += ql
    return torch.cat(dqs, dim=0), torch.cat(dks, dim=0), torch.cat(dvs, dim=0)


def _max_seqlen(cu: torch.Tensor) -> int:
    """Largest single-sequence length in a ``cu_seqlens``.

    flash's varlen kernels want the per-sequence maximum, not the total. With
    several sequences packed, ``cu[-1]`` is far larger than any one of them and
    the kernel reads past the end of the packed rows -- an illegal memory
    access, not a wrong number.
    """
    return max(int(cu[i + 1] - cu[i]) for i in range(cu.numel() - 1))


def _flash_varlen(q, k, v, cu_q, cu_kv, causal, softmax_scale, *, max_seqlens: tuple[int, int] | None = None):
    fwd, _bwd = _flash_varlen_ops()
    max_q, max_kv = max_seqlens if max_seqlens is not None else (_max_seqlen(cu_q), _max_seqlen(cu_kv))
    out, lse, _s, _rng = fwd(
        q,
        k,
        v,
        cu_q.to(torch.int32),
        cu_kv.to(torch.int32),
        max_q,
        max_kv,
        0.0,
        softmax_scale,
        causal,
    )[:4]
    return out, lse


def _flash_varlen_backward(
    dout, q, k, v, out, lse, cu_q, cu_kv, causal, softmax_scale, *, max_seqlens: tuple[int, int] | None = None
):
    _fwd, bwd = _flash_varlen_ops()
    max_q, max_kv = max_seqlens if max_seqlens is not None else (_max_seqlen(cu_q), _max_seqlen(cu_kv))
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
        cu_q.to(torch.int32),
        cu_kv.to(torch.int32),
        max_q,
        max_kv,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
    )
    return dq, dk, dv
