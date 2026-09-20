# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wire chunkflow's chunked attention into a built Megatron model.

``ChunkedCoreAttention`` is a drop-in replacement for Megatron's
``core_attention`` (same contract as ``relax/models/gemma4/attention.py``):
it splits each sequence into chunks and attends each chunk over the K/V of all
chunks up to it, held in a per-layer :class:`StateStore`.

Numerically this is identical to unchunked attention (verified against the real
Qwen3-0.6B: logits/grads match to fp32 noise, independent of chunk count). It
does not by itself reduce memory -- that needs the per-chunk backward schedule,
which is a separate step.

``install(model, chunk_size)`` walks a built model and swaps every
``self_attention.core_attention`` instance; ``uninstall(model)`` restores.
"""

from __future__ import annotations

import torch
from torch import nn

from relax.backends.megatron.chunkflow.attention_ops import chunk_attention_forward
from relax.backends.megatron.chunkflow.state_store import StateStore
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_ORIG_ATTR = "_chunkflow_original_core_attention"


def split_sizes(total: int, chunk_size: int) -> list[int]:
    """Chunk lengths covering ``total``; the tail keeps whatever is left."""
    if chunk_size <= 0 or chunk_size >= total:
        return [total]
    sizes = [chunk_size] * (total // chunk_size)
    tail = total - sum(sizes)
    if tail:
        sizes.append(tail)
    return sizes


class ChunkedCoreAttention(nn.Module):
    """core_attention replacement that runs attention chunk by chunk.

    Shapes follow the THD convention ``[T, H, D]``. Each chunk's K/V go into a
    StateStore and the chunk's queries attend over the accumulated cache ("q
    short, kv long"). Sequence boundaries come from
    ``packed_seq_params.cu_seqlens_q`` when present, otherwise the whole input
    is treated as one sequence.
    """

    def __init__(self, config, layer_number: int, chunk_size: int, softmax_scale=None, **kwargs):
        super().__init__()
        del kwargs
        self.config = config
        self.layer_number = layer_number
        self.chunk_size = chunk_size
        self.softmax_scale = softmax_scale

    def _attend_one_sequence(self, q, k, v) -> torch.Tensor:
        """q/k/v: ``[T, H, D]`` for a single sequence. Returns ``[T, H, D]``."""
        store = StateStore()
        outs = []
        offset = 0
        for chunk_id, size in enumerate(split_sizes(q.shape[0], self.chunk_size)):
            sl = slice(offset, offset + size)
            out, _lse = chunk_attention_forward(
                chunk_id, q[sl], k[sl], v[sl], store, causal=True, softmax_scale=self.softmax_scale
            )
            outs.append(out)
            offset += size
        return torch.cat(outs, dim=0)

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, packed_seq_params=None, **kwargs):
        del attention_mask, attn_mask_type, kwargs
        cp_size = getattr(self.config, "context_parallel_size", 1) or 1
        if cp_size > 1:
            raise NotImplementedError(
                "ChunkedCoreAttention does not implement context parallelism: chunkflow's "
                "StateStore is local and CP would split the very sequence it caches. "
                "Run with --context-parallel-size 1."
            )
        if query.dim() != 3:
            raise NotImplementedError(
                f"ChunkedCoreAttention expects THD [T, H, D]; got {query.dim()}D. "
                "SBHD would need its own causal-mask handling before it is safe to enable."
            )

        # GQA: widen K/V heads to match Q before chunking, so the store holds
        # what attention actually consumes.
        if key.shape[1] != query.shape[1]:
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        cu = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        if cu is None:
            out = self._attend_one_sequence(query, key, value)
        else:
            bounds = getattr(packed_seq_params, "cu_seqlens_q_cpu", None)
            if bounds is None:
                bounds = cu.detach().cpu().tolist()
            outs = []
            for start, end in zip(bounds[:-1], bounds[1:]):
                if end > start:
                    outs.append(self._attend_one_sequence(query[start:end], key[start:end], value[start:end]))
            out = torch.cat(outs, dim=0)
        return out.reshape(out.shape[0], -1)


def _attention_modules(model):
    for module in model.modules():
        core = getattr(module, "core_attention", None)
        if core is not None and hasattr(module, "layer_number"):
            yield module


def install(model, chunk_size: int) -> int:
    """Swap every ``core_attention`` for a chunked one.

    Returns how many.
    """
    n = 0
    for attn in _attention_modules(model):
        if hasattr(attn, _ORIG_ATTR):
            continue
        original = attn.core_attention
        setattr(attn, _ORIG_ATTR, original)
        attn.core_attention = ChunkedCoreAttention(
            config=getattr(original, "config", None) or attn.config,
            layer_number=attn.layer_number,
            chunk_size=chunk_size,
            softmax_scale=getattr(original, "softmax_scale", None),
        )
        n += 1
    logger.info(f"chunkflow: chunked attention installed on {n} layer(s), chunk_size={chunk_size}")
    return n


def uninstall(model) -> int:
    n = 0
    for attn in _attention_modules(model):
        original = getattr(attn, _ORIG_ATTR, None)
        if original is None:
            continue
        attn.core_attention = original
        delattr(attn, _ORIG_ATTR)
        n += 1
    return n
