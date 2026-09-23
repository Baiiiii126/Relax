# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wire chunkflow's chunked attention into a built Megatron model.

``ChunkedCoreAttention`` is a drop-in ``core_attention`` replacement: it splits
each sequence into chunks and attends each chunk over the K/V of all chunks up
to it, held in a per-layer ``StateStore``. Numerically identical to unchunked
attention. ``install(model, chunk_size)`` swaps every ``core_attention``;
``uninstall(model)`` restores.
"""

from __future__ import annotations

import itertools
import os

import torch
from torch import nn

from relax.backends.megatron.chunkflow.attention_ops import chunk_attention_forward
from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR
from relax.backends.megatron.chunkflow.state_store import ChunkState, StateStore
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_ORIG_ATTR = "_chunkflow_original_core_attention"


def _chunkflow_trace() -> bool:
    return os.environ.get("RELAX_CHUNKFLOW_TRACE") == "1"


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

    def __init__(self, config, layer_number: int, chunk_size: int, softmax_scale=None, runtime=None, **kwargs):
        super().__init__()
        del kwargs
        self.config = config
        self.layer_number = layer_number
        self.chunk_size = chunk_size
        self.softmax_scale = softmax_scale
        self.runtime = runtime

    def _forward_unit(self, query, key, value, packed_seq_params) -> torch.Tensor:
        """One attention call for a whole unit, each sequence over its own K/V.

        A dependent sequence attends its group's earlier chunks plus its own; a
        standalone one attends only its own. Both go into a single varlen call,
        which is where the call-count saving comes from.

        The store keeps a *detached leaf* per chunk rather than the live tensor.
        Later chunks then read a leaf instead of chaining into this chunk's
        graph, which is what lets the schedule backward one unit at a time --
        with the live tensor, the second ``.backward()`` would hit "backward
        through the graph a second time".

        Detaching loses the gradient later chunks owe this chunk, so a hook adds
        it back: by the time this unit backwards (the schedule goes descending),
        the leaf has collected every later chunk's contribution, and the hook
        folds it into the gradient flowing to the live tensor.
        """
        from relax.backends.megatron.chunkflow.attention_ops import UnitCacheAttention
        from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR

        unit = getattr(packed_seq_params, UNIT_ATTR, None)
        recompute = getattr(packed_seq_params, RECOMPUTE_ATTR, False)
        cu = packed_seq_params.cu_seqlens_q_cpu
        meta = [(c.group, c.group_idx) for c in unit.chunks for _ in c.segments]

        ks, vs = [], []
        for i, (group, group_idx) in enumerate(meta):
            start, end = cu[i], cu[i + 1]
            k_own, v_own = key[start:end], value[start:end]
            if group is None:
                ks.append(k_own)
                vs.append(v_own)
                continue

            store = self.runtime.store(self.layer_number, group)
            if recompute:
                # An Fp pass already put this chunk's K/V in the store as a leaf.
                # Recompute only rebuilds this chunk's graph; re-putting would
                # trip the store's ascending invariant.
                states = store.lookup(group_idx)
                leaf_k, leaf_v = states[-1].tensors["k"], states[-1].tensors["v"]
            else:
                if _chunkflow_trace():
                    logger.info(f"[chunkflow-trace] put layer={self.layer_number} group={group} idx={group_idx} live={store.live_chunk_ids}")
                leaf_k = k_own.detach().requires_grad_(True)
                leaf_v = v_own.detach().requires_grad_(True)
                store.put(group_idx, ChunkState({"k": leaf_k, "v": leaf_v}))
            _fold_leaf_grad_into(k_own, leaf_k)
            _fold_leaf_grad_into(v_own, leaf_v)

            earlier = store.lookup(group_idx)[:-1]  # this chunk's own entry is the leaf
            ks.append(torch.cat([*(s.tensors["k"] for s in earlier), k_own], dim=0))
            vs.append(torch.cat([*(s.tensors["v"] for s in earlier), v_own], dim=0))

        device = query.device
        # The unit batch is padded to an alignment multiple, so ``query`` can
        # carry rows past cu[-1]. The kernel is given exactly the rows the
        # cu_seqlens describe: a tensor longer than its own cu_seqlens is
        # out-of-bounds by construction.
        query = query[: cu[-1]]
        cu_q = torch.tensor(cu, dtype=torch.int32, device=device)
        cu_kv = torch.tensor([0, *itertools.accumulate(t.shape[0] for t in ks)], dtype=torch.int32, device=device)
        scale = self.softmax_scale if self.softmax_scale is not None else query.shape[-1] ** -0.5
        out, _lse = UnitCacheAttention.apply(query, True, scale, cu_q, cu_kv, *ks, *vs)
        return out

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

        if packed_seq_params is not None and getattr(packed_seq_params, UNIT_ATTR, None) is not None:
            padded_rows = query.shape[0]
            out = self._forward_unit(query, key, value, packed_seq_params).reshape(-1, query.shape[1] * query.shape[2])
            # The unit batch is padded for alignment; the layer expects one
            # output row per input row, and the pad rows' values are never read.
            if out.shape[0] < padded_rows:
                out = torch.nn.functional.pad(out, (0, 0, 0, padded_rows - out.shape[0]))
            return out

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


def _fold_leaf_grad_into(live: torch.Tensor, leaf: torch.Tensor) -> None:
    """Add the leaf's accumulated gradient to ``live``'s on the way back.

    ``live`` is detached into ``leaf`` for later chunks to read, so the
    gradient they owe it never reaches ``live`` through the graph. By the time
    ``live``'s own backward runs — the schedule descends — the leaf has every
    later contribution, and this folds them in. One traversal, no second
    backward.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad if leaf.grad is None else grad + leaf.grad

    # An Fp forward runs under no_grad, so its k/v carry no gradient and cannot
    # take a hook -- and they do not need one: the leaf holds the K/V and the
    # gradient lands via the Fr pass's recomputed tensors, which do carry it.
    if live.requires_grad:
        live.register_hook(hook)


def _attention_modules(model):
    for module in model.modules():
        core = getattr(module, "core_attention", None)
        if core is not None and hasattr(module, "layer_number"):
            yield module


def install(model, chunk_size: int) -> int:
    """Swap every ``core_attention`` for a chunked one.

    Also attaches the runtime the unit mode needs — the stores that outlive a
    single forward call live there. Returns how many layers were swapped.
    """
    from relax.backends.megatron.chunkflow.runtime import attach_runtime

    runtime = attach_runtime(model, chunk_size)

    if os.environ.get("RELAX_CHUNKFLOW_USE_ORIGINAL_ATTENTION") == "1":
        logger.warning(
            "chunkflow: original core attention retained for single-unit alignment; "
            "do not use this diagnostic mode with dependent chunks"
        )
        return 0

    # The unit path needs RoPE's unfused branch: it applies the position offset
    # chunkflow patches in, while Transformer Engine's fused_rope kernel takes
    # the padded cu_seqlens literally and reads past the frequency table (that
    # is the illegal memory access at step 0, not an attention bug). The flag
    # is read per forward, and install() runs after the model -- and therefore
    # its config -- is built, so flipping it here takes effect everywhere.
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
            runtime=runtime,
        )
        n += 1
    if n:
        for module in model.modules():
            cfg = getattr(module, "config", None)
            if cfg is not None and getattr(cfg, "apply_rope_fusion", False):
                cfg.apply_rope_fusion = False
        logger.info("chunkflow: apply_rope_fusion disabled (the offset path needs the unfused RoPE branch)")
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
