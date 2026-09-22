# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Slice one micro-batch into per-unit batches the existing loss can consume.

The schedule pulls a micro-batch, runs ``get_batch`` on it once (so every
alignment rule — SFT mask shift, padding, cu_seqlens — is applied exactly as
the baseline applies it), and then cuts that processed batch into one batch per
execution unit. Each unit batch is shaped so ``loss_function`` works unmodified,
which is what makes the chunked run comparable to the baseline at all.

The one field that must not survive the slice is ``unconcat_tokens``: the loss
pairs logit row *j* with token *j+1* (``loss.py``: ``logits[start-1:end-1]``
against ``tokens[-R:]``), so a unit covering parent rows ``[s, e)`` must report
as its targets the parent's tokens ``[s+1, e+1)`` — one row past its own input.
Feeding the unit its own tokens there would train it to predict the token it
was just given.

RoPE gets the same treatment from ``chunkflow.rope``: the unit's
``packed_seq_params`` stays chunk-local for the attention kernels and carries
the absolute offsets for RoPE.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from relax.backends.megatron.chunkflow import rope as cf_rope
from relax.backends.megatron.chunkflow.scheduler import Segment
from relax.backends.megatron.chunkflow.units import Unit


def _segment_rows(parent_cu: Sequence[int], segment: Segment) -> tuple[int, int]:
    """The parent rows ``[start, end)`` of one segment."""
    start = parent_cu[segment.seq_id] + segment.prefix
    return start, start + segment.length


def _rows_for_unit(parent_cu: Sequence[int], unit: Unit) -> list[tuple[int, int]]:
    return [_segment_rows(parent_cu, s) for seg in unit.chunks for s in seg.segments]


def slice_parent_batch(
    parent: dict,
    unit: Unit,
    *,
    pad_token_id: int,
    args=None,
) -> dict:
    """Cut a processed micro-batch into the batch for one execution unit.

    ``parent`` is what ``get_batch`` returned for the whole micro-batch. The
    result is a batch of the same shape holding only this unit's rows, with
    ``unconcat_tokens`` shifted one row forward (see module docstring) and the
    RoPE offset fields set on ``packed_seq_params``.
    """
    tokens = parent["tokens"]
    assert tokens.dim() == 2 and tokens.size(0) == 1, f"expected [1, T], got {tokens.shape}"
    flat_tokens = tokens[0]
    flat_mask = parent["full_loss_masks"][0]
    parent_cu = parent["packed_seq_params"].cu_seqlens_q_cpu

    rows = _rows_for_unit(parent_cu, unit)
    lengths = [e - s for s, e in rows]
    segments = [seg for chunk in unit.chunks for seg in chunk.segments]

    token_parts = [flat_tokens[s:e] for s, e in rows]
    # Targets are the parent's tokens one row ahead, but never across a segment
    # boundary: the last row of a sequence predicts nothing, and reading the next
    # sequence's first token there would train on the wrong data. The real
    # pipeline zeroes that row's mask (``F.pad(loss_mask, (prompt_length-1, 1))``
    # in get_batch), so it is unread either way -- clamping makes that a
    # guarantee of this function rather than a property of its caller.
    target_parts = []
    for (s, e), length, seg in zip(rows, lengths, segments, strict=True):
        seq_end = parent_cu[seg.seq_id + 1]
        target = flat_tokens[s + 1 : min(e + 1, seq_end)]
        if target.size(0) < length:
            target = torch.cat(
                [
                    target,
                    torch.full((length - target.size(0),), pad_token_id, dtype=target.dtype, device=target.device),
                ]
            )
        target_parts.append(target)
    mask_parts = [flat_mask[s:e] for s, e in rows]

    unit_tokens = torch.cat(token_parts)
    unit_targets = torch.cat(target_parts)
    unit_mask = torch.cat(mask_parts)

    # loss_masks and full_loss_masks are two different things: the loss splits
    # log-probs by response_lengths and multiplies by loss_masks element-wise,
    # so that one must stay the unpadded length; the model's forward wants one
    # mask row per padded token, so full_loss_masks is padded alongside tokens.
    full_mask = unit_mask.clone()
    pad = (-unit_tokens.size(0)) % 128
    if pad:
        unit_tokens = torch.cat(
            [unit_tokens, torch.full((pad,), pad_token_id, dtype=unit_tokens.dtype, device=unit_tokens.device)]
        )
        full_mask = torch.cat([full_mask, torch.zeros(pad, dtype=unit_mask.dtype, device=unit_mask.device)])

    offsets = [parent_cu[seg.seq_id] + seg.prefix for seg in segments]
    packed = _build_packed_seq_params(unit_tokens, lengths, offsets)

    batch: dict = {
        "tokens": unit_tokens.unsqueeze(0),
        "unconcat_tokens": [unit_targets],
        "full_loss_masks": full_mask.unsqueeze(0),
        "loss_masks": [unit_mask],
        "total_lengths": [sum(lengths) + 1],
        "response_lengths": [sum(lengths)],
        "max_seq_lens": [max(lengths)],
        "packed_seq_params": packed,
    }

    _slice_response_aligned_keys(parent, batch, unit, rows)
    for key in ("__is_dummy__", "__loss_scale__", "replay_micro_batch_index"):
        if key in parent:
            batch[key] = parent[key]
    return batch


def _build_packed_seq_params(unit_tokens: torch.Tensor, lengths: list[int], offsets: list[int]):
    """Chunk-local cu_seqlens for the kernels, absolute offsets for RoPE."""
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    device = unit_tokens.device

    packed = PackedSeqParams(
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
        cu_seqlens_kv=torch.tensor(cu, dtype=torch.int32, device=device),
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        qkv_format="thd",
    )
    packed.cu_seqlens_q_cpu = cu

    if any(offset for offset in offsets):
        cf_rope.apply_rope_offset(packed, offsets=offsets, lengths=lengths)
    return packed


def _slice_response_aligned_keys(parent: dict, batch: dict, unit: Unit, rows: list[tuple[int, int]]) -> None:
    """RL keys are indexed by response position; map each unit row onto one.

    A unit row ``r`` of parent sample ``k`` sits at response index
    ``r - prompt_len(k)``. Rows before the response (or past its end) carry no
    response-aligned value, so they take index 0 — their loss mask is 0, so the
    value is never read.
    """
    total_lengths = parent.get("total_lengths") or []
    response_lengths = parent.get("response_lengths") or []
    keys = ("log_probs", "ref_log_probs", "values", "advantages", "returns", "rollout_log_probs")
    if not any(parent.get(k) is not None for k in keys):
        return

    # Parent row -> (sample index, response index).
    parent_cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    sample_of_row: dict[int, tuple[int, int]] = {}
    for k, (cu_lo, cu_hi) in enumerate(zip(parent_cu[:-1], parent_cu[1:], strict=True)):
        prompt_len = (total_lengths[k] - response_lengths[k]) if k < len(total_lengths) else 0
        for r in range(int(cu_lo), int(cu_hi)):
            sample_of_row[r] = (k, max(r - int(cu_lo) - prompt_len, 0))

    for key in keys:
        vals = parent.get(key)
        if vals is None:
            continue
        picked = []
        for s, e in rows:
            idx = [sample_of_row.get(r, (0, 0)) for r in range(s, e)]
            tensor = vals[idx[0][0]] if idx else vals[0]
            picked.append(tensor[[i for _, i in idx]])
        batch[key] = [torch.cat(picked)]
