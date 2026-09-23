# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Slicing a micro-batch into per-unit batches must lose and shift nothing.

Pure tensor tests on a synthetic parent batch — no model, no GPU. The
properties here are the ones that would otherwise corrupt training silently:

* the units tile the parent's rows in order, with no row dropped or duplicated;
* a unit's targets are the parent's tokens one row ahead (not its own tokens);
* a unit's mask slice is exactly the rows it covers, so the summed loss over
  units equals the parent's loss.
"""

from __future__ import annotations

import pytest
import torch


try:
    from megatron.core.packed_seq_params import PackedSeqParams

    from relax.backends.megatron.chunkflow import rope as cf_rope
    from relax.backends.megatron.chunkflow.scheduler import build_chunks
    from relax.backends.megatron.chunkflow.unit_batch import _rope_offsets, _rows_for_unit, slice_parent_batch
    from relax.backends.megatron.chunkflow.units import build_micro_batches
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


PAD = 0


def _parent(lengths: list[int], device="cpu") -> dict:
    """A synthetic processed micro-batch, shaped like ``get_batch``'s
    output."""
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    total = cu[-1]

    tokens = torch.arange(1, total + 1, device=device).unsqueeze(0)
    # Supervise the last quarter of each sample, but never its final row: that
    # row predicts the next sequence's first token, and get_batch zeroes its
    # mask for exactly that reason (``F.pad(loss_mask, (prompt_length - 1, 1))``).
    mask = torch.zeros(total, dtype=torch.long, device=device)
    for lo, hi in zip(cu[:-1], cu[1:], strict=True):
        lo, hi = int(lo), int(hi)
        mask[max(hi - (hi - lo) // 4, lo + 1) : hi - 1] = 1

    packed = PackedSeqParams(
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
        cu_seqlens_kv=torch.tensor(cu, dtype=torch.int32, device=device),
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        qkv_format="thd",
    )
    packed.cu_seqlens_q_cpu = cu

    return {
        "tokens": tokens,
        "unconcat_tokens": [tokens[0][lo:hi].clone() for lo, hi in zip(cu[:-1], cu[1:], strict=True)],
        "full_loss_masks": mask.unsqueeze(0),
        "loss_masks": [mask[lo:hi].clone() for lo, hi in zip(cu[:-1], cu[1:], strict=True)],
        "total_lengths": list(lengths),
        "response_lengths": list(lengths),
        "max_seq_lens": list(lengths),
        "packed_seq_params": packed,
    }


def _units(lengths, chunk_size, policy="staggered"):
    chunks = build_chunks(lengths, chunk_size=chunk_size, repack_tail=False)
    return build_micro_batches(chunks, policy), chunks


def _segments_of(unit):
    """A unit's segments, flattened in the same order ``_rows_for_unit``
    walks."""
    return [seg for chunk in unit.chunks for seg in chunk.segments]


@pytest.mark.parametrize("policy", ["staggered", "lockstep"])
def test_units_cover_every_parent_row_exactly_once(policy):
    """Coverage, not order: bin-packing reorders standalone chunks (best-fit
    decreasing), and neither the masked loss nor attention depends on the order
    of independent sequences.

    Dependent units do have to stay in group order, which the scheduler tests
    already pin.
    """
    lengths = [40, 12, 16]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16, policy)

    seen: list[int] = []
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        real = batch["response_lengths"][0]  # unpadded length
        seen.extend(batch["tokens"][0][:real].tolist())

    assert sorted(seen) == sorted(parent["tokens"][0].tolist())
    assert len(seen) == len(set(seen)), "no row may be covered twice"


@pytest.mark.parametrize("policy", ["staggered", "lockstep"])
def test_unit_targets_are_the_parent_tokens_one_row_ahead(policy):
    lengths = [40, 12, 16]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16, policy)
    tokens = parent["tokens"][0]
    cu = parent["packed_seq_params"].cu_seqlens_q_cpu

    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        real = batch["response_lengths"][0]
        got = batch["unconcat_tokens"][0][:real]

        expected_parts = []
        for (s, e), seg in zip(_rows_for_unit(cu, unit), _segments_of(unit), strict=True):
            length = e - s
            # One row ahead, but never past the end of the owning sequence.
            part = tokens[s + 1 : min(e + 1, cu[seg.seq_id + 1])]
            if part.numel() < length:
                part = torch.cat([part, torch.full((length - part.numel(),), PAD, dtype=tokens.dtype)])
            expected_parts.append(part)

        torch.testing.assert_close(got, torch.cat(expected_parts))


@pytest.mark.parametrize("policy", ["staggered", "lockstep"])
def test_a_units_last_row_never_targets_the_next_sequence(policy):
    """The clamp that keeps one sequence's tail from training on the next
    sequence's head.

    get_batch zeroes that row's mask, so it is unread either way -- but relying
    on the caller for that is exactly the kind of coupling that turns into a
    silent wrong-gradient bug when something upstream changes.
    """
    lengths = [40, 12, 16]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16, policy)
    cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    checked = 0

    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        got = batch["unconcat_tokens"][0]
        row = 0
        for (s, e), seg in zip(_rows_for_unit(cu, unit), _segments_of(unit), strict=True):
            seq_end = cu[seg.seq_id + 1]
            if e == seq_end:  # this segment runs to the end of its sequence
                assert got[row + (e - s) - 1] == PAD, "tail row must not take the next sequence's token"
                checked += 1
            row += e - s

    assert checked > 0, "fixture must contain a sequence-final segment"


def test_unit_mask_is_exactly_the_rows_the_unit_covers():
    lengths = [40, 12]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16)
    mask = parent["full_loss_masks"][0]
    cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    covered = 0

    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        real = batch["response_lengths"][0]
        expected = torch.cat([mask[s:e] for s, e in _rows_for_unit(cu, unit)])
        torch.testing.assert_close(batch["full_loss_masks"][0][:real], expected)
        covered += real
    assert covered == mask.numel()


def test_summed_unit_loss_equals_the_parent_loss():
    """The property that makes the A/B comparable: a per-token loss summed over
    units must equal the same loss computed on the parent in one piece."""
    lengths = [40, 12, 16]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16)

    # A stand-in for the per-token loss: the masked sum of |logit|, where the
    # unit's logit for row r is just its token id (any deterministic function of
    # the row works -- what is under test is that no row is lost or doubled).
    def loss(rows_mask, rows_tokens):
        return (rows_mask.float() * rows_tokens.float()).sum()

    whole = loss(parent["full_loss_masks"][0], parent["tokens"][0])

    total = torch.zeros(())
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        real = batch["response_lengths"][0]
        total += loss(batch["full_loss_masks"][0][:real], batch["tokens"][0][:real])

    torch.testing.assert_close(total, whole)


def test_unit_token_normalizers_sum_to_the_parent_without_clamping_empty_chunks():
    """Prompt-only chunks contribute zero, not one, to step normalization."""
    lengths = [40]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16)

    counts = [slice_parent_batch(parent, unit, pad_token_id=PAD)["__num_tokens_override__"] for unit in units]

    assert any(count.item() == 0 for count in counts), "fixture must include a fully masked prompt chunk"
    torch.testing.assert_close(torch.stack(counts).sum(), parent["full_loss_masks"].sum())


def test_rope_offset_is_applied_only_when_there_is_a_prefix():
    lengths = [40]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16)

    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        packed = batch["packed_seq_params"]
        prefix = getattr(packed, cf_rope.PREFIX_ATTR, 0)
        headroom = getattr(packed, cf_rope.HEADROOM_ATTR, 0)

        if prefix:
            assert headroom > 0, "a chunk with a prefix must carry the RoPE offset"
            assert packed.cu_seqlens_q_padded[0].item() == prefix
        else:
            assert headroom == 0, "the first chunk must stay on the stock path"
            assert packed.cu_seqlens_q_padded is None


def test_packed_seq_params_stay_chunk_local_for_the_kernels():
    """Attention must see 0-based cu_seqlens even when RoPE needs absolute
    ones."""
    lengths = [40]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16)

    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        packed = batch["packed_seq_params"]
        assert packed.cu_seqlens_q_cpu[0] == 0
        assert packed.cu_seqlens_q_cpu[-1] == batch["response_lengths"][0]


def test_padding_is_applied_without_counting_toward_the_loss():
    lengths = [40]
    parent = _parent(lengths)
    units, _ = _units(lengths, 16)

    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        tokens = batch["tokens"][0]
        real = batch["response_lengths"][0]
        if real < tokens.numel():
            assert (tokens[real:] == PAD).all()
            # The full mask is padded to match tokens; Response length stays real.
            assert batch["full_loss_masks"].shape == batch["tokens"].shape


def test_rope_offsets_are_local_to_each_original_sequence():
    units, _ = _units([40, 40], 16)
    second_sequence_units = [unit for unit in units if unit.chunks[0].segments[0].seq_id == 1]

    assert [_rope_offsets(unit) for unit in second_sequence_units] == [[0], [16], [32]]


def test_lockstep_dependent_units_fail_closed_until_per_segment_rope_exists():
    units, _ = _units([40, 40], 16, "lockstep")
    merged = next(unit for unit in units if unit.is_dependent and len(unit.chunks) > 1)

    with pytest.raises(NotImplementedError, match="per-segment RoPE offsets"):
        _rope_offsets(merged)
