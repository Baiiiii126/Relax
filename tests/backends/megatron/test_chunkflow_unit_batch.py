# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit slicing, loss normalization and sequence-local RoPE positions."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch


try:
    from megatron.core.packed_seq_params import PackedSeqParams

    from relax.backends.megatron.chunkflow import rope as cf_rope
    from relax.backends.megatron.chunkflow.scheduler import build_chunks
    from relax.backends.megatron.chunkflow.unit_batch import _rope_offsets, _rows_for_unit, slice_parent_batch
    from relax.backends.megatron.chunkflow.units import build_micro_batches
except ImportError as exc:
    pytest.skip(f"ChunkFlow batch dependencies unavailable: {exc}", allow_module_level=True)


PAD = 0


def _parent(lengths: list[int], device="cpu") -> dict:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    total = cu[-1]

    tokens = torch.arange(1, total + 1, device=device).unsqueeze(0)
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


@pytest.mark.parametrize("policy", ["staggered", "lockstep"])
def test_units_preserve_tokens_targets_masks_and_loss(policy):
    parent = _parent([40, 12, 16])
    units, _ = _units([40, 12, 16], 16, policy)
    tokens, mask = parent["tokens"][0], parent["full_loss_masks"][0]
    cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    seen, counts, total_loss = [], [], torch.zeros(())
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        real = batch["response_lengths"][0]
        rows = _rows_for_unit(cu, unit)
        expected_tokens = torch.cat([tokens[start:end] for start, end in rows])
        expected_mask = torch.cat([mask[start:end] for start, end in rows])
        torch.testing.assert_close(batch["tokens"][0][:real], expected_tokens)
        torch.testing.assert_close(batch["full_loss_masks"][0][:real], expected_mask)
        targets = []
        segments = [seg for chunk in unit.chunks for seg in chunk.segments]
        for (start, end), seg in zip(rows, segments, strict=True):
            part = tokens[start + 1 : min(end + 1, cu[seg.seq_id + 1])]
            if part.numel() < end - start:
                part = torch.cat([part, tokens.new_full((end - start - part.numel(),), PAD)])
            targets.append(part)
        torch.testing.assert_close(batch["unconcat_tokens"][0][:real], torch.cat(targets))
        assert batch["full_loss_masks"].shape == batch["tokens"].shape
        assert (batch["tokens"][0][real:] == PAD).all()
        assert (batch["full_loss_masks"][0][real:] == 0).all()
        counts.append(batch["__num_tokens_override__"])
        seen.extend(batch["tokens"][0][:real].tolist())
        total_loss += (batch["full_loss_masks"][0].float() * batch["tokens"][0].float()).sum()
    assert any(count.item() == 0 for count in counts), "fixture must include a prompt-only unit"
    torch.testing.assert_close(torch.stack(counts).sum(), parent["full_loss_masks"].sum())
    assert sorted(seen) == tokens.tolist()
    assert len(seen) == len(set(seen))
    torch.testing.assert_close(total_loss, (mask.float() * tokens.float()).sum())


def test_attention_lengths_and_rope_offsets_are_sequence_local():
    parent = _parent([40, 40])
    units, _ = _units([40, 40], 16)
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        packed = batch["packed_seq_params"]
        real = batch["response_lengths"][0]
        prefix = unit.chunks[0].segments[0].prefix
        assert _rope_offsets(unit) == [prefix]
        assert packed.cu_seqlens_q_cpu == [0, real]
        assert getattr(packed, cf_rope.PREFIX_ATTR, 0) == prefix
        if prefix:
            assert getattr(packed, cf_rope.HEADROOM_ATTR, 0) > 0
            assert packed.cu_seqlens_q_padded.tolist() == [prefix, prefix + real]
            assert packed.max_seqlen_kv == prefix + real
        else:
            assert packed.cu_seqlens_q_padded is None


def test_lockstep_dependent_units_fail_closed_until_per_segment_rope_exists():
    units, _ = _units([40, 40], 16, "lockstep")
    merged = next(unit for unit in units if unit.is_dependent and len(unit.chunks) > 1)

    with pytest.raises(NotImplementedError, match="per-segment RoPE offsets"):
        _rope_offsets(merged)


def test_bridge_chunk_positions_and_padding_match_full_sequence():
    bridge_rope = pytest.importorskip("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope")
    from relax.backends.megatron.chunkflow.schedules import _forward_kwargs

    parent = _parent([301, 301, 11, 13])
    units, _ = _units(parent["total_lengths"], 128)
    full_positions, _ = bridge_rope.get_rope_index(
        1, -1, -2, -3, parent["tokens"], packed_seq_params=parent["packed_seq_params"]
    )
    rope = bridge_rope.Qwen3VLMultimodalRotaryEmbedding.__new__(bridge_rope.Qwen3VLMultimodalRotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.inv_freq = 1 / (1_000_000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    rope.seq_len_interpolation_factor = None
    rope.cp_group = SimpleNamespace(size=lambda: 1)
    rope.is_thd_format = True
    config = SimpleNamespace(apply_rope_fusion=False, apply_rotary_pos_emb_in_fp32=True, rotary_interleaved=False)
    full_input = torch.randn(parent["tokens"].numel(), 2, 64, requires_grad=True)
    full_output = bridge_rope.apply_rotary_pos_emb_absolute(
        full_input, rope(full_positions, [11, 11, 10]), config, parent["packed_seq_params"].cu_seqlens_q
    )
    full_output.sum().backward()
    saw_packed = saw_prefix = saw_padding = False
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD, args=SimpleNamespace(uses_unsplit_forward=True))
        packed = batch["packed_seq_params"]
        rows = _rows_for_unit(parent["packed_seq_params"].cu_seqlens_q_cpu, unit)
        real = sum(end - start for start, end in rows)
        physical = batch["tokens"].numel()
        expected = torch.cat([full_positions[..., start:end] for start, end in rows], dim=-1)
        torch.testing.assert_close(batch["position_ids"][..., :real], expected)
        assert batch["position_ids"].shape == (3, 1, physical)
        assert (batch["position_ids"][..., real:] == 1).all()
        assert _forward_kwargs(batch)["position_ids"] is batch["position_ids"]
        physical_cu = packed.cu_seqlens_q_padded.tolist()
        assert physical_cu == [*packed.cu_seqlens_q_cpu[:-1], physical]
        assert packed.max_seqlen_q == max(b - a for a, b in zip(physical_cu[:-1], physical_cu[1:]))
        mask = bridge_rope.get_packed_seq_attention_mask(batch["tokens"], packed)
        assert mask.sum() == real
        assert not mask[..., real:].any()
        chunk_input = torch.cat([full_input.detach()[start:end] for start, end in rows])
        chunk_input = torch.cat([chunk_input, chunk_input.new_zeros(physical - real, 2, 64)]).requires_grad_()
        output = bridge_rope.apply_rotary_pos_emb_absolute(
            chunk_input, rope(batch["position_ids"], [11, 11, 10], packed), config, packed.cu_seqlens_q_padded
        )
        torch.testing.assert_close(output[:real], torch.cat([full_output[start:end] for start, end in rows]))
        output[:real].sum().backward()
        torch.testing.assert_close(
            chunk_input.grad[:real], torch.cat([full_input.grad[start:end] for start, end in rows])
        )
        saw_packed |= len(rows) > 1
        saw_prefix |= any(_rope_offsets(unit))
        saw_padding |= physical > real
    assert saw_packed and saw_prefix and saw_padding


HEAD_DIM = 64


TOTAL = 512


CHUNK = 128


PREFIX = 256


class _Cfg:
    rotary_base = 1_000_000
    rotary_percent = 1.0
    rotary_interleaved = False
    seq_len_interpolation_factor = None
    apply_rope_fusion = False
    mscale = 1.0
    mscale_all_dim = 1.0


@pytest.fixture(scope="module")
def rope_module():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and Megatron model-parallel groups")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    own_dist = not dist.is_initialized()
    own_parallel = not parallel_state.is_initialized()
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(cf_rope, "_ORIGINAL_GET_ROTARY_SEQ_LEN", cf_rope._ORIGINAL_GET_ROTARY_SEQ_LEN)
        patcher.setattr(RotaryEmbedding, "get_rotary_seq_len", RotaryEmbedding.get_rotary_seq_len)
        if own_dist:
            patcher.setenv("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
            patcher.setenv("MASTER_PORT", os.environ.get("MASTER_PORT", "29593"))
        try:
            if own_dist:
                dist.init_process_group("nccl", rank=0, world_size=1)
            if own_parallel:
                parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            cf_rope.install()
            yield RotaryEmbedding
        finally:
            if own_parallel and parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            if own_dist and dist.is_initialized():
                dist.destroy_process_group()


def _rotated(rope, tensor, packed, *, use_kv: bool):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    max_seqlen = packed.max_seqlen_kv if use_kv else packed.max_seqlen_q
    cu = packed.cu_seqlens_kv_padded if use_kv else packed.cu_seqlens_q_padded
    if cu is None:
        cu = packed.cu_seqlens_kv if use_kv else packed.cu_seqlens_q

    table_len = rope.get_rotary_seq_len(None, None, None, _Cfg(), packed)
    return apply_rotary_pos_emb(
        tensor,
        rope(table_len, packed_seq=True),
        config=_Cfg(),
        cu_seqlens=cu,
        mscale=1.0,
        cp_group=None,
        max_seqlen=max_seqlen,
    )


def _packed(device, lengths, max_q, max_kv):
    cu = torch.tensor(lengths, dtype=torch.int32, device=device)
    return PackedSeqParams(
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max_q, max_seqlen_kv=max_kv, qkv_format="thd"
    )


@pytest.mark.parametrize("use_kv", [False, True])
@pytest.mark.parametrize("prefix", [0, PREFIX])
def test_chunk_rope_positions_match_the_full_sequence(rope_module, use_kv, prefix):
    device = "cuda"
    torch.manual_seed(0)
    rope = rope_module(kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_base=1_000_000).to(device)

    full = _packed(device, [0, TOTAL], TOTAL, TOTAL)
    tensor = torch.randn(TOTAL, 4, HEAD_DIM, device=device, dtype=torch.bfloat16)
    reference = _rotated(rope, tensor, full, use_kv=use_kv)[prefix : prefix + CHUNK]

    chunk = _packed(device, [0, CHUNK], CHUNK, CHUNK)
    cf_rope.apply_rope_offset(chunk, offsets=[prefix], lengths=[CHUNK])
    got = _rotated(rope, tensor[prefix : prefix + CHUNK].clone(), chunk, use_kv=use_kv)

    torch.testing.assert_close(got.float(), reference.float(), rtol=0, atol=0)
