# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A chunk must get the RoPE rotation its tokens have in the full sequence.

Feeding a chunk as its own packed batch makes RoPE read ``cu_seqlens[0]`` as the
sequence start, so positions restart at 0 — silently wrong scores for any chunk
with a non-zero prefix, and the reason ``chunkflow/rope.py`` exists at all.

GPU-gated: RoPE's packed path needs a real device and an initialised CP group.
"""

from __future__ import annotations

import os

import pytest
import torch


try:
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.packed_seq_params import PackedSeqParams

    from relax.backends.megatron.chunkflow import rope as cf_rope
except Exception as exc:
    pytest.skip(f"chunkflow rope unavailable: {exc}", allow_module_level=True)


HEAD_DIM = 64
TOTAL = 512
CHUNK = 128
PREFIX = 256  # test the third chunk: a non-zero prefix, and not the first


class _Cfg:
    """Only the attributes RotaryEmbedding reads."""

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
        pytest.skip("requires CUDA")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29593")
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    cf_rope.install()
    yield
    cf_rope.uninstall()


def _rotated(rope, tensor, packed, *, use_kv: bool):
    """Apply RoPE the way the attention module would, for q or k."""
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
def test_chunk_rope_positions_match_the_full_sequence(rope_module, use_kv):
    """The load-bearing property: identical rotation for the same tokens.

    Checked on the query side and the key side separately, because Megatron
    gates the position offset on ``freqs.size(0) > max_seqlen`` and the two
    sides use different ``max_seqlen`` values — sizing that satisfies one can
    silently fail the other.
    """
    device = "cuda"
    torch.manual_seed(0)
    rope = RotaryEmbedding(kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_base=1_000_000).to(device)

    full = _packed(device, [0, TOTAL], TOTAL, TOTAL)
    tensor = torch.randn(TOTAL, 4, HEAD_DIM, device=device, dtype=torch.bfloat16)
    reference = _rotated(rope, tensor, full, use_kv=use_kv)[PREFIX : PREFIX + CHUNK]

    chunk = _packed(device, [0, CHUNK], CHUNK, CHUNK)
    cf_rope.apply_rope_offset(chunk, offsets=[PREFIX], lengths=[CHUNK])
    got = _rotated(rope, tensor[PREFIX : PREFIX + CHUNK].clone(), chunk, use_kv=use_kv)

    # Exact, not approximate: it is the same table and the same positions.
    torch.testing.assert_close(got.float(), reference.float(), rtol=0, atol=0)


def test_without_the_offset_the_positions_are_wrong(rope_module):
    """Negative control — without it the test above could pass for the wrong
    reason.

    A chunk with no offset marker keeps the stock sizing, positions restart at
    0, and the rotation differs materially from the full-sequence reference.
    """
    device = "cuda"
    torch.manual_seed(0)
    rope = RotaryEmbedding(kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_base=1_000_000).to(device)

    full = _packed(device, [0, TOTAL], TOTAL, TOTAL)
    tensor = torch.randn(TOTAL, 4, HEAD_DIM, device=device, dtype=torch.bfloat16)
    reference = _rotated(rope, tensor, full, use_kv=False)[PREFIX : PREFIX + CHUNK]

    cf_rope.uninstall()
    try:
        plain = _packed(device, [0, CHUNK], CHUNK, CHUNK)
        wrong = _rotated(rope, tensor[PREFIX : PREFIX + CHUNK].clone(), plain, use_kv=False)
    finally:
        cf_rope.install()

    rel = ((wrong.float() - reference.float()).abs().max() / reference.float().abs().max()).item()
    assert rel > 0.1, f"positions-0 should differ from the true positions, got rel err {rel:.3e}"


def test_offset_marks_both_sides_and_covers_the_positions(rope_module):
    """The marker must make the shift fire on q *and* kv, and the table must
    reach the highest absolute position."""
    device = "cuda"
    packed = _packed(device, [0, CHUNK], CHUNK, CHUNK)
    assert getattr(packed, cf_rope.HEADROOM_ATTR, 0) == 0  # absent before the call

    cf_rope.apply_rope_offset(packed, offsets=[PREFIX], lengths=[CHUNK])

    assert getattr(packed, cf_rope.HEADROOM_ATTR) > 0
    assert packed.cu_seqlens_q_cpu == [0, CHUNK]  # attention stays chunk-local
    assert packed.cu_seqlens_q_padded.tolist() == [PREFIX, PREFIX + CHUNK]  # RoPE gets absolute
    assert packed.max_seqlen_kv == PREFIX + CHUNK

    rope = RotaryEmbedding(kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_base=1_000_000).to(device)
    table_len = rope.get_rotary_seq_len(None, None, None, _Cfg(), packed)
    assert table_len > packed.max_seqlen_q, "q-side shift gate would not fire"
    assert table_len > packed.max_seqlen_kv, "kv-side shift gate would not fire"
    assert table_len >= PREFIX + CHUNK, "table must cover the highest absolute position"


def test_offset_is_a_noop_for_a_zero_prefix(rope_module):
    """The first chunk is unchanged — it is what every batch without the marker
    already does, so this also pins the no-op guarantee."""
    device = "cuda"
    torch.manual_seed(0)
    rope = RotaryEmbedding(kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_base=1_000_000).to(device)

    tensor = torch.randn(TOTAL, 4, HEAD_DIM, device=device, dtype=torch.bfloat16)
    full = _packed(device, [0, TOTAL], TOTAL, TOTAL)
    reference = _rotated(rope, tensor, full, use_kv=False)[:CHUNK]

    first = _packed(device, [0, CHUNK], CHUNK, CHUNK)
    cf_rope.apply_rope_offset(first, offsets=[0], lengths=[CHUNK])
    got = _rotated(rope, tensor[:CHUNK].clone(), first, use_kv=False)

    torch.testing.assert_close(got.float(), reference.float(), rtol=0, atol=0)
