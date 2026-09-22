# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Give a chunk's tokens their true RoPE positions when it is its own forward.

A chunk presented as a standalone packed batch looks like a sequence starting
at token 0, so RoPE restarts positions at 0 instead of ``prefix..prefix+L-1`` --
silently wrong scores. (Chunking *inside* one attention forward never hits
this, because RoPE has already run on the whole batch.)

``apply_rope_offset()`` keeps ``cu_seqlens_q/kv`` chunk-local for the attention
kernels and puts the absolute positions in the ``*_padded`` fields RoPE reads,
with a headroom marker so the frequency table covers them. ``install()``
extends the table by that headroom. Without the marker, every existing path is
unchanged. The fused RoPE kernel takes the padded cu_seqlens literally and
reads past the table, so chunkflow also disables ``apply_rope_fusion``.
"""

from __future__ import annotations

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Markers read off ``PackedSeqParams``. Absent on every batch that is not a
# chunkflow unit, which is what keeps this a no-op elsewhere.
HEADROOM_ATTR = "chunkflow_rope_headroom"
PREFIX_ATTR = "chunkflow_rope_prefix"

_ORIGINAL_GET_ROTARY_SEQ_LEN = None


def apply_rope_offset(
    packed_seq_params,
    offsets: list[int],
    lengths: list[int],
    headroom: int = 1,
) -> None:
    """Mark ``packed_seq_params`` so RoPE places each packed sequence at its
    real position.

    ``offsets[i]`` is the number of tokens of the original sequence that precede
    packed sequence ``i``; ``lengths[i]`` is its length.

    ``cu_seqlens_q``/``cu_seqlens_kv`` stay chunk-local (0-based) for the
    attention kernels, while the ``*_padded`` fields — which only RoPE consumes
    — carry the absolute positions. ``max_seqlen_kv`` is set to the largest
    absolute end so the frequency table covers every position the offset will
    reach.
    """
    import torch

    absolute = [0]
    for length in lengths:
        absolute.append(absolute[-1] + length)
    # The split points in absolute coordinates are not contiguous when a unit
    # packs segments from different sequences, so each segment gets its own
    # offset rather than accumulating.
    padded = [offsets[0]]
    for offset, length in zip(offsets, lengths, strict=True):
        padded.append(offset + length)

    packed_seq_params.cu_seqlens_q = torch.tensor(
        absolute, dtype=torch.int32, device=packed_seq_params.cu_seqlens_q.device
    )
    packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_q
    packed_seq_params.cu_seqlens_q_padded = torch.tensor(
        padded, dtype=torch.int32, device=packed_seq_params.cu_seqlens_q.device
    )
    packed_seq_params.cu_seqlens_kv_padded = packed_seq_params.cu_seqlens_q_padded
    packed_seq_params.cu_seqlens_q_cpu = absolute
    packed_seq_params.max_seqlen_q = max(lengths)
    packed_seq_params.max_seqlen_kv = max(padded[i + 1] for i in range(len(lengths)))

    setattr(packed_seq_params, HEADROOM_ATTR, headroom)
    setattr(packed_seq_params, PREFIX_ATTR, offsets[0])


def install() -> None:
    """Patch ``RotaryEmbedding.get_rotary_seq_len`` to honour the headroom
    marker.

    Only the table *length* changes, and only when a batch carries the marker
    (``headroom`` defaults to 0 when absent), so this cannot alter any existing
    run.
    """
    global _ORIGINAL_GET_ROTARY_SEQ_LEN
    if _ORIGINAL_GET_ROTARY_SEQ_LEN is not None:
        return

    try:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    except ImportError as exc:  # pragma: no cover - megatron always present in training
        logger.warning("chunkflow RoPE patch not applied -- import failed (%r).", exc)
        return

    _ORIGINAL_GET_ROTARY_SEQ_LEN = RotaryEmbedding.get_rotary_seq_len

    def get_rotary_seq_len(
        self,
        inference_context,
        transformer,
        transformer_input,
        transformer_config,
        packed_seq_params=None,
        *,
        inference_params=None,
    ):
        length = _ORIGINAL_GET_ROTARY_SEQ_LEN(
            self,
            inference_context,
            transformer,
            transformer_input,
            transformer_config,
            packed_seq_params,
            inference_params=inference_params,
        )
        headroom = getattr(packed_seq_params, HEADROOM_ATTR, 0)
        return length + headroom if headroom else length

    RotaryEmbedding.get_rotary_seq_len = get_rotary_seq_len
    logger.info(
        "chunkflow RoPE offset patch applied to RotaryEmbedding.get_rotary_seq_len (no-op unless a batch carries %s).",
        HEADROOM_ATTR,
    )


def uninstall() -> None:
    """Restore the original ``get_rotary_seq_len`` (for tests)."""
    global _ORIGINAL_GET_ROTARY_SEQ_LEN
    if _ORIGINAL_GET_ROTARY_SEQ_LEN is None:
        return
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    RotaryEmbedding.get_rotary_seq_len = _ORIGINAL_GET_ROTARY_SEQ_LEN
    _ORIGINAL_GET_ROTARY_SEQ_LEN = None
