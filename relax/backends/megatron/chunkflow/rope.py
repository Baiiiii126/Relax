# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Give a chunk's tokens their true RoPE positions when the chunk is its own
forward.

A chunk of a long sequence presented as a standalone packed batch looks like a
sequence starting at token 0. RoPE then computes

    freq_pos = local_pos + global_seq_start,   global_seq_start = cu_seqlens[0]

with ``cu_seqlens = [0, L]`` that is ``local_pos`` — positions 0..L-1 for a
chunk whose real positions are ``prefix..prefix+L-1``. Nothing errors; the
attention scores are simply wrong. (Chunking *inside* one attention forward, as
``ChunkedCoreAttention`` does today, never hits this: RoPE has already run on
the whole packed batch by then.)

Megatron exposes the two things needed, but the stock sizing makes them
unusable together:

* ``attention.py`` applies RoPE over ``cu_seqlens_q_padded``/``cu_seqlens_kv_padded``
  when present — so the *offset* can be supplied without touching attention.
* ``rope_utils._apply_rotary_pos_emb_thd`` only adds that offset when
  ``freqs.size(0) > max_seqlen``, and ``freqs.size(0)`` is
  ``max(max_seqlen_q, max_seqlen_kv)``. That is never strictly greater than
  both, so the query side would shift and the key side would not — the same
  token getting two different positions.

So ``install()`` extends the table by a headroom instead, which makes the
comparison strict on both sides. Everything is driven by a marker attribute on
``PackedSeqParams``; when a batch has no marker, ``headroom`` is 0, the table
is sized exactly as before, and the shift never fires — every existing path is
bit-for-bit unchanged.
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
