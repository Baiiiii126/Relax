# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Install ChunkFlow after model construction on each provider path.
# CLI arguments take precedence over the environment fallback.

from __future__ import annotations

import os

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _chunk_size_from_args() -> int | None:
    """Return CLI size, 0 when disabled, or None before args are available."""
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
    except Exception:  # noqa: BLE001 - called before set_args(); fall back to env
        return None
    if args is None:
        return None
    if not getattr(args, "use_chunkflow", False):
        return 0
    return int(getattr(args, "chunkflow_chunk_size", 0) or 0)


def _chunk_size_from_env() -> int:
    try:
        return int(os.environ.get("RELAX_CHUNKFLOW_CHUNK_SIZE") or 0)
    except ValueError:
        logger.warning("RELAX_CHUNKFLOW_CHUNK_SIZE is not an integer; chunkflow stays off.")
        return 0


def _chunk_size() -> int:
    from_args = _chunk_size_from_args()
    return from_args if from_args is not None else _chunk_size_from_env()


try:
    from relax.backends.megatron import model_provider as _mp
    from relax.backends.megatron.chunkflow.integration import install as _install_chunkflow

    _ORIGINAL_INSTALL_CP_PROBE = _mp._install_cp_probe

    def _install_cp_probe_with_chunkflow(model):
        _ORIGINAL_INSTALL_CP_PROBE(model)
        chunk_size = _chunk_size()
        if chunk_size > 0:
            _install_chunkflow(model, chunk_size)

    _mp._install_cp_probe = _install_cp_probe_with_chunkflow
    logger.info(
        "Relax chunkflow attention patch applied to model_provider._install_cp_probe "
        "(enable with --use-chunkflow --chunkflow-chunk-size <tokens>, or "
        "RELAX_CHUNKFLOW_CHUNK_SIZE=<tokens> as a fallback; the effective value is read "
        "at model-build time)."
    )
except ImportError as exc:
    logger.warning(
        "Relax chunkflow attention patch not applied — import failed (%r). "
        "Chunked attention will be unavailable; training falls back to standard attention.",
        exc,
    )
