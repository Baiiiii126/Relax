# Install chunkflow's chunked attention on every freshly built model when
# ``RELAX_CHUNKFLOW_CHUNK_SIZE`` is set to a positive integer.
#
# Hook point: ``model_provider._install_cp_probe`` runs right after the model is
# constructed on all three build paths (custom provider, bridge, raw), so
# wrapping it reaches every path without touching them. The wrapper only adds a
# call; the original probe still runs.
#
# Off by default: unset or <=0 leaves the model untouched, so importing this
# module is a no-op for every existing run. The env var is read at model-build
# time, not import time, because Megatron's args do not exist yet at import.
#
# Numerically chunking is equivalent to unchunked attention (verified against
# the real Qwen3-0.6B for chunk sizes 4..64). It does not reduce memory on its
# own; that needs the per-chunk backward schedule.

import os

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _chunk_size() -> int:
    try:
        return int(os.environ.get("RELAX_CHUNKFLOW_CHUNK_SIZE") or 0)
    except ValueError:
        logger.warning("RELAX_CHUNKFLOW_CHUNK_SIZE is not an integer; chunkflow stays off.")
        return 0


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
        "(enable with RELAX_CHUNKFLOW_CHUNK_SIZE=<tokens>; currently "
        f"{'on, chunk_size=' + str(_chunk_size()) if _chunk_size() > 0 else 'off'})."
    )
except ImportError as exc:
    logger.warning(
        "Relax chunkflow attention patch not applied — import failed (%r). "
        "Chunked attention will be unavailable; training falls back to standard attention.",
        exc,
    )
