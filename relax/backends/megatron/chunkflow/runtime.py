# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-rank state that has to outlive a single forward call.

``StateStore`` is per (layer, dependency group): chunk ``i`` reads chunks
``0..i`` of that layer's K/V, and chunk ``i+1`` needs them again after chunk
``i``'s forward has returned. Stores live here, hung off the built model, and
appear with a group's first chunk and go away with its last backward
(``release_group``); ``assert_empty`` turns a leaked store into an error. Which
unit is executing travels on ``packed_seq_params``, not through a global.
"""

from __future__ import annotations

from relax.backends.megatron.chunkflow.state_store import StateStore
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

RUNTIME_ATTR = "chunkflow_runtime"
UNIT_ATTR = "chunkflow_unit"
# Set on packed_seq_params for an Fr op: this chunk's K/V leaf already lives in
# the store from its Fp pass, so the attention must reuse it instead of putting
# again (which would trip the store's ascending invariant).
RECOMPUTE_ATTR = "chunkflow_recompute"


class ChunkflowRuntime:
    """``(layer_number, group_id) -> StateStore`` for one model on one rank."""

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self._stores: dict[tuple[int, int], StateStore] = {}
        # Set by the schedule for the duration of one unit's forward; cleared
        # right after so a stray read cannot silently use the wrong group.
        self.current_unit = None
        # Group ids must be unique for the runtime's whole lifetime, not just
        # one schedule call -- the stores persist across calls, and resetting the
        # counter per call reuses ids and collides with a store an earlier call
        # left behind.
        self._group_offset = 0

    def next_group_offset(self, used_max: int | None) -> int:
        """Advance the global group counter past ``used_max``; returns the value to start from next time.

        ``used_max`` is the highest group id this micro-batch used (already
        offset), or None when it had none. The returned value is what the next
        micro-batch starts from, so ids are unique across schedule calls.
        """
        if used_max is not None:
            self._group_offset = used_max + 1
        return self._group_offset

    def store(self, layer_number: int, group_id: int) -> StateStore:
        """The store for one layer of one group, created on first use."""
        key = (layer_number, group_id)
        store = self._stores.get(key)
        if store is None:
            store = StateStore()
            self._stores[key] = store
        return store

    def release_group(self, group_id: int) -> int:
        """Drop every layer's store for a finished group.

        Returns how many.
        """
        keys = [k for k in self._stores if k[1] == group_id]
        for key in keys:
            del self._stores[key]
        return len(keys)

    def live_groups(self) -> list[int]:
        return sorted({group for _, group in self._stores})

    def num_stores(self) -> int:
        return len(self._stores)

    def num_elements(self) -> int:
        """Total live state elements -- the K/V this rank is holding."""
        return sum(store.num_elements for store in self._stores.values())

    def assert_empty(self, where: str) -> None:
        """Fail loudly if a step ended without releasing its groups.

        A leak here is a schedule bug that would otherwise show up much later
        as unexplained memory growth.
        """
        if self._stores:
            groups = self.live_groups()
            raise RuntimeError(
                f"chunkflow: {len(self._stores)} store(s) still live after {where} "
                f"(groups {groups}). A dependency group's last backward must call release_group()."
            )


def attach_runtime(model, chunk_size: int) -> ChunkflowRuntime:
    """Create (or return) the runtime for a built model."""
    runtime = getattr(model, RUNTIME_ATTR, None)
    if runtime is None:
        runtime = ChunkflowRuntime(chunk_size)
        setattr(model, RUNTIME_ATTR, runtime)
    return runtime


def find_runtime(model) -> ChunkflowRuntime | None:
    """The runtime on a model, unwrapping Megatron's module wrappers.

    The schedule is handed the same object the model provider built, but after
    DDP/`Float16Module` wrapping the attribute lives on an inner module — walk
    down until it is found, the way Megatron's own ``get_attr_wrapped_model``
    does.
    """
    seen = 0
    while model is not None and seen < 8:
        runtime = getattr(model, RUNTIME_ATTR, None)
        if runtime is not None:
            return runtime
        model = getattr(model, "module", None)
        seen += 1
    return None
