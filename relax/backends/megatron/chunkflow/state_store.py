# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local, communication-free per-chunk state storage for chunkflow.

chunkflow splits a long sequence into causally-dependent chunks: chunk ``i``'s
attention needs the K/V (or, for other attention structures, whatever
per-chunk state a forward pass produces) of every chunk ``0..i``. Because of
this dependency, forward must visit chunks in ascending order (a chunk's
state only exists once its forward has run) and backward must visit them in
descending order: key-chunk ``m`` is read by every query chunk ``j >= m``, so
in a descending backward the highest live chunk is always the last (and
therefore only safe) one to free next.

``StateStore`` does not interpret what is inside a chunk's state — callers
(softmax-attention chunking, DSA chunking, ...) decide what to store. It only
enforces the ascending-put / descending-free ordering invariant, so a
scheduling bug surfaces immediately as a ``StateStoreError`` instead of
silently producing wrong gradients later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkState:
    """Opaque per-chunk payload.

    ``tensors`` holds whatever a chunk's forward pass needs to keep alive until
    its backward runs — e.g. ``{"k": ..., "v": ...}`` for softmax attention, or
    ``{"kv": ..., "indices": ...}`` for DSA.
    """

    tensors: dict[str, torch.Tensor]

    @property
    def num_elements(self) -> int:
        """Total element count across all stored tensors (memory-accounting
        helper for tests)."""
        return sum(t.numel() for t in self.tensors.values())


class StateStoreError(RuntimeError):
    """A StateStore call violated the chunkflow ordering invariant (ascending
    put, descending free)."""


class StateStore:
    """Local per-chunk KV/state store with incremental put, lookup, and
    decremental free.

    Chunk ids must start at 0 and be put in strictly ascending order; they must
    be freed in strictly descending order (highest live id first). Both rules
    are enforced with hard errors rather than tolerated silently, because a
    violation means the caller's chunk schedule is wrong and any gradient
    computed downstream would be silently incorrect.
    """

    def __init__(self) -> None:
        self._chunks: dict[int, ChunkState] = {}
        self._next_expected_put: int = 0

    def put(self, chunk_id: int, state: ChunkState) -> None:
        """Store ``state`` for ``chunk_id`` (增量存储).

        Must be called with ``chunk_id == 0, 1, 2, ...`` in order — each
        chunk's forward only has the previous chunks' state available once they
        have actually been put.
        """
        if chunk_id != self._next_expected_put:
            raise StateStoreError(
                f"put() must be strictly ascending starting at 0: "
                f"expected chunk_id={self._next_expected_put}, got {chunk_id}"
            )
        self._chunks[chunk_id] = state
        self._next_expected_put += 1

    def lookup(self, upto_chunk_id: int) -> list[ChunkState]:
        """Return the states for chunks ``0..upto_chunk_id`` inclusive,
        ascending (查找).

        Raises if any chunk in that range is missing (never put, or already
        freed) — a query chunk should never need a chunk that has been freed,
        so this signals a scheduling bug rather than silently returning a
        partial or wrong result.
        """
        missing = [i for i in range(upto_chunk_id + 1) if i not in self._chunks]
        if missing:
            raise StateStoreError(
                f"lookup(upto_chunk_id={upto_chunk_id}) requested missing/freed chunk id(s) {missing}"
            )
        return [self._chunks[i] for i in range(upto_chunk_id + 1)]

    def free(self, chunk_id: int) -> None:
        """Release the state for ``chunk_id`` (减量删除).

        Must be called in strictly descending order relative to the highest
        currently-live chunk id.
        """
        if not self._chunks:
            raise StateStoreError("free() called on an empty StateStore")
        current_max = max(self._chunks)
        if chunk_id != current_max:
            raise StateStoreError(
                f"free() must remove the highest live chunk id first (descending): expected {current_max}, "
                f"got {chunk_id}"
            )
        del self._chunks[chunk_id]

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def live_chunk_ids(self) -> list[int]:
        """Currently-stored chunk ids, ascending."""
        return sorted(self._chunks)

    @property
    def num_elements(self) -> int:
        """Total element count across all currently-live chunk states."""
        return sum(state.num_elements for state in self._chunks.values())
