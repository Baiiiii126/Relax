# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for chunkflow's StateStore.

Pure-tensor, no GPU / no distributed init needed. Covers:

1. Ordering invariants — put must be strictly ascending from 0, free must be
   strictly descending from the highest live id, and lookup must fail loudly
   on any missing/freed chunk instead of returning a partial result.
2. Content correctness — lookup returns exactly what was put, in chunk order.
3. The "sawtooth" memory profile — ascending put builds storage up, then
   descending free tears it back down to empty, with the peak occurring
   right when the store is fully built.
"""

from __future__ import annotations

import pytest
import torch


try:
    from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime, require_runtime
    from relax.backends.megatron.chunkflow.state_store import ChunkState, StateStore, StateStoreError
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


def _mk_state(seed: int, size: int = 4) -> ChunkState:
    g = torch.Generator().manual_seed(seed)
    return ChunkState({"k": torch.randn(size, size, generator=g), "v": torch.randn(size, size, generator=g)})


# ---------------------------------------------------------------------------
# 1. Ordering invariants
# ---------------------------------------------------------------------------


def test_put_must_start_at_zero():
    store = StateStore()
    with pytest.raises(StateStoreError):
        store.put(1, _mk_state(1))


def test_put_must_be_strictly_ascending():
    store = StateStore()
    store.put(0, _mk_state(0))
    with pytest.raises(StateStoreError):
        store.put(2, _mk_state(2))  # skipped chunk 1
    with pytest.raises(StateStoreError):
        store.put(0, _mk_state(0))  # repeat


def test_free_must_target_highest_live_chunk():
    store = StateStore()
    for i in range(4):
        store.put(i, _mk_state(i))

    with pytest.raises(StateStoreError):
        store.free(0)  # must free chunk 3 first (descending)

    store.free(3)
    store.free(2)
    with pytest.raises(StateStoreError):
        store.free(2)  # already freed
    store.free(1)
    store.free(0)


def test_free_on_empty_store_raises():
    store = StateStore()
    with pytest.raises(StateStoreError):
        store.free(0)


def test_lookup_missing_or_freed_chunk_raises():
    store = StateStore()
    store.put(0, _mk_state(0))
    with pytest.raises(StateStoreError):
        store.lookup(1)  # chunk 1 was never put

    store.put(1, _mk_state(1))
    store.put(2, _mk_state(2))
    store.free(2)
    with pytest.raises(StateStoreError):
        store.lookup(2)  # chunk 2 was just freed


# ---------------------------------------------------------------------------
# 2. Content correctness
# ---------------------------------------------------------------------------


def test_lookup_returns_exact_content_in_order():
    store = StateStore()
    states = [_mk_state(i) for i in range(4)]
    for i, s in enumerate(states):
        store.put(i, s)

    looked_up = store.lookup(2)
    assert len(looked_up) == 3
    for i, s in enumerate(looked_up):
        torch.testing.assert_close(s.tensors["k"], states[i].tensors["k"])
        torch.testing.assert_close(s.tensors["v"], states[i].tensors["v"])


def test_live_chunk_ids_reflects_put_and_free():
    store = StateStore()
    for i in range(4):
        store.put(i, _mk_state(i))
    assert store.live_chunk_ids == [0, 1, 2, 3]

    store.free(3)
    assert store.live_chunk_ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# 3. Sawtooth memory profile: ascending build-up, descending tear-down, peak
#    right when fully built, empty at the end.
# ---------------------------------------------------------------------------


def test_ascending_put_then_descending_free_is_a_sawtooth():
    num_chunks = 6
    store = StateStore()

    sizes_after_put = []
    for i in range(num_chunks):
        store.put(i, _mk_state(i, size=4 + i))  # varying size, like real varying chunk lengths
        sizes_after_put.append(store.num_elements)

    peak = store.num_elements
    assert peak > 0
    assert sizes_after_put == sorted(sizes_after_put)  # only grows while building up

    sizes_after_free = []
    for i in reversed(range(num_chunks)):
        store.free(i)
        sizes_after_free.append(store.num_elements)

    assert sizes_after_free == sorted(sizes_after_free, reverse=True)  # only shrinks while tearing down
    assert sizes_after_free[-1] == 0
    assert len(store) == 0
    assert sizes_after_put[-1] == peak  # peak occurs exactly when fully built, before any free


def test_schedule_reuses_runtime_installed_below_model_wrappers():
    class Wrapper:
        def __init__(self, module):
            self.module = module

    bare = Wrapper(None)
    runtime = ChunkflowRuntime(chunk_size=16)
    bare.chunkflow_runtime = runtime
    wrapped = Wrapper(Wrapper(bare))

    assert require_runtime(wrapped) is runtime
    assert not hasattr(wrapped, "chunkflow_runtime")
