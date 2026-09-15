# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for chunkflow's scheduler.

Pure Python, no torch / GPU / distributed init needed (the scheduler only
decides chunk construction and op ordering; it never touches a tensor).
Covers:

1. Chunk construction (Algorithm 1) — long sequences split ascending into a
   dependency group with correct prefixes; short sequences bin-packed;
   the tail-chunk fragmentation defect and ``repack_tail``'s fix for it.
2. Op-sequence generation (Algorithm 2, paper-prose semantics) — small
   groups keep every chunk's activation; large groups recompute the
   earlier ones in descending order.
3. The three validators, including a deliberately-broken "ascending drop
   order" schedule (the literal-pseudocode reading a real implementation
   must not use) to prove ``validate_backward_order`` actually catches it.
"""

from __future__ import annotations

import pytest


try:
    from relax.backends.megatron.chunkflow.scheduler import (
        Chunk,
        Segment,
        build_chunks,
        build_group_schedule,
        build_schedule,
        group_of,
        validate_backward_order,
        validate_completeness,
        validate_forward_order,
        validate_schedule,
    )
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# 1. Chunk construction (Algorithm 1)
# ---------------------------------------------------------------------------


def test_long_sequence_splits_ascending_with_correct_prefixes():
    chunks = build_chunks([2050], chunk_size=1024, repack_tail=False)
    dep = [c for c in chunks if c.is_dependent]
    dep.sort(key=lambda c: c.group_idx)

    assert [c.group_idx for c in dep] == [0, 1, 2]
    assert [c.group_size for c in dep] == [3, 3, 3]
    assert [c.num_tokens for c in dep] == [1024, 1024, 2]
    # prefix = how many tokens of this sequence precede this chunk
    assert [c.segments[0].prefix for c in dep] == [0, 1024, 2048]


def test_short_sequences_are_bin_packed_not_left_standalone():
    # 5 sequences of 300 tokens each, chunk_size 1024 -> should pack ~3 per chunk, not 1-per-chunk.
    chunks = build_chunks([300] * 5, chunk_size=1024, repack_tail=False)
    assert all(not c.is_dependent for c in chunks)
    assert len(chunks) < 5
    assert sum(c.num_tokens for c in chunks) == 300 * 5


def test_tail_defect_without_repack_produces_tiny_dependent_chunk():
    chunks = build_chunks([1026], chunk_size=1024, repack_tail=False)
    dep = [c for c in chunks if c.is_dependent]
    tail = max(dep, key=lambda c: c.group_idx)
    assert tail.num_tokens == 2  # the fragment lesson 3's readme called a "bubble factory"


def test_repack_tail_folds_tiny_tail_into_short_sequence_packing():
    # One long sequence with a 2-token tail, plus short sequences to pack it with.
    lengths = [1026] + [500] * 3
    without = build_chunks(lengths, chunk_size=1024, repack_tail=False)
    with_repack = build_chunks(lengths, chunk_size=1024, repack_tail=True)

    tiny_without = [c for c in without if c.is_dependent and c.num_tokens < 1024 * 0.25]
    tiny_with = [c for c in with_repack if c.is_dependent and c.num_tokens < 1024 * 0.25]
    assert len(tiny_without) >= 1
    assert len(tiny_with) == 0  # the tail got merged into a fuller chunk

    # No chunk belongs to two dependency groups at once (tail is capped at one per chunk).
    for c in with_repack:
        if c.is_dependent:
            assert sum(1 for s in c.segments if s.seq_id == lengths.index(1026)) <= 1
    # token accounting is preserved either way
    assert sum(c.num_tokens for c in without) == sum(lengths)
    assert sum(c.num_tokens for c in with_repack) == sum(lengths)


def test_drop_tail_discards_trailing_tokens_and_shrinks_group_size():
    # 1026 tokens, chunk_size 1024 -> 1 full chunk + a 2-token tail that gets dropped.
    chunks = build_chunks([1026], chunk_size=1024, repack_tail=False, drop_tail=True)
    dep = [c for c in chunks if c.is_dependent]

    assert len(dep) == 1  # the tail never became a chunk at all
    assert dep[0].num_tokens == 1024
    assert dep[0].group_size == 1  # shrunk from 2 -> 1 to match reality, not stale
    assert dep[0].group_idx == 0

    # lossy: total output tokens is strictly less than the input length.
    assert sum(c.num_tokens for c in chunks) == 1024 < 1026


def test_drop_tail_is_noop_when_length_divides_evenly():
    # 2048 tokens, chunk_size 1024 -> exactly 2 full chunks, no tail to drop.
    with_drop = build_chunks([2048], chunk_size=1024, repack_tail=False, drop_tail=True)
    without_drop = build_chunks([2048], chunk_size=1024, repack_tail=False, drop_tail=False)
    assert sum(c.num_tokens for c in with_drop) == sum(c.num_tokens for c in without_drop) == 2048
    assert [c.group_size for c in with_drop] == [c.group_size for c in without_drop] == [2, 2]


def test_drop_tail_respects_default_max_tokens_threshold():
    # chunk_size=1024 -> default cap is 256. A 300-token tail is too big to drop.
    chunks = build_chunks([1024 + 300], chunk_size=1024, repack_tail=False, drop_tail=True)
    dep = [c for c in chunks if c.is_dependent]
    assert len(dep) == 2  # tail was NOT dropped, falls through to its own tiny chunk
    assert sum(c.num_tokens for c in dep) == 1024 + 300  # nothing lost
    assert [c.group_size for c in dep] == [2, 2]  # unchanged, since nothing was dropped


def test_drop_tail_custom_threshold_allows_a_bigger_tail():
    # Same 300-token tail, but with an explicit higher threshold it does get dropped.
    chunks = build_chunks([1024 + 300], chunk_size=1024, repack_tail=False, drop_tail=True, drop_tail_max_tokens=500)
    dep = [c for c in chunks if c.is_dependent]
    assert len(dep) == 1
    assert dep[0].num_tokens == 1024
    assert dep[0].group_size == 1


def test_drop_tail_and_repack_tail_are_mutually_exclusive():
    with pytest.raises(ValueError):
        build_chunks([1026], chunk_size=1024, repack_tail=True, drop_tail=True)


def test_drop_tail_schedule_still_validates_clean():
    chunks = build_chunks([1026] + [500] * 2, chunk_size=1024, repack_tail=False, drop_tail=True)
    ops = build_schedule(chunks, keep_last_k=2)
    assert validate_schedule(chunks, ops) == []


def test_group_of_sorts_by_group_idx_ascending():
    chunks = build_chunks([3000], chunk_size=1024, repack_tail=False)
    groups = group_of(chunks)
    assert len(groups) == 1
    (members,) = groups.values()
    assert [c.group_idx for c in members] == sorted(c.group_idx for c in members)


# ---------------------------------------------------------------------------
# 2. Op-sequence generation (Algorithm 2, paper-prose semantics)
# ---------------------------------------------------------------------------


def _make_group(n: int, group: int = 0) -> list[Chunk]:
    return [
        Chunk(chunk_id=i, segments=[Segment(group, i * 100, 100)], group=group, group_idx=i, group_size=n)
        for i in range(n)
    ]


def test_small_group_keeps_every_chunks_activation():
    group = _make_group(3)
    ops = build_group_schedule(group, keep_last_k=8)  # K >= N
    assert ops == [("F", 0), ("F", 1), ("F", 2), ("B", 2), ("B", 1), ("B", 0)]
    assert validate_schedule(group, ops) == []


def test_large_group_recomputes_earlier_chunks_descending():
    group = _make_group(5)
    ops = build_group_schedule(group, keep_last_k=2)

    fwd = [(k, c) for k, c in ops if k in ("F", "Fp", "Fr")]
    bwd_order = [c for k, c in ops if k == "B"]

    # last 2 (idx 3,4) keep activation from the first pass; first 3 get Fp then Fr later.
    first_pass = {cid: kind for kind, cid in fwd[:5]}
    assert first_pass == {0: "Fp", 1: "Fp", 2: "Fp", 3: "F", 4: "F"}
    # backward must be strictly descending overall (this is the crux the validator checks).
    assert bwd_order == [4, 3, 2, 1, 0]
    assert validate_schedule(group, ops) == []


@pytest.mark.parametrize("n,k", [(1, 1), (4, 1), (4, 4), (7, 3), (10, 1)])
def test_group_schedule_always_validates_clean(n, k):
    group = _make_group(n)
    ops = build_group_schedule(group, keep_last_k=k)
    assert validate_schedule(group, ops) == []
    # every chunk forwards exactly once at the graph-building level it needs, and backwards exactly once
    assert sum(1 for kind, _ in ops if kind == "B") == n


def test_build_schedule_combines_standalone_and_groups():
    chunks = build_chunks([3000] + [200] * 4, chunk_size=1024, repack_tail=True)
    ops = build_schedule(chunks, keep_last_k=2)
    assert validate_schedule(chunks, ops) == []
    # every chunk appears in the op sequence
    seen = {cid for _, cid in ops}
    assert seen == {c.chunk_id for c in chunks}


# ---------------------------------------------------------------------------
# 3. Validators — including the deliberately-broken schedule
# ---------------------------------------------------------------------------


def test_validate_forward_order_catches_out_of_order_forward():
    group = _make_group(3)
    broken = [("F", 1), ("F", 0), ("F", 2), ("B", 2), ("B", 1), ("B", 0)]  # chunk 1 forwarded before chunk 0
    problems = validate_forward_order(group, broken)
    assert problems and "forward order violated" in problems[0]


def test_validate_backward_order_catches_ascending_drop_order_bug():
    """Reproduces chunkflow lesson 3 / demo1 scenario C: reading Algorithm 2's
    pseudocode literally (``if idx >= K: keep``, third loop ascending)
    backward-orders the dropped chunks ascending instead of descending —
    which is not just a different valid ordering, it silently drops
    gradient contributions. Build that broken sequence by hand (the
    scheduler itself never generates it) and confirm the validator flags
    it.
    """
    n, k = 5, 2
    group = _make_group(n)
    # pseudocode reading: keep = idx>=K (last N-K), drop = idx<K (first K), processed ascending.
    keep_ids = {c.chunk_id for c in group if c.group_idx >= k}
    drop_ascending = [c for c in group if c.group_idx < k]

    ops = [(("F" if c.chunk_id in keep_ids else "Fp"), c.chunk_id) for c in group]
    ops.extend(("B", c.chunk_id) for c in reversed(group) if c.chunk_id in keep_ids)
    for c in drop_ascending:
        ops.append(("Fr", c.chunk_id))
        ops.append(("B", c.chunk_id))

    problems = validate_backward_order(group, ops)
    assert problems, "the ascending-drop-order schedule must be flagged as incomplete-gradient"
    assert "incomplete gradient" in problems[0]

    # and our own scheduler's output for the same (n, k) must NOT trigger it
    assert validate_backward_order(group, build_group_schedule(group, k)) == []


def test_validate_completeness_catches_missing_and_double_backward():
    group = _make_group(2)
    missing = [("F", 0), ("F", 1), ("B", 1)]  # chunk 0 never backwarded
    assert any("expected exactly 1" in p for p in validate_completeness(group, missing))

    double = [("F", 0), ("F", 1), ("B", 1), ("B", 1), ("B", 0)]  # chunk 1 backwarded twice
    assert any("2 time(s)" in p for p in validate_completeness(group, double))

    fp_only = [("Fp", 0), ("F", 1), ("B", 0), ("B", 1)]  # chunk 0 never got a graph-building forward
    assert any("without a prior graph-building forward" in p for p in validate_completeness(group, fp_only))
