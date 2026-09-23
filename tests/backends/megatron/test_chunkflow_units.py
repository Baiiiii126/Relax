# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for chunkflow's execution-unit assembly and unit-level schedule.

Pure Python — units only group chunks and order ops, no tensors involved.

The load-bearing property tested here is the *reduction*: when every unit holds
exactly one chunk (the ``staggered`` policy), the unit-level schedule must
produce the same op sequence as ``scheduler.gen_state_aware_1f1b_ops``, which
is itself verified against an independent simulator. That ties the new code to
the already-verified one instead of asking for trust twice.
"""

from __future__ import annotations

import pytest


try:
    from relax.backends.megatron.chunkflow.scheduler import (
        STRATEGIES,
        build_chunks,
        classify_chunks,
        gen_state_aware_1f1b_ops,
        make_bwd_order,
    )
    from relax.backends.megatron.chunkflow.units import (
        POLICIES,
        Unit,
        build_micro_batches,
        classify_units,
        completed_groups,
        expand_units,
        gen_unit_ops,
        make_unit_bwd_order,
        validate_unit_backward_order,
        validate_units,
    )
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


def _two_groups():
    """Two long sequences (dependency groups) plus short standalone
    sequences."""
    return build_chunks([1024 * 4 + 5, 1024 * 3 + 9, 300, 400], chunk_size=1024, repack_tail=False)


def _one_group():
    return build_chunks([1024 * 5 + 3, 250], chunk_size=1024, repack_tail=False)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_units_tile_the_chunks_exactly(policy):
    chunks = _two_groups()
    units = build_micro_batches(chunks, policy)
    assert validate_units(units, chunks) == []
    assert sorted(c.chunk_id for c in expand_units(units)) == sorted(c.chunk_id for c in chunks)


def test_staggered_is_one_chunk_per_unit_and_preserves_chunk_order():
    chunks = _one_group()
    units = build_micro_batches(chunks, "staggered")
    assert [len(u.chunks) for u in units] == [1] * len(chunks)
    assert expand_units(units) == list(chunks)


def test_lockstep_merges_same_depth_chunks_across_groups():
    chunks = _two_groups()
    units = build_micro_batches(chunks, "lockstep")

    # Rounds that both groups reach must hold one chunk from each.
    shared_rounds = [u for u in units if u.is_dependent and len(u.groups) > 1]
    assert shared_rounds, "lockstep should merge same-depth chunks from different groups"
    for u in shared_rounds:
        assert len({c.group for c in u.chunks}) == len(u.chunks)

    assert validate_units(units, chunks) == []


def test_group_is_completed_only_by_its_lowest_chunk_backward():
    chunks = _one_group()
    units = build_micro_batches(chunks, "staggered")
    dependent = [u for u in units if u.is_dependent]

    assert all(not completed_groups(unit) for unit in dependent[1:])
    assert completed_groups(dependent[0]) == dependent[0].groups


def test_lockstep_puts_standalone_units_first():
    chunks = _two_groups()
    units = build_micro_batches(chunks, "lockstep")
    standalone = [u for u in units if not u.is_dependent]
    assert standalone, "short sequences exist in this batch"
    assert units[: len(standalone)] == standalone


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        build_micro_batches(_one_group(), "nope")


def test_a_unit_may_not_hold_two_chunks_of_one_group():
    chunks = _one_group()
    members = [c for c in chunks if c.is_dependent][:2]
    with pytest.raises(ValueError, match="causally ordered"):
        Unit(unit_id=0, chunks=members)


# ---------------------------------------------------------------------------
# Reduction to the verified chunk-level schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("pp", [1, 2, 4])
@pytest.mark.parametrize("keep_last_k", [1, 3])
def test_staggered_units_reduce_exactly_to_the_chunk_level_schedule(strategy, pp, keep_last_k):
    """The whole reason the unit layer is trustworthy: with one chunk per unit
    it must be the already-verified schedule, op for op."""
    for factory in (_one_group, _two_groups):
        chunks = factory()
        units = build_micro_batches(chunks, "staggered")
        for stage in range(pp):
            assert gen_unit_ops(units, chunks, keep_last_k, pp, stage, strategy) == gen_state_aware_1f1b_ops(
                chunks, keep_last_k, pp, stage, strategy
            )


def test_staggered_unit_bwd_order_is_chunk_bwd_order():
    chunks = _two_groups()
    units = build_micro_batches(chunks, "staggered")
    fwd = [u.unit_id for u in units]
    # unit_id == chunk position under staggered, so the orders are comparable.
    assert make_unit_bwd_order(units, fwd) == make_bwd_order(chunks, [c.chunk_id for c in chunks])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_units_drops_only_units_whose_members_all_may_drop():
    chunks = _two_groups()
    _, chunk_need_fr = classify_chunks(chunks, keep_last_k=2)

    for policy in POLICIES:
        units = build_micro_batches(chunks, policy)
        keep, need_fr = classify_units(units, chunks, keep_last_k=2)
        assert keep | need_fr == {u.unit_id for u in units}
        assert not (keep & need_fr)
        for u in units:
            if u.unit_id in need_fr:
                assert set(u.chunk_ids) <= chunk_need_fr


def test_lockstep_merging_can_cost_part_of_the_recompute_window():
    """Documented trade: a unit keeps if any member must keep, so unequal group
    sizes leave fewer units droppable than a per-group window would."""
    chunks = build_chunks([1024 * 5 + 1, 1024 * 2 + 1], chunk_size=1024, repack_tail=False)
    staggered = build_micro_batches(chunks, "staggered")
    lockstep = build_micro_batches(chunks, "lockstep")

    _, fr_staggered = classify_units(staggered, chunks, keep_last_k=1)
    _, fr_lockstep = classify_units(lockstep, chunks, keep_last_k=1)
    assert len(fr_lockstep) <= len(fr_staggered)


# ---------------------------------------------------------------------------
# Ordering correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_generated_unit_schedules_satisfy_the_backward_invariant(policy, strategy):
    chunks = _two_groups()
    units = build_micro_batches(chunks, policy)
    for stage in range(4):
        ops = gen_unit_ops(units, chunks, keep_last_k=2, pp=4, stage=stage, strategy=strategy)
        assert validate_unit_backward_order(units, ops) == [], f"{policy}/{strategy} stage={stage}"


def test_unit_backward_order_defers_dependent_units_past_their_successors():
    chunks = _two_groups()
    units = build_micro_batches(chunks, "lockstep")
    ops = gen_unit_ops(units, chunks, keep_last_k=8, pp=1, stage=0)  # K>=N: nothing recomputed

    done: set[int] = set()
    by_id = {u.unit_id: u for u in units}
    for kind, uid in ops:
        if kind != "B":
            continue
        u = by_id[uid]
        if u.is_dependent:
            for other in units:
                if other.groups & u.groups and other.round > u.round:
                    assert other.unit_id in done, f"unit {uid} backwarded before later-round {other.unit_id}"
        done.add(uid)


def test_every_unit_is_backwarded_exactly_once():
    chunks = _two_groups()
    units = build_micro_batches(chunks, "lockstep")
    ops = gen_unit_ops(units, chunks, keep_last_k=2, pp=1, stage=0)
    counts = {}
    for kind, uid in ops:
        if kind == "B":
            counts[uid] = counts.get(uid, 0) + 1
    assert counts == {u.unit_id: 1 for u in units}
