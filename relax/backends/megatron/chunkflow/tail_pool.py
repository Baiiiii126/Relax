# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence

from relax.backends.megatron.chunkflow.schedule_model import ModelPlan
from relax.backends.megatron.chunkflow.units import Unit, make_unit_bwd_order


def _ordered_plan(units: Sequence[Unit], pp_size: int) -> ModelPlan:
    forward = tuple(u.unit_id for u in units)
    backward = tuple(make_unit_bwd_order(units, forward))
    done = {g: i for i, u in enumerate(units) for g in u.groups}
    by_id = {u.unit_id: u for u in units}
    positions = {uid: i for i, uid in enumerate(forward)}
    delay = max(
        (max((done[g] for g in by_id[uid].groups), default=positions[uid]) - i for i, uid in enumerate(backward)),
        default=0,
    )
    return ModelPlan(
        {u.unit_id: (u.num_tokens + 127) // 128 * 128 for u in units},
        forward,
        backward,
        frozenset(),
        tuple(min(delay + pp_size - s - 1, len(units)) for s in range(pp_size)),
    )


def peak_live_rows(plan: ModelPlan) -> tuple[int, ...]:
    peaks = []
    for warmup in plan.warmups:
        live = sum(plan.rows[uid] for uid in plan.forward[:warmup])
        peak = live
        for i, uid in enumerate(plan.forward[warmup:]):
            live += plan.rows[uid]
            peak = max(peak, live)
            live -= plan.rows[plan.backward[i]]
        peaks.append(peak)
    return tuple(peaks)


def pool_terminal_units(
    units: Mapping[int, Unit], parents: Mapping[int, int], plan: ModelPlan, capacity: int
) -> tuple[dict[int, Unit], dict[int, tuple[int, ...]], ModelPlan]:
    """Merge nearby terminal tails without increasing warmup or live activation
    rows."""
    if plan.recompute:
        raise ValueError("tail pooling requires keep_last_k=0")
    pooled = dict(units)
    members = {uid: (uid,) for uid in units}
    order = list(plan.forward)
    limit = peak_live_rows(plan)
    candidates = {
        uid
        for uid, unit in units.items()
        if len(unit.chunks) == 1
        and len(unit.chunks[0].segments) == 1
        and unit.chunks[0].is_dependent
        and unit.chunks[0].group_idx > 0
        and unit.chunks[0].group_idx == unit.chunks[0].group_size - 1
        and unit.num_tokens < capacity
    }
    for right in plan.forward:
        if right not in candidates:
            continue
        for left in sorted(order[: order.index(right)], key=lambda uid: -pooled[uid].num_tokens):
            if left not in candidates or pooled[left].num_tokens + pooled[right].num_tokens > capacity:
                continue
            ids = members[left] + members[right]
            owner_ids = [parents[uid] for uid in ids]
            if max(owner_ids) - min(owner_ids) > 1 or pooled[left].groups & pooled[right].groups:
                continue
            unit = Unit(right, pooled[left].chunks + pooled[right].chunks)
            trial_order = [uid for uid in order if uid != left]
            trial = _ordered_plan([unit if uid == right else pooled[uid] for uid in trial_order], len(plan.warmups))
            if any(a > b for a, b in zip(trial.warmups, plan.warmups)) or any(
                a > b for a, b in zip(peak_live_rows(trial), limit)
            ):
                continue
            pooled[right], members[right] = unit, ids
            del pooled[left], members[left]
            candidates.remove(left)
            order, plan = trial_order, trial
    return pooled, members, plan
