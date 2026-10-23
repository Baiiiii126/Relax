# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import replace

import pytest

from relax.backends.megatron.chunkflow.schedule_model import make_model_plan, merge_model_plans, simulate
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.tail_pool import peak_live_rows, pool_terminal_units
from relax.backends.megatron.chunkflow.telemetry import StepPlanTelemetry, summarize_plan
from relax.backends.megatron.chunkflow.units import Unit


@pytest.mark.parametrize("lengths", [[[10, 5, 14]], [[10, 5], [14, 5]], [[14], [14]], [[10, 5], [2], [14, 5]]])
def test_tail_pool_preserves_dependencies_and_telemetry(lengths):
    units, owners, plans = {}, {}, []
    telemetry = StepPlanTelemetry()
    for parent, values in enumerate(lengths):
        chunks = build_chunks(values, 4, repack_tail=False)
        local = [Unit(i, [c]) for i, c in enumerate(chunks)]
        model = make_model_plan(local, chunks, 1 << 30, 4)
        plans.append(model)
        telemetry.add(
            replace(
                summarize_plan(values, chunks, local, 1 << 30, 4, model.warmups, include_schedule=True),
                parent_window=len(lengths),
            )
        )
        for unit in local:
            uid = len(units)
            units[uid] = Unit(
                uid,
                [
                    replace(c, group=parent * 100 + c.group if c.group is not None else None, chunk_id=uid)
                    for c in unit.chunks
                ],
            )
            owners[uid] = parent
    original = merge_model_plans(plans, 4)
    pooled, members, plan = pool_terminal_units(units, owners, original, 4)
    assert sorted(uid for ids in members.values() for uid in ids) == sorted(units)
    assert set(plan.forward) == set(plan.backward) == set(pooled)
    assert all(a <= b for a, b in zip(plan.warmups, original.warmups))
    assert all(a <= b for a, b in zip(peak_live_rows(plan), peak_live_rows(original)))
    assert sum(u.num_tokens for u in pooled.values()) == sum(map(sum, lengths))
    fpos, bpos = ({uid: i for i, uid in enumerate(order)} for order in (plan.forward, plan.backward))
    groups = {}
    for uid, unit in pooled.items():
        assert unit.num_tokens <= 4
        if len(members[uid]) > 1:
            assert max(owners[m] for m in members[uid]) - min(owners[m] for m in members[uid]) <= 1
            assert all(c.group_idx == c.group_size - 1 for c in unit.chunks)
        for c in unit.chunks:
            if c.is_dependent:
                groups.setdefault(c.group, []).append((c.group_idx, uid))
    for group in groups.values():
        ordered = [uid for _, uid in sorted(group)]
        assert [fpos[u] for u in ordered] == sorted(fpos[u] for u in ordered)
        assert [bpos[u] for u in ordered] == sorted((bpos[u] for u in ordered), reverse=True)
        for s, w in enumerate(plan.warmups):
            for uid in ordered:
                assert max(fpos[u] for u in ordered) <= min(w + bpos[uid], len(plan.forward) - 1)
    assert simulate([plan], 4)["makespan"] > 0
    telemetry.windows.append(plan)
    telemetry.pooled_units = sum(len(ids) > 1 for ids in members.values())
    metrics = telemetry.to_metrics(4, simulate_schedule=True)
    assert metrics["num_microbatches"] == len(lengths)
    assert metrics["num_units"] == len(pooled)
    assert metrics["tail_pool_saved_units"] == len(units) - len(pooled)
    assert metrics["schedule_model_chunkflow_makespan"] == simulate([plan], 4)["makespan"]
