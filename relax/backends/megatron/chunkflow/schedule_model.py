# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Relative costs: F=rows/128, B=2F, Fr=F; zero network latency."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from relax.backends.megatron.chunkflow.scheduler import Chunk
from relax.backends.megatron.chunkflow.units import Unit, classify_units, make_unit_bwd_order


if TYPE_CHECKING:
    from relax.backends.megatron.chunkflow.telemetry import PlanTelemetry


def pipeline_plan(
    units: Sequence[Unit], chunks: Sequence[Chunk], keep_last_k: int
) -> tuple[list[int], list[int], set[int], int]:
    standalone = [u.unit_id for u in units if not u.is_dependent]
    dependent = sorted((u for u in units if u.is_dependent), key=lambda u: u.round)
    fwd_seq = standalone + [u.unit_id for u in dependent]
    bwd_seq = make_unit_bwd_order(units, fwd_seq)
    _, need_fr = classify_units(units, chunks, keep_last_k)
    by_id = {u.unit_id: u for u in units}
    positions = {u: i for i, u in enumerate(fwd_seq)}
    done_at: dict[int, int] = {}
    for unit in units:
        for group in unit.groups:
            done_at[group] = max(done_at.get(group, -1), positions[unit.unit_id])
    delay = max(
        (max((done_at[g] for g in by_id[u].groups), default=positions[u]) - i for i, u in enumerate(bwd_seq)),
        default=0,
    )
    return fwd_seq, bwd_seq, need_fr, delay


@dataclass(frozen=True)
class ModelPlan:
    rows: dict[int, int]
    forward: tuple[int, ...]
    backward: tuple[int, ...]
    recompute: frozenset[int]
    warmups: tuple[int, ...]


def make_model_plan(units: Sequence[Unit], chunks: Sequence[Chunk], keep_last_k: int, pp_size: int) -> ModelPlan:
    fwd, bwd, recompute, delay = pipeline_plan(units, chunks, keep_last_k)
    return ModelPlan(
        {u.unit_id: ((u.num_tokens + 127) // 128) * 128 for u in units},
        tuple(fwd),
        tuple(bwd),
        frozenset(recompute),
        tuple(min(delay + pp_size - s - 1, len(fwd)) for s in range(pp_size)),
    )


def merge_model_plans(plans: Sequence[ModelPlan], pp_size: int) -> ModelPlan:
    """Join parent orders without interleaving their dependency groups."""
    rows, forward, backward, recompute = {}, [], [], set()
    delay = 0
    for plan in plans:
        ids = {uid: len(rows) + i for i, uid in enumerate(plan.rows)}
        rows.update((ids[uid], size) for uid, size in plan.rows.items())
        forward.extend(ids[uid] for uid in plan.forward)
        backward.extend(ids[uid] for uid in plan.backward)
        recompute.update(ids[uid] for uid in plan.recompute)
        # Equal F/B offsets preserve each parent's dependency delay.
        delay = max(delay, plan.warmups[-1])
    return ModelPlan(
        rows,
        tuple(forward),
        tuple(backward),
        frozenset(recompute),
        tuple(min(delay + pp_size - s - 1, len(forward)) for s in range(pp_size)),
    )


def simulate(plans: Sequence[ModelPlan], pp_size: int) -> dict[str, Any]:
    if pp_size < 1:
        raise ValueError("pp_size must be positive")
    for plan in plans:
        if len(plan.warmups) != pp_size or set(plan.forward) != set(plan.backward):
            raise ValueError("invalid stage count or forward/backward unit orders")
        if set(plan.rows) != set(plan.forward) or len(plan.forward) != len(plan.rows):
            raise ValueError("each unit must be forwarded exactly once")
        if len(plan.backward) != len(plan.rows) or not plan.recompute <= set(plan.rows):
            raise ValueError("invalid backward/recompute units")
        if any(w < 0 or w > len(plan.forward) for w in plan.warmups):
            raise ValueError("invalid warmup depth")
        if any(row <= 0 for row in plan.rows.values()):
            raise ValueError("unit rows must be positive")
    nodes: list[dict[str, Any]] = []
    previous = [-1] * pp_size
    messages: dict[tuple, dict[str, int]] = defaultdict(dict)
    counters: dict[tuple, int] = defaultdict(int)

    def add(stage: int, parent: int, kind: str, unit: int, duration: float) -> int:
        index = len(nodes)
        pred = previous[stage]
        nodes.append(
            {
                "stage": stage,
                "parent": parent,
                "kind": kind,
                "unit": unit,
                "duration": duration,
                "posted_after": pred,
                "deps": {pred} if pred >= 0 else set(),
            }
        )
        previous[stage] = index
        return index

    def communicate(stage: int, parent: int, actions: Sequence[str]) -> None:
        if not actions:
            return
        for phase in ("shape", "data"):
            index = add(stage, parent, phase, -1, 0.0)
            for action in actions:
                if action == "sf":
                    channel, side = (stage, stage + 1, "fwd"), "send"
                elif action == "rf":
                    channel, side = (stage - 1, stage, "fwd"), "recv"
                elif action == "sb":
                    channel, side = (stage, stage - 1, "bwd"), "send"
                else:
                    channel, side = (stage + 1, stage, "bwd"), "recv"
                key = (channel, phase, counters[channel, phase, side])
                counters[channel, phase, side] += 1
                messages[key][side] = index

    for stage in range(pp_size):
        first, last = stage == 0, stage == pp_size - 1
        for parent, plan in enumerate(plans):
            warmup, n = plan.warmups[stage], len(plan.forward)

            def forward(unit: int) -> None:
                add(stage, parent, "Fp" if unit in plan.recompute else "F", unit, plan.rows[unit] / 128)

            def backward(unit: int) -> None:
                if unit in plan.recompute:
                    add(stage, parent, "Fr", unit, plan.rows[unit] / 128)
                add(stage, parent, "B", unit, 2 * plan.rows[unit] / 128)

            for i in range(warmup):
                communicate(stage, parent, [] if first else ["rf"])
                forward(plan.forward[i])
                communicate(stage, parent, [] if last else ["sf"])
            steady = n - warmup
            if steady:
                communicate(stage, parent, [] if first else ["rf"])
            for i in range(steady):
                forward(plan.forward[warmup + i])
                communicate(stage, parent, [] if last else ["sf", "rb"])
                backward(plan.backward[i])
                communicate(stage, parent, [] if first else (["sb", "rf"] if i + 1 < steady else ["sb"]))
            for i in range(steady, n):
                communicate(stage, parent, [] if last else ["rb"])
                backward(plan.backward[i])
                communicate(stage, parent, [] if first else ["sb"])

    for key, peers in messages.items():
        if set(peers) != {"send", "recv"}:
            raise ValueError(f"unmatched pipeline transfer: {key}")
        for side, other in (("send", "recv"), ("recv", "send")):
            posted_after = nodes[peers[other]]["posted_after"]
            if posted_after >= 0:
                nodes[peers[side]]["deps"].add(posted_after)

    outgoing: dict[int, list[int]] = defaultdict(list)
    degree = []
    for i, node in enumerate(nodes):
        degree.append(len(node["deps"]))
        for pred in node["deps"]:
            outgoing[pred].append(i)
    ready = [i for i, count in enumerate(degree) if count == 0]
    heapq.heapify(ready)
    ends = [0.0] * len(nodes)
    timeline = []
    visited = 0
    while ready:
        i = heapq.heappop(ready)
        node = nodes[i]
        start = max((ends[pred] for pred in node["deps"]), default=0.0)
        ends[i] = start + node["duration"]
        visited += 1
        if node["duration"]:
            timeline.append(
                {k: node[k] for k in ("stage", "parent", "kind", "unit")} | {"start": start, "end": ends[i]}
            )
        for child in outgoing[i]:
            degree[child] -= 1
            if degree[child] == 0:
                heapq.heappush(ready, child)
    if visited != len(nodes):
        raise ValueError("blocking pipeline model deadlocked")
    makespan = max(ends, default=0.0)
    busy = sum(node["duration"] for node in nodes)
    return {
        "makespan": makespan,
        "bubble_fraction": 1 - busy / (pp_size * makespan) if makespan else 0.0,
        "timeline": timeline,
    }


def predict_step(plans: Sequence[PlanTelemetry], pp_size: int) -> dict[str, Any]:
    if not plans or any(plan.model_plan is None for plan in plans):
        raise ValueError("schedule prediction needs nonempty plans captured with include_schedule=True")
    baseline_rows = {i: plan.input_rows for i, plan in enumerate(plans)}
    order = tuple(baseline_rows)
    baseline = simulate(
        [
            ModelPlan(
                baseline_rows,
                order,
                order,
                frozenset(),
                tuple(min(pp_size - s - 1, len(order)) for s in range(pp_size)),
            )
        ],
        pp_size,
    )
    window = plans[0].parent_window
    chunkflow = simulate(
        [
            merge_model_plans([p.model_plan for p in plans[i : i + window]], pp_size)
            for i in range(0, len(plans), window)
        ],
        pp_size,
    )
    return {
        "baseline": baseline,
        "chunkflow": chunkflow,
        "speedup": baseline["makespan"] / chunkflow["makespan"],
    }
