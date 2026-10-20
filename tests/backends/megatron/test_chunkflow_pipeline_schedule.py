# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Blocking P2P simulation and dependency checks for the PP schedule."""

from __future__ import annotations

from collections import defaultdict

import pytest


try:
    from relax.backends.megatron.chunkflow.pipeline_schedules import _pipeline_schedule
    from relax.backends.megatron.chunkflow.scheduler import build_chunks
    from relax.backends.megatron.chunkflow.units import build_micro_batches
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


def _channel(stage, action):
    if action == "sf":
        return (stage, stage + 1, "fwd")
    if action == "rf":
        return (stage - 1, stage, "fwd")
    if action == "sb":
        return (stage - 1, stage, "bwd")
    return (stage, stage + 1, "bwd")


def _simulate(calls_by_stage):
    pp = len(calls_by_stage)
    pos = defaultdict(lambda: {"send": 0, "recv": 0})
    per_call = []
    for stage, calls in enumerate(calls_by_stage):
        stage_calls = []
        for recvs, sends in calls:
            phases = []
            for _part in ("shape", "data"):
                msgs = []
                for a in recvs:
                    ch = _channel(stage, a)
                    msgs.append((ch, "recv", pos[ch]["recv"]))
                    pos[ch]["recv"] += 1
                for a in sends:
                    ch = _channel(stage, a)
                    msgs.append((ch, "send", pos[ch]["send"]))
                    pos[ch]["send"] += 1
                phases.append(msgs)
            stage_calls.append(phases)
        per_call.append(stage_calls)

    sends_posted = defaultdict(int)
    recvs_posted = defaultdict(int)
    ptr = [(0, 0) for _ in range(pp)]
    posted = [None] * pp

    def enter(stage):
        c, p = ptr[stage]
        if c >= len(per_call[stage]):
            posted[stage] = None
            return
        posted[stage] = [list(per_call[stage][c][p]), 0]
        for ch, kind, _p in posted[stage][0]:
            (sends_posted if kind == "send" else recvs_posted)[ch] += 1

    for stage in range(pp):
        enter(stage)

    guard = 0
    while True:
        guard += 1
        assert guard < 10_000_000, "iteration limit"
        progress = False
        for stage in range(pp):
            if posted[stage] is None:
                continue
            msgs, done = posted[stage]
            still = []
            for ch, kind, p in msgs:
                if (recvs_posted if kind == "send" else sends_posted)[ch] > p:
                    done += 1
                else:
                    still.append((ch, kind, p))
            if done != posted[stage][1]:
                posted[stage] = [still, done]
                progress = True
            if not still:
                c, p = ptr[stage]
                ptr[stage] = (c, p + 1) if p + 1 < len(per_call[stage][c]) else (c + 1, 0)
                enter(stage)
                progress = True
        if all(ptr[s][0] >= len(per_call[s]) for s in range(pp)):
            return
        assert progress, f"p2p deadlock: stalled at {[(s, ptr[s]) for s in range(pp) if posted[s]]}"


def _calls(W, n, is_first, is_last):
    calls = []
    for _ in range(W):
        calls.append((["rf"] if not is_first else [], []))
        calls.append(([], ["sf"]) if not is_last else ([], []))
    n_steady = n - W
    if n_steady > 0:
        calls.append((["rf"] if not is_first else [], []))
    for i in range(n_steady):
        calls.append((["rb"] if not is_last else [], ["sf"] if not is_last else []))
        calls.append((["rf"] if (not is_first and i < n_steady - 1) else [], ["sb"] if not is_first else []))
    for _ in range(W):
        calls.append((["rb"] if not is_last else [], []))
        calls.append(([], ["sb"]) if not is_first else ([], []))
    return calls


def _schedule(lengths, pp, policy, k, stage):
    chunks = build_chunks(lengths, chunk_size=512, repack_tail=False)
    units = build_micro_batches(chunks, policy)
    return chunks, units, _pipeline_schedule(units, chunks, k or 1 << 30, pp, stage)


@pytest.mark.parametrize(
    "lengths,pp,policy,k",
    [
        ([96, 96, 96, 96], 4, "staggered", 0),
        ([2600], 4, "staggered", 0),
        ([2600], 4, "staggered", 1),
        ([2048, 700, 300, 1500], 4, "staggered", 0),
        ([2048, 700, 300, 1500], 4, "staggered", 1),
        ([2000, 1800, 2100, 1900], 3, "staggered", 2),
        ([2048, 700, 300, 1500], 2, "lockstep", 1),
    ],
)
def test_pipeline_plan_is_complete_dependency_valid_and_deadlock_free(lengths, pp, policy, k):
    calls = []
    for stage in range(pp):
        chunks, units, (fwd, bwd, need_fr, warmup) = _schedule(lengths, pp, policy, k, stage)
        n = len(fwd)
        assert sorted(fwd) == sorted(bwd) == sorted(u.unit_id for u in units)
        assert set(need_fr) <= set(bwd)
        assert bool(need_fr) == (k > 0)
        assert 0 <= warmup <= n
        calls.append(_calls(warmup, n, stage == 0, stage == pp - 1))
        by_id = {u.unit_id: u for u in units}
        for i, uid in enumerate(bwd):
            done = set(fwd[: min(warmup + i + 1, n)])
            assert uid in done, f"backward {uid} before its forward"
            for group in by_id[uid].groups:
                assert all(u.unit_id in done for u in units if group in u.groups), (
                    f"backward {uid} before group {group} full"
                )
        fwd_pos = {uid: i for i, uid in enumerate(fwd)}
        bwd_pos = {uid: i for i, uid in enumerate(bwd)}
        unit_of = {c.chunk_id: u.unit_id for u in units for c in u.chunks}
        groups = defaultdict(list)
        for chunk in chunks:
            if chunk.is_dependent:
                groups[chunk.group].append(chunk)
        for members in groups.values():
            members.sort(key=lambda chunk: chunk.group_idx)
            forward_positions = [fwd_pos[unit_of[c.chunk_id]] for c in members]
            backward_positions = [bwd_pos[unit_of[c.chunk_id]] for c in reversed(members)]
            assert forward_positions == sorted(forward_positions)
            assert backward_positions == sorted(backward_positions)
    _simulate(calls)
