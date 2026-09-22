# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The chunkflow pipeline schedule: is it deadlock-free, and is it valid?

Pure Python, no tensors. The pipeline executor drives a hand-built p2p
sequence, and getting it wrong hangs 8 GPUs minutes into warmup -- so both
properties are checked here instead.

The executor follows Megatron's standard non-interleaved 1F1B call pattern
(bare warmup, fused steady, bare cooldown), which is deadlock-free because the
backward a steady step receives lags the forward it sends by the warmup depth.
On top of that the schedule has to keep chunkflow's causal order: a group's
backward only runs once the whole group has forwarded (the store is only full
then), highest-chunk-first. ``_pipeline_schedule`` supplies the forward order,
the backward order, and a warmup of ``C + (pp - stage - 1)`` where ``C`` covers
the first group's completion.

The simulator is faithful to ``_communicate``: every call runs a shape exchange
as one phase and the data exchange as the next, everything in a phase is posted
together, and matching is a FIFO rendezvous per (stage-pair, direction). It is
the same model that reproduced the original deadlock in
``fuse-send-forward-with-next-backward``.
"""

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
    return (stage, stage + 1, "bwd")  # rb


def _simulate(calls_by_stage):
    """Run every stage's calls under blocking p2p; raise if it would hang."""
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
    """The standard 1F1B call sequence for warmup depth W over n units."""
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


def _cases():
    lengths = (
        [2048, 700, 300, 1024, 1500, 512, 900, 2600],
        [200, 180, 150, 220, 90],
        [300, 2048, 250, 900, 120],
        [96, 96, 96, 96],
        [1234, 56, 789, 2100, 333, 88],
        [2000, 1800, 2100, 1900],
        [2600],
        [300, 250, 96],
    )
    for lens in lengths:
        for pp in (2, 3, 4):
            for policy in ("staggered", "lockstep"):
                for k in (1, 2, 100):
                    yield lens, pp, policy, k


def _schedule(lengths, pp, policy, k, stage):
    chunks = build_chunks(lengths, chunk_size=512, repack_tail=False)
    units = build_micro_batches(chunks, policy)
    return chunks, units, _pipeline_schedule(units, chunks, k, pp, stage)


@pytest.mark.parametrize("lengths,pp,policy,k", list(_cases()))
def test_pipeline_plan_never_deadlocks(lengths, pp, policy, k):
    chunks, units, (fwd_seq, bwd_seq, need_fr, W0) = _schedule(lengths, pp, policy, k, 0)
    n = len(fwd_seq)
    if n < pp:
        pytest.skip("fewer units than stages")
    calls = []
    for stage in range(pp):
        _, _, _, W = _schedule(lengths, pp, policy, k, stage)[2]
        calls.append(_calls(min(W, n), n, stage == 0, stage == pp - 1))
    _simulate(calls)


@pytest.mark.parametrize("lengths,pp,policy,k", list(_cases()))
def test_backward_order_is_dependency_valid(lengths, pp, policy, k):
    for stage in range(pp):
        chunks, units, (fwd_seq, bwd_seq, need_fr, W) = _schedule(lengths, pp, policy, k, stage)
        n = len(fwd_seq)
        W = min(W, n)
        fwd_pos = {u: i for i, u in enumerate(fwd_seq)}
        by_id = {u.unit_id: u for u in units}
        # Every backward runs after its own forward and after its group completes.
        for i, u in enumerate(bwd_seq):
            n_done = (W + i + 1) if i < n - W else n
            assert u in set(fwd_seq[:n_done]), f"backward {u} before its forward"
            for c in by_id[u].chunks:
                if c.group is not None:
                    members = [x.unit_id for x in units if any(cc.group == c.group for cc in x.chunks)]
                    assert all(m in set(fwd_seq[:n_done]) for m in members), (
                        f"backward {u} before group {c.group} full"
                    )
        # Within a group, forwards ascend and backwards descend. A chunk maps to its
        # unit -- under lockstep several chunks share one, so go through the unit.
        unit_of = {}
        group_idx_of = {}
        for u in units:
            for c in u.chunks:
                unit_of[c.chunk_id] = u.unit_id
                group_idx_of[c.chunk_id] = c.group_idx
        groups = defaultdict(list)
        for c in chunks:
            if c.group is not None:
                groups[c.group].append(c.chunk_id)
        bwd_pos = {u: i for i, u in enumerate(bwd_seq)}
        for g, mem in groups.items():
            fp = sorted(fwd_pos[unit_of[m]] for m in mem)
            assert fp == sorted(fp), f"group {g} forwards not ascending"
            highest_first = sorted(mem, key=lambda m: -group_idx_of[m])
            bp = [bwd_pos[unit_of[m]] for m in highest_first]
            assert bp == sorted(bp), f"group {g} backwards not descending"


@pytest.mark.parametrize("lengths,pp,policy,k", list(_cases()))
def test_every_unit_forwarded_and_backwarded_once(lengths, pp, policy, k):
    for stage in range(pp):
        _, units, (fwd_seq, bwd_seq, need_fr, _) = _schedule(lengths, pp, policy, k, stage)
        assert sorted(fwd_seq) == sorted(bwd_seq) == sorted(u.unit_id for u in units)
        assert set(need_fr) <= set(bwd_seq)
