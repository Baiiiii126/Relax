# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The pipeline executor's p2p plan: is it deadlock-free, and do stages agree?

Pure Python, no tensors. ``pipeline_schedules`` drives the point-to-point
traffic by hand, and the failure mode of getting it wrong is a hang on 8 GPUs
several minutes into warmup -- so the plan is simulated here instead.

The simulation runs ``plan_p2p_calls`` itself, the same function the executor
walks, so what is checked is what will run: every call is atomic (everything in
it is posted together, as ``batch_isend_irecv`` does), matching is FIFO per
(source, destination, direction), and a send blocks until the peer posts the
receive for it just like a receive blocks until the peer posts the send. A plan
where every send stands alone is included as a negative control -- it hangs,
which is the whole reason the executor fuses sends with opposite-direction
receives at each turn.
"""

from __future__ import annotations

from collections import defaultdict, deque

import pytest


try:
    from relax.backends.megatron.chunkflow.pipeline_schedules import plan_p2p_calls
    from relax.backends.megatron.chunkflow.scheduler import build_chunks
    from relax.backends.megatron.chunkflow.units import build_micro_batches, gen_unit_ops
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


def _plans(lengths, chunk_size, policy, keep_last_k, pp):
    chunks = build_chunks(lengths, chunk_size=chunk_size, repack_tail=False)
    units = build_micro_batches(chunks, policy)
    return [
        plan_p2p_calls(
            gen_unit_ops(units, chunks, keep_last_k, pp=pp, stage=stage),
            is_first=stage == 0,
            is_last=stage == pp - 1,
        )
        for stage in range(pp)
    ]


def _ops(lengths, chunk_size, policy, keep_last_k, pp, stage):
    chunks = build_chunks(lengths, chunk_size=chunk_size, repack_tail=False)
    units = build_micro_batches(chunks, policy)
    return gen_unit_ops(units, chunks, keep_last_k, pp=pp, stage=stage)


def _simulate(plans, pp) -> None:
    """Run every stage's plan under blocking p2p; raise if it would hang.

    The model is a rendezvous, not a buffer: posting a call puts all of its
    operations to the neighbour at once (``batch_isend_irecv``), and the call
    finishes only once every send has found the peer's matching receive and
    every receive has found the peer's matching send. Matching is FIFO per
    (source, destination, direction), which is what the communicator gives us.
    A send therefore blocks exactly like a receive does -- that is the whole
    reason the fused calls exist, so the model has to reproduce it.
    """
    queues: dict[tuple[int, int, str], dict[str, deque]] = defaultdict(lambda: {"send": deque(), "recv": deque()})
    call_idx = [0] * pp
    done: list[list[bool]] = [[] for _ in range(pp)]

    def channel(stage: int, action: str) -> tuple[int, int, str]:
        if action == "send_fwd":
            return (stage, stage + 1, "fwd")
        if action == "recv_fwd":
            return (stage - 1, stage, "fwd")
        if action == "send_bwd":
            return (stage, stage - 1, "bwd")
        return (stage + 1, stage, "bwd")

    def post(stage: int) -> None:
        before, after = plans[stage][call_idx[stage]]
        actions = list(before) + list(after)
        done[stage] = [False] * len(actions)
        for index, action in enumerate(actions):
            queues[channel(stage, action)]["send" if action.startswith("send") else "recv"].append((stage, index))

    for stage in range(pp):
        post(stage)

    while True:
        progressed = False
        for pair in queues.values():
            while pair["send"] and pair["recv"]:
                sender, sender_slot = pair["send"].popleft()
                receiver, receiver_slot = pair["recv"].popleft()
                done[sender][sender_slot] = True
                done[receiver][receiver_slot] = True
                progressed = True
        for stage in range(pp):
            if call_idx[stage] < len(plans[stage]) and all(done[stage]):
                call_idx[stage] += 1
                if call_idx[stage] < len(plans[stage]):
                    post(stage)
                progressed = True
        if all(call_idx[stage] == len(plans[stage]) for stage in range(pp)):
            return
        assert progressed, f"p2p deadlock: stalls with call {[call_idx[s] for s in range(pp)]}"


def _cases():
    # A long-sequence batch makes every stage need so many forwards before its
    # first backward that the warmup depth stops mattering and the stages end up
    # with the *same* op list. Short sequences are the interesting case: the
    # first group can backward almost immediately, so stages drift out of phase
    # and the fusion at each turn is what keeps them alive.
    batches = (
        [2048, 700, 300, 1024, 1500, 512, 900, 2600],
        [200, 180, 150, 220, 90],
        [300, 2048, 250, 900, 120],
        [96, 96, 96, 96],
    )
    for lengths in batches:
        for pp in (2, 3, 4):
            for policy in ("staggered", "lockstep"):
                for keep_last_k in (1, 2, 100):
                    yield lengths, 512, policy, keep_last_k, pp


@pytest.mark.parametrize("lengths,chunk_size,policy,keep_last_k,pp", list(_cases()))
def test_plan_has_no_deadlock(lengths, chunk_size, policy, keep_last_k, pp):
    plans = _plans(lengths, chunk_size, policy, keep_last_k, pp)
    _simulate(plans, pp)


@pytest.mark.parametrize("pp", [2, 3, 4])
def test_simulation_rejects_an_unfused_plan(pp):
    """The check above is only worth something if it can fail: the same op
    order with every send on its own -- no fusion at the turns -- is the
    arrangement the fused calls exist to avoid, and the simulator has to call
    it a hang.

    Only short batches show it, because that is where the stages stop sharing
    an op list; with one list per stage the phases line up and bare sends
    happen to pair by accident.
    """
    lengths = [200, 180, 150, 220, 90]
    plans = []
    for stage in range(pp):
        plan = []
        for kind, _ in _ops(lengths, 512, "staggered", 1, pp, stage):
            before = after = ()
            if kind in ("F", "Fp"):
                before = () if stage == 0 else ("recv_fwd",)
                after = () if stage == pp - 1 else ("send_fwd",)
            elif kind == "B":
                before = () if stage == pp - 1 else ("recv_bwd",)
                after = () if stage == 0 else ("send_bwd",)
            plan.append((before, after))
        plans.append(plan)
    with pytest.raises(AssertionError, match="deadlock"):
        _simulate(plans, pp)


@pytest.mark.parametrize("lengths,chunk_size,policy,keep_last_k,pp", list(_cases()))
def test_plan_covers_every_kind_exactly_once(lengths, chunk_size, policy, keep_last_k, pp):
    """One plan entry per op, so the executor's ``plan[idx]`` lines up."""
    plans = _plans(lengths, chunk_size, policy, keep_last_k, pp)
    for stage, plan in enumerate(plans):
        assert len(plan) == len(_ops(lengths, chunk_size, policy, keep_last_k, pp, stage)), f"stage {stage}"
        for kind, (before, after) in zip(_ops(lengths, chunk_size, policy, keep_last_k, pp, stage), plan):
            if kind == "Fr":
                assert before == () and after == (), "a local recompute must not communicate"
            if kind == "B":
                assert not any("fwd" in action for action in after), "a backward sends only backwards"
            elif kind in ("F", "Fp"):
                assert not any("bwd" in action for action in before), "a forward receives only forwards"
            assert sum("send" in action for action in after) <= 1
            assert sum("recv" in action for action in before) <= 1


@pytest.mark.parametrize("lengths,chunk_size,policy,keep_last_k,pp", list(_cases()))
def test_neighbours_pair_the_same_units(lengths, chunk_size, policy, keep_last_k, pp):
    """A p2p pair is matched by position within a direction, so stage ``s``'s
    k-th forward send must be the unit stage ``s+1`` receives as its k-th."""
    per_stage = [_ops(lengths, chunk_size, policy, keep_last_k, pp, stage) for stage in range(pp)]
    forwards = {tuple(uid for kind, uid in ops if kind in ("F", "Fp")) for ops in per_stage}
    backwards = {tuple(uid for kind, uid in ops if kind == "B") for ops in per_stage}
    assert len(forwards) == 1, f"pp={pp}: stages disagree on the forward order"
    assert len(backwards) == 1, f"pp={pp}: stages disagree on the backward order"


@pytest.mark.parametrize("lengths,chunk_size,policy,keep_last_k,pp", list(_cases()))
def test_every_unit_is_forwarded_and_backwarded_exactly_once(lengths, chunk_size, policy, keep_last_k, pp):
    """The executor keys in-flight tensors by unit and pops exactly one per
    ``B``, so a unit with two forwards or no backward would corrupt the
    pairing."""
    for stage in range(pp):
        ops = _ops(lengths, chunk_size, policy, keep_last_k, pp, stage)
        for unit_id in {uid for _, uid in ops}:
            kinds = [kind for kind, uid in ops if uid == unit_id]
            assert kinds.count("B") == 1, f"pp={pp} stage={stage}: unit {unit_id} has {kinds}"
            assert sum(kind in ("F", "Fp") for kind in kinds) == 1, f"pp={pp} stage={stage}: unit {unit_id} {kinds}"
            assert sum(kind == "Fr" for kind in kinds) <= 1, f"pp={pp} stage={stage}: unit {unit_id} {kinds}"
