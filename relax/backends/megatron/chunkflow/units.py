# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execution units: which chunks share one forward/backward call.

``scheduler.py`` decides the op order over *chunks*. This module decides how
chunks are grouped into the things that actually get executed, and produces the
same op order over those units.

Why grouping matters: an attention call happens once per (layer, unit), and the
measured cost of chunkflow's attention is dominated by call count rather than
FLOPs — one call per sequence instead of one per batch already costs 4.57x on
its own. So the number of *calls* is set by how many chunks share a unit, not
by how the op sequence is arranged. Merging chunks into units is therefore the
lever for that cost, and the trade is explicit:

``staggered``
    One unit per chunk. Dependency groups can sit at different depths, which
    is what lets the schedule fill pipeline bubbles — but every chunk is its
    own attention call.

``lockstep``
    Chunks at the same depth (``group_idx``) across dependency groups share a
    unit. Far fewer calls, but groups march together, so there is little
    stagger left to hide bubbles with.

Both are correct; they trade call count against schedule flexibility, and only
end-to-end measurement can say which wins.

Merging is safe because chunks at the same depth in *different* groups are
mutually independent: a dependency only ever runs between adjacent depths of
the same group (``group_idx - 1`` -> ``group_idx``). A unit may never contain
two chunks of the same group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from relax.backends.megatron.chunkflow.scheduler import (
    STRATEGIES,
    Chunk,
    Op,
    classify_chunks,
    interleave_1f1b,
    splice_recomputes,
)


POLICIES = ("staggered", "lockstep")


@dataclass
class Unit:
    """One execution unit: the chunks that share a single forward/backward
    call.

    ``chunks`` is never empty. A unit whose members all belong to one
    dependency group carries that group; a unit spanning several groups (only
    possible under ``lockstep``) carries all of them, and its backward must
    wait for every one of them to finish forwarding.
    """

    unit_id: int
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("a Unit must hold at least one chunk")
        groups = [c.group for c in self.chunks if c.is_dependent]
        if len(groups) != len(set(groups)):
            raise ValueError(
                f"unit {self.unit_id} holds two chunks of the same dependency group {groups} -- "
                "they are causally ordered and must not share a unit"
            )

    @property
    def groups(self) -> frozenset[int]:
        """Dependency groups this unit touches, ascending.

        Empty if purely standalone.
        """
        return frozenset(c.group for c in self.chunks if c.is_dependent)

    @property
    def round(self) -> int:
        """Depth in the dependency chain: ``group_idx``, or -1 for standalone
        units."""
        return max((c.group_idx for c in self.chunks if c.is_dependent), default=-1)

    @property
    def num_tokens(self) -> int:
        return sum(c.num_tokens for c in self.chunks)

    @property
    def is_dependent(self) -> bool:
        return bool(self.groups)

    @property
    def chunk_ids(self) -> list[int]:
        return [c.chunk_id for c in self.chunks]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_micro_batches(chunks: Sequence[Chunk], policy: str = "staggered") -> list[Unit]:
    """Partition ``chunks`` into execution units, in execution order.

    The flattened order of the result is the *forward order*: the caller feeds
    it to the unit-level schedule functions. For ``staggered`` that order is
    exactly the chunk order, so it reduces to the chunk-level schedule.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy!r}")

    if policy == "staggered":
        return [Unit(unit_id=i, chunks=[c]) for i, c in enumerate(chunks)]

    units: list[Unit] = []
    # Standalone chunks are already bin-packed by ``build_chunks`` and carry no
    # ordering constraint, so they go first and backward immediately.
    for c in chunks:
        if not c.is_dependent:
            units.append(Unit(unit_id=len(units), chunks=[c]))

    by_round: dict[int, list[Chunk]] = {}
    for c in chunks:
        if c.is_dependent:
            by_round.setdefault(c.group_idx, []).append(c)
    for round_idx in sorted(by_round):
        units.append(Unit(unit_id=len(units), chunks=by_round[round_idx]))

    return units


def expand_units(units: Sequence[Unit]) -> list[Chunk]:
    """Every chunk, in unit order — the flat list the chunk-level planner
    needs."""
    return [c for u in units for c in u.chunks]


# ---------------------------------------------------------------------------
# Unit-level op ordering
# ---------------------------------------------------------------------------


def classify_units(units: Sequence[Unit], chunks: Sequence[Chunk], keep_last_k: int) -> tuple[set[int], set[int]]:
    """Split units into those that keep their first-pass activation and those
    to recompute.

    A unit can only be dropped when *every* member can be: dropping discards
    activations for the whole unit, so a single member that must keep forces
    the unit to keep. With equal-depth merging and groups of differing sizes
    this costs some of the window's saving — the alternative (splitting the
    unit) would cost the call-count saving instead.
    """
    _, chunk_need_fr = classify_chunks(chunks, keep_last_k)

    keep: set[int] = set()
    need_fr: set[int] = set()
    for u in units:
        if set(u.chunk_ids) <= chunk_need_fr:
            need_fr.add(u.unit_id)
        else:
            keep.add(u.unit_id)
    return keep, need_fr


def make_unit_bwd_order(units: Sequence[Unit], fwd_order: Sequence[int]) -> list[int]:
    """Global backward order over units for a given forward order.

    A standalone unit backwards immediately after its own forward. A unit
    touching dependency groups backwards (descending by round) only once every
    unit of *every* group it touches has forwarded — that is when the dK/dV
    accumulating into its members is complete. Reduces exactly to
    ``scheduler.make_bwd_order`` when every unit holds one chunk.
    """
    by_id = {u.unit_id: u for u in units}
    units_of_group: dict[int, list[int]] = {}
    for u in units:
        for g in u.groups:
            units_of_group.setdefault(g, []).append(u.unit_id)

    seen: set[int] = set()
    out: list[int] = []
    emitted: set[int] = set()
    for uid in fwd_order:
        seen.add(uid)
        u = by_id[uid]
        if not u.is_dependent:
            out.append(uid)
            emitted.add(uid)
            continue
        if not all(all(m in seen for m in units_of_group[g]) for g in u.groups):
            continue
        for g in sorted(u.groups):
            for m in sorted(units_of_group[g], key=lambda x: -by_id[x].round):
                if m not in emitted:
                    out.append(m)
                    emitted.add(m)
    return out


def gen_unit_ops(
    units: Sequence[Unit],
    chunks: Sequence[Chunk],
    keep_last_k: int,
    pp: int,
    stage: int,
    strategy: str = "local",
    fr_lead: int | None = None,
    bwd_lag: int | None = None,
) -> list[Op]:
    """The unit-level op sequence one stage executes, in order.

    Same shape as ``scheduler.gen_state_aware_1f1b_ops`` with units in place of
    chunks; every stage must derive it from the same ``units`` to keep p2p
    pairs aligned.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    keep, need_fr = classify_units(units, chunks, keep_last_k)
    fwd_order = [u.unit_id for u in units]
    bwd_order = make_unit_bwd_order(units, fwd_order)

    if fr_lead is None:
        fr_lead = 2 * max(0, pp - stage - 1) if strategy == "eager" else 0

    base = interleave_1f1b(fwd_order, bwd_order, keep, pp, stage, bwd_lag)
    if not need_fr:
        return base
    return splice_recomputes(base, need_fr, fr_lead)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_units(units: Sequence[Unit], chunks: Sequence[Chunk]) -> list[str]:
    """Every chunk appears in exactly one unit, and no unit spans two chunks of
    one group."""
    problems: list[str] = []
    seen: list[int] = []
    for u in units:
        groups = [c.group for c in u.chunks if c.is_dependent]
        if len(groups) != len(set(groups)):
            problems.append(f"unit {u.unit_id} holds two chunks of the same group {groups}")
        seen.extend(u.chunk_ids)

    expected = sorted(c.chunk_id for c in chunks)
    if sorted(seen) != expected:
        missing = set(expected) - set(seen)
        extra = set(seen) - set(expected)
        problems.append(f"units do not tile the chunks: missing={sorted(missing)} extra={sorted(extra)}")
    if len(seen) != len(set(seen)):
        problems.append("a chunk appears in more than one unit")
    return problems


def validate_unit_backward_order(units: Sequence[Unit], ops: Sequence[Op]) -> list[str]:
    """Every dependent unit's backward must follow all later-round units of its
    groups.

    A violation means dK/dV for that unit's members is incomplete when they
    backward — a silently wrong gradient, not a crash. Unit-level analogue of
    ``scheduler.validate_backward_order``.
    """
    by_id = {u.unit_id: u for u in units}
    done: set[int] = set()
    problems: list[str] = []
    for kind, uid in ops:
        if kind != "B":
            continue
        u = by_id[uid]
        if u.is_dependent:
            successors = [o.unit_id for o in units if o.groups & u.groups and o.round > u.round]
            missing = [x for x in successors if x not in done]
            if missing:
                problems.append(
                    f"backward order violated: unit {uid} (round {u.round}, groups {sorted(u.groups)}) "
                    f"backwarded before later-round unit(s) {missing} -> incomplete gradient"
                )
        done.add(uid)
    return problems


def unit_lookup(units: Iterable[Unit]) -> dict[int, Unit]:
    """``unit_id -> Unit``, for the execution layer."""
    return {u.unit_id: u for u in units}
