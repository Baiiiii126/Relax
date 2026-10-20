# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from collections.abc import Sequence
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
        return frozenset(c.group for c in self.chunks if c.is_dependent)

    @property
    def round(self) -> int:
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


def build_micro_batches(chunks: Sequence[Chunk], policy: str = "staggered") -> list[Unit]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy!r}")

    if policy == "staggered":
        return [Unit(unit_id=i, chunks=[c]) for i, c in enumerate(chunks)]

    units: list[Unit] = []
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


def classify_units(units: Sequence[Unit], chunks: Sequence[Chunk], keep_last_k: int) -> tuple[set[int], set[int]]:
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


def completed_groups(unit: Unit) -> frozenset[int]:
    return frozenset(c.group for c in unit.chunks if c.is_dependent and c.group_idx == 0)


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
