# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


Op = tuple[str, int]


@dataclass(frozen=True)
class Segment:
    seq_id: int
    prefix: int
    length: int


@dataclass
class Chunk:
    chunk_id: int
    segments: list[Segment]
    group: int | None = None
    group_idx: int = 0
    group_size: int = 1

    @property
    def num_tokens(self) -> int:
        return sum(s.length for s in self.segments)

    @property
    def is_dependent(self) -> bool:
        return self.group is not None


def _bin_pack_min_bins(lengths: list[int], cap: int) -> list[list[int]]:
    if not lengths:
        return []
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    lower = max(1, math.ceil(sum(lengths) / cap))
    for num_bins in range(lower, len(lengths) + 1):
        loads = [0] * num_bins
        bins: list[list[int]] = [[] for _ in range(num_bins)]
        ok = True
        for i in order:
            best, best_load = -1, -1
            for b in range(num_bins):
                if loads[b] + lengths[i] <= cap and loads[b] > best_load:
                    best, best_load = b, loads[b]
            if best < 0:
                ok = False
                break
            loads[best] += lengths[i]
            bins[best].append(i)
        if ok:
            return [b for b in bins if b]
    return [[i] for i in range(len(lengths))]


def build_chunks(
    seq_lengths: Sequence[int],
    chunk_size: int,
    repack_tail: bool = True,
    drop_tail: bool = False,
    drop_tail_max_tokens: int | None = None,
) -> list[Chunk]:
    if repack_tail and drop_tail:
        raise ValueError("repack_tail and drop_tail are mutually exclusive tail-handling strategies")

    max_drop = drop_tail_max_tokens if drop_tail_max_tokens is not None else chunk_size // 4

    long_ids = [i for i, n in enumerate(seq_lengths) if n > chunk_size]
    short_ids = [i for i, n in enumerate(seq_lengths) if n <= chunk_size]

    chunks: list[Chunk] = []
    cid = 0
    tails: list[tuple[int, int, int, int, int]] = []

    for seq_id in long_ids:
        length = seq_lengths[seq_id]
        num_chunks = math.ceil(length / chunk_size)
        tail_len = length - (num_chunks - 1) * chunk_size
        should_drop_tail = drop_tail and tail_len < chunk_size and tail_len <= max_drop
        group_size = num_chunks - 1 if should_drop_tail else num_chunks

        for i in range(num_chunks):
            start = i * chunk_size
            seg_len = min(chunk_size, length - start)
            is_tail = i == num_chunks - 1 and seg_len < chunk_size
            if is_tail and should_drop_tail:
                continue
            if is_tail and repack_tail:
                tails.append((seq_id, start, seg_len, i, group_size))
                continue
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    segments=[Segment(seq_id, start, seg_len)],
                    group=seq_id,
                    group_idx=i,
                    group_size=group_size,
                )
            )
            cid += 1

    short_lengths = [seq_lengths[i] for i in short_ids]
    if not tails:
        for members in _bin_pack_min_bins(short_lengths, chunk_size):
            segs = [Segment(short_ids[i], 0, short_lengths[i]) for i in members]
            chunks.append(Chunk(chunk_id=cid, segments=segs))
            cid += 1
        return chunks

    slots: list[list] = [[t[2], t, []] for t in tails]
    free_cap = sum(max(0, chunk_size - slot[0]) for slot in slots)
    extra_slots = max(0, math.ceil(max(0, sum(short_lengths) - free_cap) / chunk_size))
    for _ in range(extra_slots):
        slots.append([0, None, []])
    for i in sorted(range(len(short_lengths)), key=lambda i: -short_lengths[i]):
        length = short_lengths[i]
        best, best_load = -1, -1
        for slot_idx, slot in enumerate(slots):
            if slot[0] + length <= chunk_size and slot[0] > best_load:
                best, best_load = slot_idx, slot[0]
        if best < 0:
            slots.append([length, None, [i]])
        else:
            slots[best][0] += length
            slots[best][2].append(i)

    for load, tail, idxs in slots:
        if load == 0:
            continue
        segs: list[Segment] = []
        group = group_idx = group_size = None
        if tail is not None:
            seq_id, prefix, seg_len, idx, size = tail
            segs.append(Segment(seq_id, prefix, seg_len))
            group, group_idx, group_size = seq_id, idx, size
        segs.extend(Segment(short_ids[i], 0, short_lengths[i]) for i in idxs)
        chunks.append(
            Chunk(chunk_id=cid, segments=segs, group=group, group_idx=group_idx or 0, group_size=group_size or 1)
        )
        cid += 1

    return chunks


def group_of(chunks: Sequence[Chunk]) -> dict[int, list[Chunk]]:
    groups: dict[int, list[Chunk]] = {}
    for c in chunks:
        if c.is_dependent:
            groups.setdefault(c.group, []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: c.group_idx)
    return groups


STRATEGIES = ("eager", "lazy", "local")


def classify_chunks(chunks: Sequence[Chunk], keep_last_k: int) -> tuple[set[int], set[int]]:
    keep: set[int] = set()
    need_fr: set[int] = set()
    for _, members in sorted(group_of(chunks).items()):
        n = len(members)
        if n <= keep_last_k:
            keep.update(c.chunk_id for c in members)
        else:
            keep.update(c.chunk_id for c in members[n - keep_last_k :])
            need_fr.update(c.chunk_id for c in members[: n - keep_last_k])
    for c in chunks:
        if not c.is_dependent:
            keep.add(c.chunk_id)
    return keep, need_fr


def interleave_1f1b(
    fwd_order: Sequence[int],
    bwd_order: Sequence[int],
    keep: set[int],
    pp: int,
    stage: int,
    bwd_lag: int | None = None,
) -> list[Op]:
    m, nb = len(fwd_order), len(bwd_order)
    base: list[Op] = []
    fi = bi = 0
    fpos: dict[int, int] = {}
    need = max(1, bwd_lag) if bwd_lag else 1

    def emit_fwd() -> None:
        nonlocal fi
        u = fwd_order[fi]
        fpos[u] = len(base)
        base.append(("F" if u in keep else "Fp", u))
        fi += 1

    for _ in range(min(max(0, pp - stage - 1), m)):
        emit_fwd()
    while fi < m or bi < nb:
        if fi < m:
            emit_fwd()
        if bi >= nb:
            continue
        ub = bwd_order[bi]
        if ub in fpos and (len(base) - fpos[ub] >= need or fi >= m):
            base.append(("B", ub))
            bi += 1
        elif fi >= m and ub not in fpos:
            break
    return base


def splice_recomputes(base: Sequence[Op], need_fr: set[int], fr_lead: int) -> list[Op]:
    out: list[Op] = []
    emitted: set[int] = set()
    fp_seen: set[int] = set()
    for i, op in enumerate(base):
        for j in range(i, min(len(base), i + fr_lead + 1)):
            kj, uj = base[j]
            if kj == "B" and uj in need_fr and uj not in emitted and uj in fp_seen:
                out.append(("Fr", uj))
                emitted.add(uj)
        out.append(op)
        if op[0] == "Fp":
            fp_seen.add(op[1])
    return out
