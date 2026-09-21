# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""chunkflow scheduling: chunk construction, op ordering, and the multi-stage
view.

This module decides *what* needs to happen and in *what order* — it does not
execute anything (see ``attention_ops.py`` / ``StateStore`` for that).

Two building blocks, both scoped to *one training step's* batch of sequences:

``build_chunks``
    Splits sequences longer than ``chunk_size`` into ascending, causally-
    dependent chunks (a "dependency group"); bin-packs the remaining short
    sequences (and, optionally, long sequences' leftover tail chunks — see
    ``repack_tail``) into standalone chunks that carry no ordering
    constraint at all.

``build_group_schedule`` / ``build_schedule``
    For a chunk_size that would otherwise force a dependency group to hold
    every one of its chunks' activations simultaneously, only the *last*
    ``K`` chunks keep their activation from the first forward pass; the
    earlier ones are recomputed right before their own backward. This
    implements the paper-prose reading of "the forward passes of the first
    (N-K) chunks are executed twice" — deliberately *not* the literal
    Algorithm-2-pseudocode reading (``if idx >= K: keep``), which produces
    incomplete gradients (a later query chunk's backward can run before an
    even-later chunk that reads the same K/V has backpropagated). See
    ``validate_backward_order`` and the "ascending drop order" test in
    ``tests/backends/megatron/test_chunkflow_scheduler.py`` for the concrete
    failure this avoids.

On top of those sits the multi-stage layer (``classify_chunks``,
``make_bwd_order``, ``gen_state_aware_1f1b_ops``): given a pipeline size and a
stage index it produces that one stage's op sequence, keeping the same global
forward and backward orders on every stage so p2p send/recv pairs stay
aligned. Everything it needs comes from the chunk list, and the chunk list
comes from sequence lengths — which every PP stage has, so no stage needs the
plan broadcast to it.

Every generated schedule can (and, for anything driving real chunk_attention_*
calls, should) be run through ``validate_schedule`` first: a violation means
the caller's chunk ordering is wrong and any gradient computed downstream
would be silently incorrect, exactly the failure mode ``StateStore`` guards
against at runtime. These validators are the offline, look-before-you-run
counterpart to that runtime enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


Op = tuple[str, int]  # ("F" | "Fp" | "Fr" | "B", chunk_id)

# "F"  — normal forward, builds the graph (keeps the intermediate results / activation)
# "Fp" — first-pass forward, keeps only K/V, discards the activation
# "Fr" — recompute forward, right before this chunk's own backward
# "B"  — backward
FORWARD_KINDS = ("F", "Fp", "Fr")


@dataclass(frozen=True)
class Segment:
    """A contiguous run of tokens from one original sequence, inside a chunk.

    ``prefix`` is how many tokens of this sequence precede this segment —
    needed by a real attention op to know how much KV-cache history this
    segment's queries must attend over.
    """

    seq_id: int
    prefix: int
    length: int


@dataclass
class Chunk:
    """One pipeline unit: one or more ``Segment``s packed together.

    ``group`` is the id of the dependency group this chunk belongs to (the
    originating long sequence's id), or ``None`` for a standalone chunk with no
    ordering constraint. ``group_idx`` is this chunk's 0-based position within
    the group, ascending.
    """

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


# ---------------------------------------------------------------------------
# Chunk construction (Algorithm 1)
# ---------------------------------------------------------------------------


def _bin_pack_min_bins(lengths: list[int], cap: int) -> list[list[int]]:
    """Pack ``lengths`` into as few bins of capacity ``cap`` as possible.

    Starts from the theoretical lower bound (``ceil(sum/cap)``) and tries best-
    fit-decreasing at each bin count until one succeeds, so short sequences
    that need no ordering constraint at all don't waste pipeline units one-per-
    chunk.
    """
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
    """Split/pack a batch of sequences (by length, 0-indexed) into chunks.

    Sequences longer than ``chunk_size`` become an ascending dependency
    group of ``ceil(length / chunk_size)`` chunks. Sequences at or under
    ``chunk_size`` are bin-packed together into standalone chunks.

    A long sequence's last chunk is often smaller than ``chunk_size`` (e.g.
    a ``chunk_size + 2`` sequence produces one full chunk plus a lone
    2-token tail) — pure overhead if left alone: that tiny chunk still
    occupies a full pipeline unit *and* carries the group's ordering
    constraint. ``repack_tail`` and ``drop_tail`` are two independent,
    mutually exclusive switches for what to do about it — passing both
    raises ``ValueError``; passing neither keeps that tiny tail chunk as-is
    (the fragmentation defect described above).

    ``repack_tail`` (default on): folds the tail into the short-sequence
    packing pool instead (capped at one tail per chunk, so a chunk never
    ends up belonging to two dependency groups at once). This trades memory
    for it: those short sequences are no longer "free"
    (forward-then-immediately-backward-and-free) — they must now wait for
    the whole dependency group's forward pass to finish before the chunk
    they share can backward. Prefer it on data that is strongly
    long-tailed; on weakly long-tailed data the memory cost can outweigh
    the pipeline-slot savings.

    ``drop_tail`` (default off): discards the tail's tokens outright
    instead of building a chunk for them at all — the sequence's
    dependency group shrinks by one chunk (``group_size`` is adjusted to
    match) and those trailing tokens never get a forward or backward.
    This is **lossy**: if the tail happens to fall inside content that
    matters (e.g. the final answer tokens or an EOS marker in an RL
    response), that content silently never contributes to the loss for
    this sequence. Only turn this on once you've confirmed the dropped
    fragment doesn't matter for your data/objective — it buys the cheapest
    possible fix (no extra chunk, no extra memory) at a correctness risk
    that ``repack_tail`` doesn't carry.

    ``drop_tail_max_tokens`` bounds how much ``drop_tail`` is allowed to
    discard per sequence — dropping is meant for a handful of leftover
    tokens, not for silently truncating a large fraction of a sequence. A
    tail at or above this cap is **not** dropped; it falls through to the
    same tiny-own-chunk behavior as if both switches were off. Defaults to
    a quarter of ``chunk_size`` when left unset; ignored when ``drop_tail``
    is off.
    """
    if repack_tail and drop_tail:
        raise ValueError("repack_tail and drop_tail are mutually exclusive tail-handling strategies")

    max_drop = drop_tail_max_tokens if drop_tail_max_tokens is not None else chunk_size // 4

    long_ids = [i for i, n in enumerate(seq_lengths) if n > chunk_size]
    short_ids = [i for i, n in enumerate(seq_lengths) if n <= chunk_size]

    chunks: list[Chunk] = []
    cid = 0
    tails: list[tuple[int, int, int, int, int]] = []  # (seq_id, prefix, length, group_idx, group_size)

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
                continue  # discard these trailing tokens entirely — see drop_tail docstring
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

    # Each tail claims its own slot first; short sequences best-fit into whatever's left.
    slots: list[list] = [[t[2], t, []] for t in tails]  # [load, tail_or_None, [short idx, ...]]
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
    """Gather dependent chunks by group id, each group sorted ascending by
    ``group_idx``."""
    groups: dict[int, list[Chunk]] = {}
    for c in chunks:
        if c.is_dependent:
            groups.setdefault(c.group, []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: c.group_idx)
    return groups


# ---------------------------------------------------------------------------
# Op-sequence generation (Algorithm 2, paper-prose semantics only)
# ---------------------------------------------------------------------------


def build_group_schedule(group: Sequence[Chunk], keep_last_k: int) -> list[Op]:
    """Ascending-forward / descending-backward op sequence for one dependency
    group.

    If the group has at most ``keep_last_k`` chunks, every chunk simply
    keeps its activation from the first forward (no recompute needed).
    Otherwise, only the *last* ``keep_last_k`` chunks keep their activation;
    the earlier ``N - keep_last_k`` are recomputed (``Fr``) immediately
    before their own backward, processed in descending order — this is what
    keeps the overall backward order strictly descending (see module
    docstring for why the alternative, ascending-order reading, produces
    incomplete gradients).
    """
    group = sorted(group, key=lambda c: c.group_idx)
    n = len(group)

    if n <= keep_last_k:
        ops: list[Op] = [("F", c.chunk_id) for c in group]
        ops.extend(("B", c.chunk_id) for c in reversed(group))
        return ops

    keep_ids = {c.chunk_id for c in group[n - keep_last_k :]}
    drop_descending = list(reversed(group[: n - keep_last_k]))

    ops = [(("F" if c.chunk_id in keep_ids else "Fp"), c.chunk_id) for c in group]
    ops.extend(("B", c.chunk_id) for c in reversed(group) if c.chunk_id in keep_ids)
    for c in drop_descending:
        ops.append(("Fr", c.chunk_id))
        ops.append(("B", c.chunk_id))
    return ops


def build_schedule(chunks: Sequence[Chunk], keep_last_k: int) -> list[Op]:
    """Combine standalone chunks and every dependency group's schedule into one
    PP=1-view op sequence."""
    ops: list[Op] = []
    for c in chunks:
        if not c.is_dependent:
            ops.append(("F", c.chunk_id))
            ops.append(("B", c.chunk_id))
    for _, group in sorted(group_of(chunks).items()):
        ops.extend(build_group_schedule(group, keep_last_k))
    return ops


# ---------------------------------------------------------------------------
# Validators — the offline counterpart to StateStore's runtime assertions
# ---------------------------------------------------------------------------


def validate_forward_order(chunks: Sequence[Chunk], ops: Sequence[Op]) -> list[str]:
    """Every dependent chunk's forward must come after all lower-``group_idx``
    chunks in its group."""
    by_id = {c.chunk_id: c for c in chunks}
    seen_fwd: set[int] = set()
    problems: list[str] = []
    for kind, cid in ops:
        c = by_id[cid]
        if kind in FORWARD_KINDS:
            if c.is_dependent:
                predecessors = [o.chunk_id for o in chunks if o.group == c.group and o.group_idx < c.group_idx]
                missing = [x for x in predecessors if x not in seen_fwd]
                if missing:
                    problems.append(
                        f"forward order violated: chunk {cid} (group {c.group}, idx {c.group_idx}) "
                        f"forwarded before predecessor chunk(s) {missing}"
                    )
            seen_fwd.add(cid)
    return problems


def validate_backward_order(chunks: Sequence[Chunk], ops: Sequence[Op]) -> list[str]:
    """Every dependent chunk's backward must come after all
    higher-``group_idx`` chunks in its group.

    A violation means dK/dV accumulation for the lower chunk is incomplete
    (some later query chunk's contribution hasn't landed yet) — a silent wrong-
    gradient bug, not a crash.
    """
    by_id = {c.chunk_id: c for c in chunks}
    done_bwd: set[int] = set()
    problems: list[str] = []
    for kind, cid in ops:
        c = by_id[cid]
        if kind == "B":
            if c.is_dependent:
                successors = [o.chunk_id for o in chunks if o.group == c.group and o.group_idx > c.group_idx]
                missing = [x for x in successors if x not in done_bwd]
                if missing:
                    problems.append(
                        f"backward order violated: chunk {cid} (group {c.group}, idx {c.group_idx}) "
                        f"backwarded before successor chunk(s) {missing} -> incomplete gradient"
                    )
            done_bwd.add(cid)
    return problems


def validate_completeness(chunks: Sequence[Chunk], ops: Sequence[Op]) -> list[str]:
    """Every chunk must backward exactly once, and only after a graph-building
    forward (F or Fr, not just Fp)."""
    problems: list[str] = []
    graph_fwd: set[int] = set()
    bwd_count: dict[int, int] = {}
    for kind, cid in ops:
        if kind in ("F", "Fr"):
            graph_fwd.add(cid)
        elif kind == "B":
            if cid not in graph_fwd:
                problems.append(
                    f"chunk {cid} backwarded without a prior graph-building forward (Fp alone isn't enough)"
                )
            bwd_count[cid] = bwd_count.get(cid, 0) + 1
    for c in chunks:
        count = bwd_count.get(c.chunk_id, 0)
        if count != 1:
            problems.append(f"chunk {c.chunk_id} backwarded {count} time(s) (expected exactly 1)")
    return problems


def validate_schedule(chunks: Sequence[Chunk], ops: Sequence[Op]) -> list[str]:
    """Run all three checks; empty return means the schedule is safe to
    execute."""
    return (
        validate_forward_order(chunks, ops) + validate_backward_order(chunks, ops) + validate_completeness(chunks, ops)
    )


# ---------------------------------------------------------------------------
# Multi-stage view: how one stage's op sequence is laid out across a pipeline
# ---------------------------------------------------------------------------
#
# Everything above is the single-GPU (PP=1) view. The functions below answer
# "which ops does *this* stage execute, in what order" for a real pipeline —
# the layer the module docstring calls "a separate, later layer".
#
# They are pure functions of the chunk list, and the chunk list is built from
# sequence lengths, which every PP stage receives. That is what makes the
# schedule derivable independently on each stage: the p2p send/recv pairs stay
# aligned without anyone broadcasting the plan.
#
# Strategies control where a recompute op (``Fr``) lands, mirroring the paper's
# Figure 6 discussion:
#
#   eager — Fr issued early (``fr_lead = 2*(pp - 1 - stage)``); crosses stage
#           boundaries, so it overlaps.
#   lazy  — Fr right before its own B but still waiting on the previous stage's
#           Fr; serialises badly across stages.
#   local — Fr right before its own B, recomputed from this stage's own Fp
#           only; keeps eager's speed and lazy's activation peak.

STRATEGIES = ("eager", "lazy", "local")


def classify_chunks(chunks: Sequence[Chunk], keep_last_k: int) -> tuple[set[int], set[int]]:
    """Split chunks into those that keep their first-pass activation and those
    that must be recomputed (``Fr``) before their own backward.

    Per Algorithm 2's paper-prose semantics: the *last* ``K`` chunks of each
    dependency group keep; standalone chunks always keep (their N is 1).
    """
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


def make_bwd_order(chunks: Sequence[Chunk], fwd_order: Sequence[int]) -> list[int]:
    """Global backward order for a given forward order.

    A standalone chunk backwards immediately after its own forward. A
    dependency group backwards (descending) only once every member has appeared
    in ``fwd_order`` — that is the condition for the group's dK/dV accumulation
    to be complete when its lowest chunk runs (see
    ``validate_backward_order``).
    """
    by_id = {c.chunk_id: c for c in chunks}
    groups = {g: [c.chunk_id for c in members] for g, members in group_of(chunks).items()}
    seen: set[int] = set()
    out: list[int] = []
    for cid in fwd_order:
        seen.add(cid)
        c = by_id[cid]
        if not c.is_dependent:
            out.append(cid)
            continue
        members = groups[c.group]
        if all(m in seen for m in members):
            out.extend(reversed(members))
    return out


def gen_state_aware_1f1b_ops(
    chunks: Sequence[Chunk],
    keep_last_k: int,
    pp: int,
    stage: int,
    strategy: str = "local",
    fr_lead: int | None = None,
    bwd_lag: int | None = None,
) -> list[Op]:
    """The op sequence *one stage* executes, in order.

    Four steps: classify chunks into keep (``F``) vs drop (``Fp``); build the
    global backward order; run the standard 1F1B warmup/steady interleave over
    it; splice an ``Fr`` in ``fr_lead`` ops ahead of each ``B`` that needs one.

    Every stage must call this with the same ``chunks``/``keep_last_k``/``pp``
    and with its own ``stage`` — that is what keeps p2p pairs aligned.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    keep, need_fr = classify_chunks(chunks, keep_last_k)
    fwd_order = [c.chunk_id for c in chunks]
    bwd_order = make_bwd_order(chunks, fwd_order)

    if fr_lead is None:
        fr_lead = 2 * max(0, pp - stage - 1) if strategy == "eager" else 0

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
            break  # bwd_order isn't a permutation of fwd_order -- caller bug

    if not need_fr:
        return base

    out: list[Op] = []
    emitted: set[int] = set()
    fp_seen: set[int] = set()  # Fr(u) must come after Fp(u) -- no K/V to reuse otherwise
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
