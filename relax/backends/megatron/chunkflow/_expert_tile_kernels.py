# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Device-scheduled GEMMs; imported only by the CUDA expert prototype."""

import triton as tr
import triton.language as tl


@tr.jit
def linear(
    X,
    W,
    Y,
    OFFSETS,
    ROWS: tl.constexpr,
    EXPERTS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WORKERS: tl.constexpr,
):
    task = tl.program_id(0).to(tl.int64)
    first_task = tl.full((), 0, tl.int64)
    n_tiles = tr.cdiv(N, BN)
    for expert in range(EXPERTS):
        start = tl.load(OFFSETS + expert)
        end = tl.load(OFFSETS + expert + 1)
        tasks = tl.cdiv(end - start, BM) * n_tiles
        while task < first_task + tasks:
            local_task = task - first_task
            rows = start + (local_task // n_tiles) * BM + tl.arange(0, BM)
            cols = (local_task % n_tiles) * BN + tl.arange(0, BN)
            ks = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for k_tile in range(tr.cdiv(K, BK)):
                k = k_tile * BK + ks
                x = tl.load(
                    X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < end) & (rows[:, None] < ROWS) & (k[None, :] < K),
                    0,
                )
                w = tl.load(
                    W + expert * WE + cols[None, :] * WN + k[:, None] * WK,
                    (cols[None, :] < N) & (k[:, None] < K),
                    0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                Y + rows[:, None] * N + cols[None, :],
                acc,
                (rows[:, None] < end) & (rows[:, None] < ROWS) & (cols[None, :] < N),
            )
            task += WORKERS
        first_task += tasks


@tr.jit
def weight_grad(
    X,
    DY,
    DW,
    OFFSETS,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    expert = tl.program_id(1)
    tile = tl.program_id(0)
    ns = (tile // tr.cdiv(K, BK)) * BN + tl.arange(0, BN)
    ks = (tile % tr.cdiv(K, BK)) * BK + tl.arange(0, BK)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    # Each output element has one writer; no atomic or partial-dW workspace.
    if end > start or not ACCUMULATE:
        acc = tl.zeros((BN, BK), tl.float32)
        for row_tile in range(tl.cdiv(end - start, BR)):
            rows = start + row_tile * BR + tl.arange(0, BR)
            dy = tl.load(
                DY + rows[None, :] * N + ns[:, None],
                (rows[None, :] < end) & (rows[None, :] < ROWS) & (ns[:, None] < N),
                0,
            )
            x = tl.load(
                X + rows[:, None] * K + ks[None, :],
                (rows[:, None] < end) & (rows[:, None] < ROWS) & (ks[None, :] < K),
                0,
            )
            acc = tl.dot(dy, x, acc)
        ptr = DW + expert * N * K + ns[:, None] * K + ks[None, :]
        mask = (ns[:, None] < N) & (ks[None, :] < K)
        if ACCUMULATE:
            acc += tl.load(ptr, mask, 0)
        tl.store(ptr, acc, mask)
