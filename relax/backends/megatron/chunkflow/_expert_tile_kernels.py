# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Device-scheduled GEMMs; imported only by the CUDA expert prototype."""

import triton as tr
import triton.language as tl


@tr.jit
def _linear_tile(
    X,
    W,
    Y,
    ROWS,
    start,
    end,
    column,
    N: tl.constexpr,
    K: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = start + tl.arange(0, BM)
    cols = column + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k_tile in range(tr.cdiv(K, BK)):
        k = k_tile * BK + ks
        x = tl.load(
            X + rows[:, None] * K + k[None, :], (rows[:, None] < end) & (rows[:, None] < ROWS) & (k[None, :] < K), 0
        )
        w = tl.load(W + cols[None, :] * WN + k[:, None] * WK, (cols[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < end) & (rows[:, None] < ROWS) & (cols[None, :] < N),
    )


@tr.jit(do_not_specialize=["ROWS"])
def linear(
    X,
    W,
    Y,
    OFFSETS,
    ROWS,
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
    POINTERS: tl.constexpr = False,
    SOURCE_COUNTS=None,
    SOURCE: tl.constexpr = -1,
):
    task = tl.program_id(0).to(tl.int64)
    first_task = tl.full((), 0, tl.int64)
    n_tiles = tr.cdiv(N, BN)
    for expert in range(EXPERTS):
        start = tl.load(OFFSETS + expert)
        end = tl.load(OFFSETS + expert + 1)
        if SOURCE >= 0:
            if SOURCE == 1:
                start += tl.load(SOURCE_COUNTS + expert)
            end = start + tl.load(SOURCE_COUNTS + SOURCE * EXPERTS + expert)
        if POINTERS:
            weight = tl.load(W + expert).to(tl.pointer_type(tl.bfloat16))
        else:
            weight = W + expert * WE
        if BM == 0:
            full = (end - start) // 256
            half = ((end - start) % 256) // 128
            tail = tl.cdiv((end - start) % 128, 64)
            tasks = (full + half + tail) * n_tiles
        else:
            tasks = tl.cdiv(end - start, BM) * n_tiles
        while task < first_task + tasks:
            local_task = task - first_task
            block = local_task // n_tiles
            column = (local_task % n_tiles) * BN
            if BM == 0:
                if block < full:
                    _linear_tile(X, weight, Y, ROWS, start + block * 256, end, column, N, K, WN, WK, 256, BN, BK)
                elif block < full + half:
                    _linear_tile(X, weight, Y, ROWS, start + full * 256, end, column, N, K, WN, WK, 128, BN, BK)
                else:
                    _linear_tile(
                        X,
                        weight,
                        Y,
                        ROWS,
                        start + full * 256 + half * 128 + (block - full - half) * 64,
                        end,
                        column,
                        N,
                        K,
                        WN,
                        WK,
                        64,
                        BN,
                        BK,
                    )
            else:
                _linear_tile(X, weight, Y, ROWS, start + block * BM, end, column, N, K, WN, WK, BM, BN, BK)
            task += WORKERS
        first_task += tasks


@tr.jit(do_not_specialize=["ROWS"])
def weight_grad(
    X,
    DY,
    DW,
    OFFSETS,
    ROWS,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    POINTERS: tl.constexpr = False,
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
        if POINTERS:
            destination = tl.load(DW + expert).to(tl.pointer_type(tl.float32))
        else:
            destination = DW + expert * N * K
        ptr = destination + ns[:, None] * K + ks[None, :]
        mask = (ns[:, None] < N) & (ks[None, :] < K)
        if ACCUMULATE:
            acc += tl.load(ptr, mask, 0)
        tl.store(ptr, acc, mask)


@tr.jit
def dispatch_offsets(COUNTS, OFFSETS, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.arange(0, BLOCK)
    first = tl.load(COUNTS + expert, expert < EXPERTS, 0).to(tl.int64)
    second = tl.load(COUNTS + EXPERTS + expert, expert < EXPERTS, 0).to(tl.int64)
    prefix0, prefix1 = tl.cumsum(first), tl.cumsum(second)
    tl.store(OFFSETS, 0)
    tl.store(OFFSETS + EXPERTS + 1, 0)
    tl.store(OFFSETS + 2 * (EXPERTS + 1), 0)
    tl.store(OFFSETS + expert + 1, prefix0, expert < EXPERTS)
    tl.store(OFFSETS + EXPERTS + 2 + expert, prefix1, expert < EXPERTS)
    tl.store(OFFSETS + 2 * (EXPERTS + 1) + expert + 1, prefix0 + prefix1, expert < EXPERTS)


@tr.jit(do_not_specialize=["ROWS"])
def source_copy(
    X,
    P,
    Y,
    Q,
    SOURCE_OFFSETS,
    OFFSETS,
    SOURCE_COUNTS,
    ROWS,
    H: tl.constexpr,
    EXPERTS: tl.constexpr,
    SOURCE: tl.constexpr,
    REVERSE: tl.constexpr,
    BR: tl.constexpr = 16,
    BC: tl.constexpr = 256,
):
    """Copy one source's expert-major rows to/from canonical [expert, source]
    order."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    left = tl.full((BR,), 0, tl.int32)
    right = tl.full((BR,), EXPERTS, tl.int32)
    for _ in range((EXPERTS + 1).bit_length()):
        middle = (left + right) // 2
        boundary = tl.load(SOURCE_OFFSETS + middle + 1, middle < EXPERTS, 0x7FFFFFFFFFFFFFFF)
        move = rows >= boundary
        left = tl.where(move, middle + 1, left)
        right = tl.where(move, right, middle)
    expert = tl.minimum(left, EXPERTS - 1)
    target = tl.load(OFFSETS + expert) + rows - tl.load(SOURCE_OFFSETS + expert)
    if SOURCE == 1:
        target += tl.load(SOURCE_COUNTS + expert)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < H)
    source_ptr = X + rows[:, None] * H + cols[None, :]
    target_ptr = Y + target[:, None] * H + cols[None, :]
    if REVERSE:
        tl.store(source_ptr, tl.load(target_ptr, mask, 0), mask)
    else:
        tl.store(target_ptr, tl.load(source_ptr, mask, 0), mask)
    if tl.program_id(1) == 0:
        if REVERSE:
            tl.store(P + rows, tl.load(Q + target, rows < ROWS, 0), rows < ROWS)
        else:
            tl.store(Q + target, tl.load(P + rows, rows < ROWS, 0), rows < ROWS)
