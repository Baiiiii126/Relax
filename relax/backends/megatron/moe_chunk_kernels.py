# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in kernels for BF16 / FP64 MoE experiments.

No persistent GPU storage.
"""

import triton
import triton.language as tl


# Routing-dependent row counts and split points are runtime scalars: making
# them constexpr would compile a new kernel for almost every chunk/layer.
@triton.jit
def pack_ids_probs(Map, Probs, Ids, Packed, N, E: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) * B + tl.arange(0, B)
    for k in tl.static_range(K):
        row = tl.load(Map + token * (2 * E + 1) + k, token < N, 0)
        expert = tl.load(Map + token * (2 * E + 1) + E + k, token < N, 0)
        probability = tl.load(Probs + token * E + expert, token < N, 0)
        tl.store(Ids + row, token, token < N)
        tl.store(Packed + row, probability, token < N)


@triton.jit
def expand_received(Local, Peer, Ids, Output, R, H: tl.constexpr, Split0, Rank: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, col = offset // H, offset % H
    token = tl.load(Ids + row, row < R, 0)
    is_local = (row < Split0) if Rank == 0 else (row >= Split0)
    local = tl.load(Local + token * H + col, (row < R) & is_local, 0)
    peer = tl.load(Peer + token * H + col, (row < R) & ~is_local, 0)
    tl.store(Output + offset, tl.where(is_local, local, peer), row < R)


@triton.jit
def weighted_product(Gate, Input, Weights, Output, N, D: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, col = offset // D, offset % D
    mask = row < N
    gate = tl.load(Gate + offset, mask, 0).to(tl.float32)
    linear = tl.load(Input + row * (2 * D) + D + col, mask, 0).to(tl.float32)
    # Preserve the BF16 rounding boundary of silu(gate) * linear before
    # multiplication by FP64 router weights. Never downcast the weights.
    activation = (gate * linear).to(tl.bfloat16).to(tl.float64)
    weights = tl.load(Weights + row, mask, 0)
    tl.store(Output + offset, activation * weights, mask)
