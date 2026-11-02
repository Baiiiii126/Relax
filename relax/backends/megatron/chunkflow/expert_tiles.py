# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Experimental GPU-counts expert computation, without a transport adapter.

Inputs must already be packed by expert and fully ready on the current stream.
This module does not implement tile-ready communication or install model hooks.
Offsets are exclusive row boundaries: 0 <= offsets[e] <= offsets[e+1] <= rows,
with first/last values 0/rows. Their values stay on the GPU. Callers own this
invariant and must keep saved inputs/offsets alive and unmodified until
backward.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    import torch


def make_offsets(counts: torch.Tensor) -> torch.Tensor:
    """Prefix-sum nonnegative CUDA expert counts without a host scalar read."""
    import torch

    if not counts.is_cuda or counts.ndim != 1 or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("counts must be a one-dimensional CUDA int32/int64 tensor")
    if counts.numel() == 0 or not counts.is_contiguous():
        raise ValueError("counts must be nonempty and contiguous")
    offsets = torch.empty(counts.numel() + 1, dtype=torch.int64, device=counts.device)
    offsets[:1].zero_()
    torch.cumsum(counts, dim=0, dtype=torch.int64, out=offsets[1:])
    return offsets


def _check_tensor(tensor: torch.Tensor, ndim: int, name: str) -> None:
    import torch

    if not tensor.is_cuda or tensor.ndim != ndim or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a {ndim}-dimensional CUDA BF16 tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_offsets(offsets: torch.Tensor, experts: int, device: torch.device) -> None:
    import torch

    if (
        offsets.device != device
        or offsets.dtype not in (torch.int32, torch.int64)
        or offsets.shape != (experts + 1,)
        or not offsets.is_contiguous()
    ):
        raise ValueError("offsets must be contiguous CUDA int32/int64 [experts + 1] on the input device")


@lru_cache(maxsize=None)
def _workers(device_index: int) -> int:
    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _linear(
    x: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor, block_m: int, transpose: bool
) -> torch.Tensor:
    import torch

    from . import _expert_tile_kernels as kernels

    _check_tensor(x, 2, "x")
    _check_tensor(weights, 3, "weights")
    experts, outputs, inputs = weights.shape
    if transpose:
        inputs, outputs = outputs, inputs
    if experts == 0 or inputs == 0 or outputs == 0 or x.shape[1] != inputs or weights.device != x.device:
        raise ValueError("weights [experts, out, in] do not match the input")
    if block_m not in (32, 64, 128):
        raise ValueError("block_m must be 32, 64 or 128")
    _check_offsets(offsets, experts, x.device)
    y = torch.empty((x.shape[0], outputs), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        workers = min(
            _workers(x.device.index), ((x.shape[0] + block_m - 1) // block_m + experts) * ((outputs + 63) // 64)
        )
        kernels.linear[(workers,)](
            x,
            weights,
            y,
            offsets,
            x.shape[0],
            experts,
            outputs,
            inputs,
            weights.stride(0),
            weights.stride(2 if transpose else 1),
            weights.stride(1 if transpose else 2),
            block_m,
            64,
            32,
            workers,
            num_warps=4,
            num_stages=3,
        )
    return y


def forward(x: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor, *, block_m: int = 64) -> torch.Tensor:
    """Compute BF16 Y[e] = X[e] @ W[e].T with FP32 accumulation."""
    return _linear(x, weights, offsets, block_m, False)


def backward_input(
    dy: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor, *, block_m: int = 64
) -> torch.Tensor:
    """Compute BF16 dX[e] = dY[e] @ W[e] with FP32 accumulation."""
    return _linear(dy, weights, offsets, block_m, True)


def backward_weight(
    x: torch.Tensor,
    dy: torch.Tensor,
    offsets: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """Write FP32 dW, optionally accumulating directly into caller-owned
    storage."""
    import torch

    from . import _expert_tile_kernels as kernels

    _check_tensor(x, 2, "x")
    _check_tensor(dy, 2, "dy")
    experts = offsets.numel() - 1
    _check_offsets(offsets, experts, x.device)
    if experts <= 0 or dy.device != x.device or dy.shape[0] != x.shape[0] or min(x.shape[1], dy.shape[1]) <= 0:
        raise ValueError("x, dy and offsets have incompatible shapes/devices")
    shape = (experts, dy.shape[1], x.shape[1])
    if out is None:
        if accumulate:
            raise ValueError("accumulate=True requires an initialized out buffer")
        out = torch.empty(shape, dtype=torch.float32, device=x.device)
    if out.shape != shape or out.dtype != torch.float32 or out.device != x.device or not out.is_contiguous():
        raise ValueError("out must be contiguous FP32 [experts, out, in] on the input device")
    if out.requires_grad:
        raise ValueError("the FP32 gradient destination must not require gradients")
    kernels.weight_grad[((dy.shape[1] + 63) // 64 * ((x.shape[1] + 63) // 64), experts)](
        x,
        dy,
        out,
        offsets,
        x.shape[0],
        dy.shape[1],
        x.shape[1],
        64,
        64,
        32,
        accumulate,
        num_warps=4,
        num_stages=3,
    )
    return out


@lru_cache(maxsize=1)
def _autograd() -> type:
    import torch
    from torch.autograd.function import once_differentiable

    class ExpertLinear(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            x: torch.Tensor,
            weights: torch.Tensor,
            offsets: torch.Tensor,
            main_grad: torch.Tensor | None,
            block_m: int,
        ) -> torch.Tensor:
            ctx.save_for_backward(x, weights, offsets)
            ctx.main_grad = main_grad
            ctx.block_m = block_m
            return forward(x, weights, offsets, block_m=block_m)

        @staticmethod
        @once_differentiable
        def backward(ctx: Any, dy: torch.Tensor) -> tuple:
            x, weights, offsets = ctx.saved_tensors
            dy = dy.contiguous()
            dx = backward_input(dy, weights, offsets, block_m=ctx.block_m) if ctx.needs_input_grad[0] else None
            dw = None
            if ctx.needs_input_grad[1]:
                grad = backward_weight(x, dy, offsets, out=ctx.main_grad, accumulate=ctx.main_grad is not None)
                if ctx.main_grad is None:
                    dw = grad.to(weights.dtype)
            return dx, dw, None, None, None

    return ExpertLinear


def grouped_linear(
    x: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    *,
    main_grad: torch.Tensor | None = None,
    block_m: int = 64,
) -> torch.Tensor:
    """Standalone first-order autograd adapter, NOT an MCore DDP adapter.

    With main_grad, weights.grad stays None and FP32 dW accumulates in
    main_grad. The caller must initialize/zero that buffer. MCore's gradient-
    ready hooks and grad_added_to_main_grad protocol are deliberately not
    installed here.
    """
    return _autograd().apply(x, weights, offsets, main_grad, block_m)


def expert_mlp(
    x: torch.Tensor,
    weights1: torch.Tensor,
    weights2: torch.Tensor,
    offsets: torch.Tensor,
    probs: torch.Tensor,
    activation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    main_grads: tuple[torch.Tensor, torch.Tensor] | None = None,
    block_m: int = 64,
) -> torch.Tensor:
    """Compose FC1 -> the caller's original weighted activation -> FC2."""
    if weights1.shape[0] != weights2.shape[0] or weights1.shape[1] != 2 * weights2.shape[2]:
        raise ValueError("SwiGLU weights must have shapes [E, 2F, H] and [E, H, F]")
    if weights2.shape[1] != x.shape[1] or probs.shape != (x.shape[0],) or probs.device != x.device:
        raise ValueError("input, output and probability shapes/devices do not match")
    grad1, grad2 = (None, None) if main_grads is None else main_grads
    h = grouped_linear(x, weights1, offsets, main_grad=grad1, block_m=block_m)
    h = activation(h, probs)
    return grouped_linear(h, weights2, offsets, main_grad=grad2, block_m=block_m)
