# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Conservative expert activation experiment; GEMMs and backward are unchanged.

Keep native PyTorch SiLU, then fuse BF16 product/rounding, FP64 weighting and
BF16 output into one Triton launch. The production reference may itself be
torch.compile-fused, so this is a candidate, not an assumed speedup.
"""

from typing import Any

import torch


class _WeightedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        import triton

        from .moe_chunk_kernels import weighted_product

        n, width = inputs.shape
        d = width // 2
        ctx.save_for_backward(inputs, weights)
        # Keep the original SiLU implementation instead of substituting an
        # approximate exp/sigmoid. The GPU regression checks the compiled
        # production reference as well, not just an eager mathematical formula.
        gate = torch.nn.functional.silu(inputs[:, :d])
        output = inputs.new_empty((n, d))
        if n:
            weighted_product[(triton.cdiv(n * d, 1024),)](
                gate, inputs, weights, output, n, d, 1024, enable_fp_fusion=False
            )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu_back

        inputs, weights = ctx.saved_tensors
        return weighted_swiglu_back(grad_output, inputs, weights)


def weighted_bias_swiglu(
    inputs: torch.Tensor,
    bias: torch.Tensor | None,
    weights: torch.Tensor,
    fp8_input_store: bool = False,
    clamp_value: float | None = None,
) -> torch.Tensor:
    if bias is not None or fp8_input_store or clamp_value is not None:
        raise ValueError("Expert activation experiment requires no bias, FP8 storage or clamp")
    if inputs.dtype != torch.bfloat16 or weights.dtype != torch.float64:
        raise ValueError("Expert activation experiment requires BF16 inputs and FP64 weights")
    if inputs.ndim not in (2, 3) or inputs.shape[-1] % 2 or not inputs.is_contiguous():
        raise ValueError("Expert activation experiment requires contiguous, even-width inputs")
    if not weights.is_contiguous() or weights.shape != (inputs.numel() // inputs.shape[-1], 1):
        raise ValueError("Expected one contiguous FP64 routing weight per expert row")
    shape = inputs.shape
    result = _WeightedSwiGLU.apply(inputs.view(-1, shape[-1]), weights)
    return result.view(*shape[:-1], shape[-1] // 2)
