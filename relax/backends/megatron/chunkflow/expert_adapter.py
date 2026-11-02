# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in BF16 expert adapter preserving Megatron's parameters and DDP hooks.

The existing dispatcher still owns communication and its host split sizes.
Optional EP2 local-first dispatch overlaps communication with the local FC1.
"""

from __future__ import annotations

import os
from functools import lru_cache
from types import MethodType
from typing import Any

from relax.utils.logging_utils import get_logger

from . import expert_tiles


logger = get_logger(__name__)
BACKENDS = {"tiles128": 128, "tiles256": 256, "mixed": 0}


class _DeviceCounts:
    """Bypass the validated bias-free TEGroupedMLP's legacy tolist call.

    Only the two replaced linears consume this object. The original activation
    and its probability dtype, clamp and recomputation behavior are retained.
    """

    def __init__(self, counts: Any) -> None:
        self.dispatch = getattr(counts, "_relax_dispatch", None)
        self.offsets = self.dispatch.offsets if self.dispatch is not None else expert_tiles.make_offsets(counts)

    def tolist(self) -> _DeviceCounts:
        return self


class _Pointers:
    def __init__(self) -> None:
        self.key: tuple = ()
        self.table: Any = None

    def get(self, tensors: tuple) -> Any:
        import torch

        key = (tensors[0].device, *(tensor.data_ptr() for tensor in tensors))
        if key != self.key:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Expert pointer tables must be warmed up before CUDA Graph capture")
            self.table = torch.tensor(key[1:], dtype=torch.int64, device=tensors[0].device)
            self.key = key
        return self.table


class _Linear:
    def __init__(self, module: Any, block_m: int, source_first: bool = False) -> None:
        self.module = module
        self.block_m = block_m
        self.source_first = source_first
        self.weights = _Pointers()
        self.gradients = _Pointers()

    def parameters(self) -> tuple:
        import torch

        weights = tuple(getattr(self.module, f"weight{i}") for i in range(self.module.num_gemms))
        for weight in weights:
            expert_tiles._check_tensor(weight, 2, "expert weight")
            if not isinstance(weight, torch.nn.Parameter) or hasattr(weight, "__fsdp_param__"):
                raise ValueError("Expert tiles require original unsharded Parameters")
            if weight.shape != weights[0].shape or weight.device != weights[0].device:
                raise ValueError("Expert weights must have a common shape and device")
            if weight.requires_grad != weights[0].requires_grad:
                raise ValueError("Mixed frozen/trainable expert weights are unsupported")
        return weights

    def matmul(
        self,
        x: Any,
        table: Any,
        offsets: Any,
        shape: tuple,
        transpose: bool,
        output: Any = None,
        dispatch: Any = None,
        source: int = -1,
    ) -> Any:
        import torch

        from . import _expert_tile_kernels as kernels

        expert_tiles._check_tensor(x, 2, "expert input")
        outputs, inputs = shape
        if transpose:
            inputs, outputs = outputs, inputs
        if x.shape[1] != inputs or x.device != table.device:
            raise ValueError("Expert input shape/device does not match weights")
        y = output if output is not None else torch.empty((x.shape[0], outputs), dtype=x.dtype, device=x.device)
        if x.shape[0]:
            workers = expert_tiles._workers(x.device.index)
            kernels.linear[(workers,)](
                x,
                table,
                y,
                offsets,
                x.shape[0],
                self.module.num_gemms,
                outputs,
                inputs,
                0,
                1 if transpose else inputs,
                outputs if transpose else 1,
                self.block_m,
                64,
                32,
                workers,
                True,
                None if dispatch is None else dispatch.counts,
                source,
                num_warps=4,
                num_stages=3,
            )
        return y

    def wgrad(self, x: Any, dy: Any, offsets: Any, weights: tuple, first: bool | None) -> tuple:
        import torch
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        from . import _expert_tile_kernels as kernels

        gradients = tuple(weight.main_grad for weight in weights)
        for weight, grad in zip(weights, gradients):
            if grad.dtype != torch.float32 or grad.device != x.device or grad.shape != weight.shape:
                raise ValueError("Expert main_grad must retain the original CUDA FP32 shape/device")
            if not grad.is_contiguous():
                raise ValueError("Expert main_grad must be contiguous")
        overwrite = {bool(getattr(w, "overwrite_main_grad", False)) for w in weights}
        if len(overwrite) != 1:
            raise ValueError("Expert overwrite_main_grad flags must agree within a group")
        accumulate = first is not True and not next(iter(overwrite))
        table = self.gradients.get(gradients)
        n, k = dy.shape[1], x.shape[1]
        kernels.weight_grad[(((n + 63) // 64) * ((k + 63) // 64), len(weights))](
            x,
            dy,
            table,
            offsets,
            x.shape[0],
            n,
            k,
            64,
            64,
            32,
            accumulate,
            True,
            num_warps=4,
            num_stages=3,
        )
        result = []
        for weight in weights:
            if hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                result.append(
                    get_dummy_wgrad(list(weight.shape), weight.dtype, zero=getattr(weight, "zero_out_wgrad", False))
                )
            else:
                result.append(None)
        return tuple(result)

    def forward(self, x: Any, counts: _DeviceCounts) -> tuple:
        if not isinstance(counts, _DeviceCounts):
            raise TypeError("Expert tiles require GPU counts from the adapted dispatcher")
        weights = self.parameters()
        first = None if self.module.disable_parameter_transpose_cache else self.module.is_first_microbatch
        result = _linear_autograd().apply(x, counts, self, first, *weights)
        self.module.is_first_microbatch = False
        return result, None


@lru_cache(maxsize=1)
def _linear_autograd() -> type:
    import torch
    from torch.autograd.function import once_differentiable

    class Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, counts: _DeviceCounts, state: _Linear, first: bool | None, *weights: Any) -> Any:
            offsets = counts.offsets
            table = state.weights.get(weights)
            ctx.save_for_backward(x, offsets, table, *weights)
            ctx.original_weights = weights
            ctx.state, ctx.first = state, first
            transport = counts.dispatch if state.source_first else None
            if transport is None:
                return state.matmul(x, table, offsets, weights[0].shape, False)
            output = state.matmul(
                x, table, offsets, weights[0].shape, False, dispatch=transport, source=transport.rank
            )
            torch.cuda.current_stream(x.device).wait_event(transport.ready)
            return state.matmul(
                x,
                table,
                offsets,
                weights[0].shape,
                False,
                output=output,
                dispatch=transport,
                source=1 - transport.rank,
            )

        @staticmethod
        @once_differentiable
        def backward(ctx: Any, dy: Any) -> tuple:
            x, offsets, table, *weights = ctx.saved_tensors
            dy = dy.contiguous()
            dx = ctx.state.matmul(dy, table, offsets, weights[0].shape, True) if ctx.needs_input_grad[0] else None
            gradients = (None,) * len(weights)
            if ctx.needs_input_grad[4]:
                gradients = ctx.state.wgrad(x, dy, offsets, ctx.original_weights, ctx.first)
            return dx, None, None, None, *gradients

    return Linear


def _validate(experts: Any, dispatcher: Any, overlap: bool) -> None:
    config = experts.config
    forbidden = (
        "fp8",
        "fp4",
        "add_bias_linear",
        "use_transformer_engine_op_fuser",
        "fine_grained_activation_offloading",
        "delay_wgrad_compute",
        "overlap_dispatch_backward_with_experts_wgrad",
        "moe_single_grouped_weight",
        "moe_router_padding_for_quantization",
        "quant_recipe",
        "moe_paged_stash",
    )
    enabled = [key for key in forbidden if getattr(config, key, False)]
    if enabled:
        raise ValueError(f"Expert tiles do not support these settings: {enabled}")
    if not getattr(config, "gradient_accumulation_fusion", False):
        raise ValueError("Expert tiles require FP32 main_grad accumulation fusion")
    if getattr(config, "expert_tensor_parallel_size", 1) != 1:
        raise ValueError("Expert tiles currently require expert tensor parallel size 1")
    if type(dispatcher).__name__ != "MoEAlltoAllTokenDispatcher" or getattr(dispatcher, "drop_and_pad", False):
        raise ValueError("Expert tiles require the dropless alltoall dispatcher")
    if overlap:
        if dispatcher.ep_size != 2 or dispatcher.tp_size != 1 or dispatcher.num_local_experts < 2:
            raise ValueError("Local-first dispatch requires EP2, ETP1 and multiple local experts")
        if not config.moe_permute_fusion or getattr(dispatcher, "shared_experts", None) is not None:
            raise ValueError("Local-first dispatch requires permutation fusion and no shared-expert overlap")
        if config.moe_apply_probs_on_input or getattr(config, "log_moe_overload_factor", False):
            raise ValueError("Local-first dispatch requires FC1 to be the first reader of dispatched activations")
        if getattr(config, "moe_router_dtype", None) != "fp64":
            raise ValueError("Local-first dispatch currently requires FP64 router probabilities")
    if os.environ.get("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0") != "0":
        raise ValueError("Expert tiles do not support fake weight quantization")
    for linear in (experts.linear_fc1, experts.linear_fc2):
        if getattr(linear, "te_quant_params", None) is not None or getattr(linear, "use_bias", False):
            raise ValueError("Expert tiles require unquantized, bias-free grouped linears")
        if not getattr(linear, "fuse_wgrad_accumulation", False):
            raise ValueError("Expert linears must use FP32 fused gradient accumulation")
        if not hasattr(linear, "weight0"):
            raise ValueError("Expert tiles require separate weight0..weightN Parameters")


def _install_layer(experts: Any, dispatcher: Any, block_m: int, overlap: bool) -> None:
    original_preprocess = dispatcher.preprocess
    original_postprocess = dispatcher.dispatch_postprocess
    original_forward = experts.forward

    def preprocess(_self: Any, *args: Any, **kwargs: Any) -> Any:
        counts = original_preprocess(*args, **kwargs)
        if not counts.is_cuda:
            raise ValueError("The dispatcher must produce CUDA expert counts")
        _self._relax_device_expert_counts = counts
        return counts

    def postprocess(_self: Any, *args: Any, **kwargs: Any) -> tuple:
        if overlap:
            x, probabilities = args
            _self.tokens_per_expert = _self._maybe_dtoh_and_synchronize(
                "before_permutation_2", _self.tokens_per_expert
            )
            _self._maybe_dtoh_and_synchronize("before_finish", _self.tokens_per_expert)
            _self.tokens_per_expert = None
        else:
            x, _counts, probabilities = original_postprocess(*args, **kwargs)
        counts = _self._relax_device_expert_counts
        del _self._relax_device_expert_counts
        return x, counts, probabilities

    def forward(_self: Any, x: Any, counts: Any, probabilities: Any, **kwargs: Any) -> Any:
        return original_forward(x, _DeviceCounts(counts), probabilities, **kwargs)

    dispatcher.preprocess = MethodType(preprocess, dispatcher)
    dispatcher.dispatch_postprocess = MethodType(postprocess, dispatcher)
    experts.forward = MethodType(forward, experts)
    if overlap:
        from .expert_transport import Dispatch, dispatch

        def token_dispatch(_self: Any, x: Any, probabilities: Any) -> tuple:
            _self.tokens_per_expert = _self._maybe_dtoh_and_synchronize("before_ep_alltoall", _self.tokens_per_expert)
            # These are already CPU numpy split arrays produced by the original dispatcher.
            if any(getattr(splits, "is_cuda", False) for splits in (_self.input_splits, _self.output_splits)):
                raise RuntimeError("The dispatcher must finish its host split discovery before local-first dispatch")
            state = Dispatch(
                _self.ep_group,
                tuple(int(n) for n in _self.input_splits),
                tuple(int(n) for n in _self.output_splits),
                _self.num_global_tokens_per_local_expert,
            )
            _self._relax_device_expert_counts._relax_dispatch = state
            return dispatch(x, probabilities, state)

        dispatcher.token_dispatch = MethodType(token_dispatch, dispatcher)
    for index, linear in enumerate((experts.linear_fc1, experts.linear_fc2)):
        state = _Linear(linear, block_m, source_first=overlap and index == 0)
        linear.forward = state.forward
    experts._relax_expert_tiles_installed = True


def install(model: Any) -> int:
    backend = os.environ.get("RELAX_MOE_EXPERT_BACKEND", "te")
    overlap_flag = os.environ.get("RELAX_MOE_DISPATCH_OVERLAP", "0")
    if overlap_flag not in ("0", "1") or (overlap_flag == "1" and backend == "te"):
        raise ValueError("RELAX_MOE_DISPATCH_OVERLAP must be 0/1; overlap requires an expert tiles backend")
    overlap = overlap_flag == "1"
    if backend == "te":
        return 0
    if backend not in BACKENDS:
        raise ValueError(f"Unknown RELAX_MOE_EXPERT_BACKEND: {backend}")
    layers = []
    installed = 0
    for module in model.modules():
        experts, dispatcher = getattr(module, "experts", None), getattr(module, "token_dispatcher", None)
        if experts is None or dispatcher is None:
            continue
        if getattr(experts, "_relax_expert_tiles_installed", False):
            installed += 1
            continue
        if type(experts).__name__ != "TEGroupedMLP":
            raise ValueError("Expert tiles currently support TEGroupedMLP only")
        _validate(experts, dispatcher, overlap)
        layers.append((experts, dispatcher))
    if not layers and not installed:
        raise ValueError("Expert backend requested but no eligible MoE layers were found")
    for experts, dispatcher in layers:
        _install_layer(experts, dispatcher, BACKENDS[backend], overlap)
    logger.info("Installed expert backend=%s on %d MoE layers; local-first dispatch=%s", backend, len(layers), overlap)
    return len(layers)
