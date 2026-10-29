# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Install reversible, opt-in MoE experiments before model construction."""

from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
MODES = ("off", "dedup", "expert", "both")
_originals: dict[str, Any] = {}


def validate_mode(mode: str, metadata_mode: str = "off") -> None:
    if mode not in MODES:
        raise ValueError(f"RELAX_MOE_CHUNK_MODE must be one of {MODES}")
    if mode != "off" and metadata_mode != "off":
        raise ValueError("MoE chunk experiments require RELAX_MOE_METADATA_MODE=off to isolate their effects")


def validate_dispatch_config(config: Any) -> None:
    required = {
        "tensor_model_parallel_size": 2,
        "sequence_parallel": True,
        "expert_model_parallel_size": 2,
        "expert_tensor_parallel_size": 1,
        "num_moe_experts": 256,
        "moe_router_topk": 8,
        "moe_permute_fusion": True,
        "moe_router_dtype": "fp64",
        "cuda_graph_impl": "none",
    }
    for key, expected in required.items():
        if getattr(config, key, None) != expected:
            raise ValueError(f"EP2 dedup requires {key}={expected!r}")
    for key in (
        "moe_pad_expert_input_to_capacity",
        "moe_router_padding_for_quantization",
        "moe_shared_expert_overlap",
        "fp8",
        "fp4",
    ):
        if getattr(config, key, False):
            raise ValueError(f"EP2 dedup does not support {key}")
    if getattr(config, "moe_expert_capacity_factor", None) is not None:
        raise ValueError("EP2 dedup requires dropless routing")


def validate_expert_config(config: Any) -> None:
    import torch.nn.functional as functional

    if not getattr(config, "bias_activation_fusion", False) or getattr(config, "use_te_activation_func", False):
        raise ValueError("Expert experiment requires weighted SwiGLU bias_activation_fusion path")
    if not getattr(config, "gated_linear_unit", False) or config.activation_func is not functional.silu:
        raise ValueError("Expert experiment requires SwiGLU")
    if getattr(config, "moe_mlp_glu_interleave_size", None) is not None:
        raise ValueError("Expert experiment does not support interleaved GLU")
    if (
        getattr(config, "activation_func_clamp_value", None) is not None
        or getattr(config, "glu_linear_offset", 0) != 0
    ):
        raise ValueError("Expert experiment requires unclamped SwiGLU without a linear offset")
    for key in (
        "fp8",
        "fp4",
        "activation_func_fp8_input_store",
        "moe_apply_probs_on_input",
        "use_transformer_engine_op_fuser",
        "add_bias_linear",
        "fine_grained_activation_offloading",
    ):
        if getattr(config, key, False):
            raise ValueError(f"Expert experiment does not support {key}")
    if getattr(config, "cuda_graph_impl", "none") != "none":
        raise ValueError("Expert experiment requires CUDA Graph disabled")


def install(mode: str, metadata_mode: str = "off") -> None:
    """Patch only the selected path.

    No imports of GPU dependencies when disabled.
    """
    validate_mode(mode, metadata_mode)
    if mode == "off" and not _originals:
        return
    from importlib.metadata import version

    from megatron.core.transformer.moe import experts, token_dispatcher

    dispatcher = token_dispatcher.MoEAlltoAllTokenDispatcher
    grouped = experts.TEGroupedMLP
    if not _originals:
        _originals.update(
            dispatch_init=dispatcher.__init__,
            preprocess=dispatcher.dispatch_preprocess,
            dispatch=dispatcher.token_dispatch,
            expert_init=grouped.__init__,
            activation=experts.weighted_bias_swiglu_impl,
        )
    dispatcher.__init__ = _originals["dispatch_init"]
    dispatcher.dispatch_preprocess = _originals["preprocess"]
    dispatcher.token_dispatch = _originals["dispatch"]
    grouped.__init__ = _originals["expert_init"]
    experts.weighted_bias_swiglu_impl = _originals["activation"]
    if mode == "off":
        return
    # We call TE's private permutation backward to retain the original reduction
    # order. Pin the tested interface instead of silently using another layout.
    if version("transformer-engine") != "2.14.1":
        raise RuntimeError("MoE chunk experiments are pinned to Transformer Engine 2.14.1")
    if mode in ("dedup", "both"):
        from .moe_ep2_dispatch import dispatch_preprocess, token_dispatch

        def init_dispatch(
            self: Any, num_local_experts: int, local_expert_indices: list[int], config: Any, pg_collection: Any = None
        ) -> None:
            import torch.distributed as dist
            from megatron.core import parallel_state

            validate_dispatch_config(config)
            _originals["dispatch_init"](self, num_local_experts, local_expert_indices, config, pg_collection)
            tp_group = getattr(pg_collection, "tp", None)
            if tp_group is None:
                tp_group = parallel_state.get_tensor_model_parallel_group()
            if (
                self.tp_size != 1
                or self.ep_size != 2
                or dist.get_process_group_ranks(tp_group) != dist.get_process_group_ranks(self.ep_group)
            ):
                raise ValueError("EP2 dedup requires EP ranks identical, in order, to the sequence-parallel TP2 group")
            if num_local_experts != 128 or local_expert_indices != list(
                range(self.ep_rank * 128, (self.ep_rank + 1) * 128)
            ):
                raise ValueError("EP2 dedup requires 128 contiguous experts per rank")
            logger.info("MoE chunk optimization installed: dedup (BF16 inputs, FP64 probs, original backward)")

        dispatcher.__init__ = init_dispatch
        dispatcher.dispatch_preprocess = dispatch_preprocess
        dispatcher.token_dispatch = token_dispatch
    if mode in ("expert", "both"):
        from .moe_expert_activation import weighted_bias_swiglu

        def init_expert(self: Any, *args: Any, **kwargs: Any) -> None:
            _originals["expert_init"](self, *args, **kwargs)
            validate_expert_config(self.config)
            logger.info(
                "MoE chunk optimization installed: expert (native SiLU, fused FP64 weighting, original backward)"
            )

        grouped.__init__ = init_expert
        experts.weighted_bias_swiglu_impl = weighted_bias_swiglu
