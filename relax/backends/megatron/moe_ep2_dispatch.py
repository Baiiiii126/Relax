# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""EP2 input broadcast with expert expansion at the receiver.

The two EP ranks must be the same ranks as the sequence-parallel TP2 group:
each owns equally many rows of the same padded chunk, including short tails.
All source rows are sent once to the peer (including rows with no remote route).
This is a dense EP2 specialization, not a sparse arbitrary-EP dispatcher.

Unlike a unique-token autograd gather, backward deliberately sends EXPANDED
gradients and uses the original TE permutation backward. This preserves its
expert accumulation order. Combine and router-probability precision are unchanged.
"""

from typing import Any

import torch
import torch.distributed as dist


def _exchange(tensor: torch.Tensor, inputs: tuple[int, ...], outputs: tuple[int, ...], group: Any) -> torch.Tensor:
    result = tensor.new_empty((sum(outputs), *tensor.shape[1:]))
    dist.all_to_all_single(result, tensor.contiguous(), list(outputs), list(inputs), group=group)
    return result


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        probs: torch.Tensor,
        row_map: torch.Tensor,
        inputs: tuple,
        outputs: tuple,
        group: Any,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import triton

        from .moe_chunk_kernels import expand_received, pack_ids_probs

        n, h = hidden.shape
        experts = probs.shape[1]
        rank = dist.get_rank(group)
        ids = torch.empty(n * topk, dtype=torch.int32, device=hidden.device)
        packed_probs = probs.new_empty(n * topk)
        pack_ids_probs[(triton.cdiv(n, 128),)](row_map, probs, ids, packed_probs, n, experts, topk, 128)
        # Equal sequence shards are guaranteed by the checked TP2/SP topology.
        # No self-send, no expanded hidden-state send buffer, no CPU shape copy.
        peer_splits = (0, n) if rank == 0 else (n, 0)
        peer = _exchange(hidden, peer_splits, peer_splits, group)
        received_ids = _exchange(ids, inputs, outputs, group)
        result = hidden.new_empty((sum(outputs), h))
        if result.numel():
            expand_received[(triton.cdiv(result.numel(), 1024),)](
                hidden, peer, received_ids, result, sum(outputs), h, outputs[0], rank, 1024
            )
        # These temporaries are never saved for backward; stream ordering keeps
        # their storage alive until the gather completes.
        del peer, received_ids, ids
        result_probs = _exchange(packed_probs, inputs, outputs, group)
        ctx.save_for_backward(row_map)
        ctx.inputs, ctx.outputs, ctx.group = inputs, outputs, group
        ctx.n, ctx.h, ctx.experts = n, h, experts
        ctx.probs_dtype = probs.dtype
        return result, result_probs

    @staticmethod
    def backward(ctx: Any, grad_hidden: torch.Tensor, grad_probs: torch.Tensor) -> tuple:
        from transformer_engine.pytorch.triton.permutation import unpermute_with_mask_map

        # Match baseline: FP64 probability gradients and expanded BF16 hidden
        # gradients use their original split arrays. Never reduce by rank first.
        if grad_probs is None:
            grad_probs = torch.zeros(sum(ctx.outputs), device=grad_hidden.device, dtype=ctx.probs_dtype)
        grad_probs = _exchange(grad_probs, ctx.outputs, ctx.inputs, ctx.group)
        grad_hidden = _exchange(grad_hidden, ctx.outputs, ctx.inputs, ctx.group)
        (row_map,) = ctx.saved_tensors
        dx, dp = unpermute_with_mask_map(grad_hidden, row_map, None, grad_probs, None, ctx.n, ctx.experts, ctx.h)
        return dx, dp, None, None, None, None, None


def dispatch_preprocess(
    self: Any, hidden: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformer_engine.pytorch.triton.permutation import make_row_id_map

    if self.shared_experts is not None:
        raise ValueError("EP2 dedup does not support shared-expert communication overlap")
    if hidden.dtype != torch.bfloat16 or probs.dtype != torch.float64:
        raise ValueError("EP2 dedup requires BF16 hidden states and FP64 router probabilities")
    if routing_map.dtype != torch.bool or not probs.is_contiguous():
        raise ValueError("EP2 dedup requires bool routing and contiguous probabilities")
    self.hidden_shape = hidden.shape
    hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    if hidden.shape[0] == 0:
        raise ValueError("EP2 dedup requires nonempty, equally padded sequence shards")
    if probs.shape != (hidden.shape[0], self.num_experts) or routing_map.shape != probs.shape:
        raise ValueError("EP2 dedup requires matching token-by-expert probabilities and routing map")
    self.hidden_shape_before_permute = hidden.shape
    self.probs, self.routing_map = probs, routing_map
    self.tokens_per_expert = self.preprocess(routing_map)
    self.tokens_per_expert = self._maybe_dtoh_and_synchronize("before_permutation_1", self.tokens_per_expert)
    self.reversed_local_input_permutation_mapping = make_row_id_map(routing_map, hidden.shape[0], self.num_experts)
    return hidden, probs


def token_dispatch(self: Any, hidden: torch.Tensor, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.tokens_per_expert = self._maybe_dtoh_and_synchronize("before_ep_alltoall", self.tokens_per_expert)
    # Existing metadata has already reached the CPU at the original sync point.
    # Copy the tiny CPU arrays into invocation-owned tuples for delayed backward.
    inputs = tuple(int(v) for v in self.input_splits)
    outputs = tuple(int(v) for v in self.output_splits)
    if sum(inputs) != hidden.shape[0] * self.config.moe_router_topk:
        raise ValueError("EP2 dedup requires exactly top-k routed assignments for every token")
    return _Dispatch.apply(
        hidden,
        probs,
        self.reversed_local_input_permutation_mapping,
        inputs,
        outputs,
        self.ep_group,
        self.config.moe_router_topk,
    )
