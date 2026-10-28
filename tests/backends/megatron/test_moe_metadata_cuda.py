# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-GPU dispatcher regression, run with torchrun -m pytest on this file.

Exercises real fused permutation, NCCL and their backward with two outstanding
forwards. The synthetic experts are deliberately small; production loss and
allocator peaks are checked separately with the Qwen comparison script.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires two NVIDIA GPUs, NCCL, Megatron and TE")
def test_metadata_modes_preserve_dispatch_outputs_and_gradients(monkeypatch):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Launch with torchrun --standalone --nproc_per_node=2 -m pytest")
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    assert getattr(MoEAlltoAllTokenDispatcher, "_relax_moe_metadata_patch_version", None) == 1
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world = dist.group.WORLD
    rank = dist.get_rank(world)
    singletons = [dist.new_group(ranks=[i], backend="nccl") for i in range(2)]
    groups = SimpleNamespace(ep=world, expt_tp=singletons[rank], tp_ep=world)
    config = SimpleNamespace(
        num_moe_experts=256,
        moe_permute_fusion=True,
        moe_pad_expert_input_to_capacity=False,
        moe_expert_capacity_factor=None,
        moe_router_padding_for_quantization=False,
        moe_router_topk=8,
        cuda_graph_impl="none",
        cuda_graph_modules=[],
    )

    def execute(mode):
        monkeypatch.setenv("RELAX_MOE_METADATA_MODE", mode)
        dispatcher = MoEAlltoAllTokenDispatcher(128, list(range(rank * 128, (rank + 1) * 128)), config, groups)
        torch.manual_seed(918 + rank)
        scale = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
        xs, logits_list, outputs, counts = [], [], [], []
        for n in (8192, 97):
            x = torch.randn(n, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            logits = torch.randn(n, 256, device="cuda", dtype=torch.float64, requires_grad=True)
            indices = logits.topk(8, dim=-1).indices
            routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, indices, True)
            probs = logits.softmax(dim=-1).float() * routing_map
            hidden, p = dispatcher.dispatch_preprocess(x, routing_map, probs)
            hidden, p = dispatcher.token_dispatch(hidden, p)
            hidden, tokens, p = dispatcher.dispatch_postprocess(hidden, p)
            counts.append(tokens.clone())
            chunks = hidden.split(tokens.tolist())
            hidden = torch.cat([torch.nn.functional.silu(t.float()) * scale[i] for i, t in enumerate(chunks)])
            hidden = (hidden * p.unsqueeze(-1)).to(x.dtype)
            hidden = dispatcher.combine_preprocess(hidden)
            hidden = dispatcher.token_combine(hidden)
            outputs.append(dispatcher.combine_postprocess(hidden))
            xs.append(x)
            logits_list.append(logits)
        # Both forward contexts must survive; their saved CPU splits differ.
        losses = [o.float().square().mean() for o in outputs]
        for loss in reversed(losses):
            loss.backward()
        torch.cuda.synchronize()
        return [
            t.detach().cpu()
            for t in (*counts, *outputs, *losses, *(x.grad for x in xs), *(l.grad for l in logits_list), scale.grad)
        ]

    try:
        reference = execute("off")
        for mode in ("a", "b"):
            actual = execute(mode)
            for expected, result in zip(reference, actual):
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
