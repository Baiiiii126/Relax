# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Eight-GPU gate: four EP2 pairs, exact outputs/gradients and allocator peaks.

Uses real NCCL, TE permutations and the production compiled weighted SwiGLU.
This is operator correctness coverage, not a replacement for TP2/PP4 training.
"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires eight NVIDIA GPUs, NCCL, TE 2.14.1 and Megatron")
def test_moe_chunk_modes_exact_forward_backward_and_memory(monkeypatch):
    memory_policy = os.environ.get("RELAX_MOE_MEMORY_POLICY", "strict")
    if memory_policy not in ("strict", "report"):
        raise ValueError("RELAX_MOE_MEMORY_POLICY must be strict or report")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("Use torchrun --standalone --nproc_per_node=8 -m pytest on this file")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from megatron.core.transformer.moe import experts
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    from relax.backends.megatron.moe_chunk_optimizations import install

    monkeypatch.setenv("RELAX_MOE_METADATA_MODE", "off")
    dist.init_process_group("nccl")
    world = dist.group.WORLD
    rank = dist.get_rank(world)
    pairs = [dist.new_group(ranks=[i, i + 1], backend="nccl") for i in range(0, 8, 2)]
    singletons = [dist.new_group(ranks=[i], backend="nccl") for i in range(8)]
    ep = pairs[rank // 2]
    erank = dist.get_rank(ep)
    groups = SimpleNamespace(ep=ep, expt_tp=singletons[rank], tp_ep=ep, tp=ep)
    config = SimpleNamespace(
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        num_moe_experts=256,
        moe_permute_fusion=True,
        moe_pad_expert_input_to_capacity=False,
        moe_expert_capacity_factor=None,
        moe_router_padding_for_quantization=False,
        moe_router_topk=8,
        moe_router_dtype="fp64",
        cuda_graph_impl="none",
        cuda_graph_modules=[],
    )

    def execute(mode):
        install(mode)
        dispatcher = MoEAlltoAllTokenDispatcher(128, list(range(erank * 128, (erank + 1) * 128)), config, groups)
        torch.manual_seed(1235 + rank)
        xs, logits_list, outputs, losses, counts = [], [], [], [], []
        # A full 8192 chunk is split into two 4096-row SP shards. Also exercise
        # a short tail, an empty receiving expert rank and a full baseline shard.
        for n, h, skew in ((4096, 2048, False), (97, 2048, False), (257, 64, True), (16384, 64, False)):
            x = torch.randn(n, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            logits = torch.randn(n, 256, device="cuda", dtype=torch.float64, requires_grad=True)
            indices = torch.arange(8, device="cuda").expand(n, 8) if skew else logits.topk(8, dim=-1).indices
            route = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, indices, True)
            probs = logits.softmax(dim=-1) * route  # Keep FP64 throughout, including dprobs.
            hidden, p = dispatcher.dispatch_preprocess(x, route, probs)
            hidden, p = dispatcher.token_dispatch(hidden, p)
            hidden, tokens, p = dispatcher.dispatch_postprocess(hidden, p)
            counts.append(tokens.clone())
            # Synthetic FC1/FC2 leave GEMMs out of this dispatcher/activation gate.
            # Their input gradients still travel through the complete MoE path.
            intermediate = torch.cat((hidden * 0.125, hidden * 0.25), dim=-1)
            hidden = experts.weighted_bias_swiglu_impl(intermediate, None, p.unsqueeze(-1))
            hidden = dispatcher.combine_preprocess(hidden)
            hidden = dispatcher.token_combine(hidden)
            output = dispatcher.combine_postprocess(hidden)
            outputs.append(output)
            losses.append(output.float().square().mean())
            xs.append(x)
            logits_list.append(logits)
        # Four outstanding autograd contexts, deliberately consumed in reverse.
        for loss in reversed(losses):
            loss.backward()
        torch.cuda.synchronize()
        peaks = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        tensors = [
            v.detach().cpu()
            for v in (*counts, *outputs, *losses, *(x.grad for x in xs), *(p.grad for p in logits_list))
        ]
        return tensors, peaks

    results = {}
    memory_increases = []
    try:
        reference = None
        for mode in ("off", "dedup", "expert", "both"):
            # Warm compilation and NCCL first; exclude compile caches/temporary
            # tensors from the measured run without retaining GPU reference data.
            execute(mode)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            actual, peaks = execute(mode)
            results[mode] = peaks
            error = None
            if reference is None:
                reference = actual
            else:
                try:
                    for index, (expected, result) in enumerate(zip(reference, actual, strict=True)):
                        torch.testing.assert_close(result, expected, rtol=0, atol=0, msg=f"{mode} tensor {index}")
                    for metric in ("allocated", "reserved"):
                        if peaks[metric] > results["off"][metric]:
                            note = dict(
                                mode=mode, metric=metric, baseline=results["off"][metric], actual=peaks[metric]
                            )
                            memory_increases.append(note)
                            if memory_policy == "strict":
                                raise AssertionError((mode, metric, results))
                            print(f"MoE gate rank {rank}: memory increase (report only): {note}", flush=True)
                except AssertionError as exc:
                    error = f"rank {rank}: {exc}"
            # All ranks finish this mode before any rank can enter its successor.
            # Otherwise a local comparison failure can strand peers in NCCL.
            errors = [None] * 8
            dist.all_gather_object(errors, error, group=world)
            assert not any(errors), "\n".join(e for e in errors if e)
        directory = os.environ.get("RELAX_MOE_GATE_DIR")
        if directory:
            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"rank{rank}.json").write_text(
                json.dumps(
                    dict(
                        rank=rank,
                        passed=True,
                        peaks=results,
                        memory_policy=memory_policy,
                        memory_increases=memory_increases,
                    )
                )
                + "\n"
            )
    finally:
        install("off")
        dist.destroy_process_group()
