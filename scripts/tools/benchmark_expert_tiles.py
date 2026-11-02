# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Computation-only expert MLP benchmark; no EP traffic or full-model training.

Each torchrun rank benchmarks one GPU independently. TE and Triton use
identical BF16 weight storage, FP32 gradient buffers and MCore's FP64 weighted
activation. TE receives precomputed CPU counts: timings do not charge it a
synthetic DtoH.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def load_tiles() -> Any:
    folder = Path(__file__).resolve().parents[2] / "relax/backends/megatron/chunkflow"
    package = ModuleType("_expert_tiles_benchmark")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    name = package.__name__ + ".expert_tiles"
    spec = importlib.util.spec_from_file_location(name, folder / "expert_tiles.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32768, help="Routed rows on one GPU, not sequence length")
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument(
        "--block-m",
        type=int,
        nargs="+",
        choices=(0, 32, 64, 128, 256),
        default=[128, 256, 0],
        help="0 selects mixed 256/128/64 tiles within one kernel",
    )
    parser.add_argument("--distribution", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--counts-file", type=Path, help="Replay a profiler expert_counts_*.jsonl record")
    parser.add_argument("--counts-record", type=int, default=0, help="Zero-based record containing counts")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if min(args.rows, args.experts, args.hidden, args.ffn, args.warmup, args.steps) <= 0:
        parser.error("dimensions, warmup and steps must be positive")
    if args.counts_record < 0 or (args.counts_record and args.counts_file is None):
        parser.error("counts-record requires counts-file and must be nonnegative")
    return args


def read_counts(path: Path, index: int) -> dict:
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if "counts" not in record:
                continue
            if index:
                index -= 1
                continue
            counts = record["counts"]
            if not isinstance(counts, list) or not counts or any(type(n) is not int or n < 0 for n in counts):
                raise ValueError("Replay counts must be a nonempty list of nonnegative integers")
            if not sum(counts) or record.get("rows", sum(counts)) != sum(counts):
                raise ValueError("Replay counts must sum to a positive row count matching the record")
            return record
    raise ValueError(f"Requested counts record is absent in {path}")


def run(args: argparse.Namespace) -> None:
    replay = read_counts(args.counts_file, args.counts_record) if args.counts_file else None
    if replay:
        if replay.get("hidden", args.hidden) != args.hidden:
            raise ValueError("--hidden must match the captured layer")
        args.experts, args.rows = len(replay["counts"]), sum(replay["counts"])
    import torch

    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    import transformer_engine.pytorch as te
    from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

    torch.manual_seed(1235 + rank)
    tiles = load_tiles()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"rank-{rank}.jsonl"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    def emit(record: dict) -> None:
        line = json.dumps({"rank": rank, **record}, sort_keys=True)
        print(line, flush=True)
        with destination.open("a") as output:
            output.write(line + "\n")

    emit(
        {
            "event": "start",
            "scope": "independent_gpu_expert_compute_only; no EP or ChunkFlow training",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": importlib.metadata.version("triton"),
            "transformer_engine": importlib.metadata.version("transformer-engine"),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "precision": "BF16 operands, FP32 accumulation/main_grad, FP64 probabilities",
            "memory_scope": "PyTorch allocator; common weights/main_grad/TE objects stay resident in all arms",
            "reserved_scope": "Cumulative allocator cache across arms; not an arm-to-arm memory comparison",
        }
    )
    if replay:
        counts = replay["counts"]
    else:
        proportions = [1 if args.distribution == "balanced" else (e + 1) ** 2 for e in range(args.experts)]
        counts = [args.rows * p // sum(proportions) for p in proportions]
        for e in range(args.rows - sum(counts)):
            counts[e % args.experts] += 1
    emit(
        {
            "event": "expert_counts",
            "distribution": "captured" if replay else args.distribution,
            "counts": counts,
            "capture": replay,
            "replay_scope": "real counts with synthetic operands" if replay else "synthetic counts and operands",
        }
    )
    offsets = tiles.make_offsets(torch.tensor(counts, device="cuda", dtype=torch.int64))
    x = (torch.randn(args.rows, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    probs = torch.rand(args.rows, device="cuda", dtype=torch.float64, requires_grad=True)
    w1 = (
        torch.randn(args.experts, 2 * args.ffn, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn(args.experts, args.hidden, args.ffn, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    grads = (torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
    sample_indices = []
    for grad in grads:
        samples = min(4096, grad.numel())
        indices = torch.arange(samples, device="cuda", dtype=torch.int64)
        sample_indices.append(indices * (grad.numel() - 1) // max(1, samples - 1))
    dout = torch.randn_like(x) * 0.1

    def te_linear(weights, main_grad):
        experts, outputs, inputs = weights.shape
        layer = te.GroupedLinear(
            experts,
            inputs,
            outputs,
            bias=False,
            params_dtype=torch.bfloat16,
            device="meta",
            fuse_wgrad_accumulation=True,
        )
        for e in range(experts):
            parameter = torch.nn.Parameter(weights[e])
            parameter.main_grad = main_grad[e]
            layer.register_parameter(f"weight{e}", parameter)
        return layer

    fc1, fc2 = te_linear(w1, grads[0]), te_linear(w2, grads[1])

    def activation(h, p):
        return weighted_bias_swiglu_impl(h, None, p[:, None], False, None)

    def step(block_m):
        x.grad = probs.grad = None
        if block_m is None:
            y = fc2(activation(fc1(x, counts), probs), counts)
        else:
            y = tiles.expert_mlp(x, w1, w2, offsets, probs, activation, main_grads=grads, block_m=block_m)
        y.backward(dout)
        return y

    reference = None
    for block_m in [None, *dict.fromkeys(args.block_m)]:
        arm = "te" if block_m is None else ("gpu_tiles_mixed" if block_m == 0 else f"gpu_tiles_m{block_m}")
        emit({"event": "arm_start", "arm": arm})
        for grad in grads:
            grad.zero_()
        y = step(block_m)
        torch.cuda.synchronize()
        # Full output/dX/dp checks; bounded dW samples. GPU unit tests cover full dW.
        tensors = {"output": y, "dx": x.grad, "dp": probs.grad}
        for index, (grad, indices) in enumerate(zip(grads, sample_indices)):
            tensors[f"dw{index + 1}_sample"] = grad.flatten().index_select(0, indices)
        snapshot = {key: value.detach().float().cpu() for key, value in tensors.items()}
        # Preserve FP64 in the probability-gradient comparison.
        snapshot["dp"] = probs.grad.detach().cpu()
        if reference is None:
            reference = snapshot
        else:
            error = {}
            for key, value in snapshot.items():
                difference = value.double() - reference[key].double()
                error[key] = {
                    "max_abs": difference.abs().max().item(),
                    "relative_l2": (difference.norm() / reference[key].double().norm().clamp_min(1e-30)).item(),
                }
            emit({"event": "numerics", "arm": arm, "errors": error, "acceptance_gate": False})
        del y, tensors, snapshot
        for _ in range(args.warmup):
            step(block_m)
        x.grad = probs.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_time = time.perf_counter()
        begin.record()
        for _ in range(args.steps):
            step(block_m)
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - start_time) * 1000 / args.steps
        emit(
            {
                "event": "timing",
                "arm": arm,
                "gpu_ms_per_mlp_forward_backward": begin.elapsed_time(end) / args.steps,
                "wall_ms_per_mlp_forward_backward": wall_ms,
                "allocated_before_bytes": allocated_before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - allocated_before,
                "cumulative_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    emit({"event": "complete"})


if __name__ == "__main__":
    run(arguments())
