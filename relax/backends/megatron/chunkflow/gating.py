# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Experimental pre-step gate for materialized, text-only DP=CP=1 batches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from numbers import Integral
from typing import Any

from relax.backends.megatron.chunkflow.schedule_model import pipeline_plan
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.telemetry import StepPlanTelemetry, summarize_plan
from relax.backends.megatron.chunkflow.units import build_micro_batches


GATE_ENV = "RELAX_CHUNKFLOW_GATE_GAIN_PCT"


def remote_metrics(metrics: dict[str, Any], chunkflow_prefix: str) -> dict[str, Any]:
    """Keep diagnostics in the local log, outside synchronous reporting."""
    return {key: value for key, value in metrics.items() if not key.startswith(chunkflow_prefix)}


def gain_threshold(value: str) -> float | None:
    if not value:
        return None
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"{GATE_ENV} must be a finite nonnegative gain percentage")
    return threshold


def forecast_step(
    iterator: Any, num_microbatches: int, *, chunk_size: int, pp_size: int, tp_size: int, pad_multiplier: int
) -> tuple[dict[str, int | float], str]:
    """Peek CPU lengths at the current offset; never fetch or advance data."""
    if not isinstance(num_microbatches, Integral) or num_microbatches < 1:
        raise ValueError("gate requires a positive CPU microbatch count")
    indices = getattr(iterator, "micro_batch_indices", None)
    if indices is None or not hasattr(iterator, "rollout_data"):
        raise ValueError("gate requires a materialized dynamic DataIterator; streaming iterators are unsupported")
    offset = iterator.offset
    partitions = indices[offset : offset + num_microbatches]
    if len(partitions) != num_microbatches or any(not group for group in partitions):
        raise ValueError("gate cannot peek a complete nonempty optimizer step")
    if any(i in getattr(iterator, "dummy_micro_batch_offsets", ()) for i in range(offset, offset + num_microbatches)):
        raise ValueError("gate does not support dummy microbatches")
    all_lengths = iterator.rollout_data["total_lengths"]
    samples = [[all_lengths[i] for i in group] for group in partitions]
    if any(not isinstance(n, Integral) or n < 1 for group in samples for n in group):
        raise ValueError("gate requires positive CPU integer token lengths")
    samples = [[int(n) for n in group] for group in samples]
    pad_size = tp_size * pad_multiplier
    if min(chunk_size, pp_size, pad_size) < 1:
        raise ValueError("invalid gate layout")
    step = StepPlanTelemetry()
    for source in samples:
        lengths = [int(n) for n in source]
        padding = (-sum(lengths)) % pad_size
        if padding:
            lengths.append(padding)
        chunks = build_chunks(lengths, chunk_size, repack_tail=False)
        units = build_micro_batches(chunks, "staggered")
        _, _, _, delay = pipeline_plan(units, chunks, 1 << 30)
        warmups = [min(delay + pp_size - s - 1, len(units)) for s in range(pp_size)]
        step.add(
            summarize_plan(
                lengths, chunks, units, 1 << 30, pp_size, warmups, source_rows=sum(source), include_schedule=True
            )
        )
    fingerprint = hashlib.sha256(json.dumps(samples).encode()).hexdigest()
    return step.to_metrics(pp_size, simulate_schedule=True), fingerprint


def prepare_step_gate(args: Any, model: Any, iterator: Any, num_microbatches: int) -> bool:
    """Agree before scheduling; every rank enters the same CPU Gloo
    collective."""
    import torch.distributed as dist
    from megatron.core import mpu

    from relax.backends.megatron.chunkflow.runtime import require_runtime
    from relax.utils.distributed_utils import get_gloo_group
    from relax.utils.logging_utils import get_logger

    threshold = gain_threshold(os.environ.get(GATE_ENV, ""))
    if threshold is None:
        return True
    runtime = require_runtime(model)
    started = time.perf_counter()
    metrics, error, fingerprint = {}, None, None
    try:
        runtime.assert_empty("before adaptive step")
        if (
            mpu.get_data_parallel_world_size(with_context_parallel=True) != 1
            or mpu.get_context_parallel_world_size() != 1
            or mpu.get_virtual_pipeline_model_parallel_world_size() is not None
            or getattr(args, "dynamic_context_parallel", False)
            or getattr(args, "is_vl_model", False)
            or args.qkv_format != "thd"
            or getattr(args, "loss_type", None) != "sft"
            or getattr(args, "chunkflow_pack_policy", "staggered") != "staggered"
            or getattr(args, "chunkflow_keep_last_k", 0) != 0
        ):
            raise ValueError("gate supports only text SFT, DP=CP=1, no VPP, THD, staggered, K=0")
        metrics, fingerprint = forecast_step(
            iterator,
            num_microbatches,
            chunk_size=args.chunkflow_chunk_size,
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            pad_multiplier=args.data_pad_size_multiplier,
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as exc:
        error = str(exc)
    prediction_s = time.perf_counter() - started
    speedup = metrics.get("schedule_model_speedup", 0.0)
    selected = speedup > 1.0 + threshold / 100.0
    group = get_gloo_group()
    peers = [None] * dist.get_world_size(group=group)
    consensus_started = time.perf_counter()
    local = (error, fingerprint, speedup, selected, threshold)
    dist.all_gather_object(peers, local, group=group)
    consensus_s = time.perf_counter() - consensus_started
    if any(peer[0] is not None for peer in peers):
        raise ValueError(f"adaptive ChunkFlow preflight failed on a rank: {peers}")
    if any(peer != local for peer in peers):
        raise ValueError("adaptive ChunkFlow prediction/input differs between ranks; refusing mixed schedules")
    runtime.begin_step()
    runtime.gate_metrics = metrics | {
        "gate_selected": int(selected),
        "gate_threshold_gain_pct": threshold,
        "gate_prediction_time_s": prediction_s,
        "gate_consensus_time_s": consensus_s,
    }
    if dist.get_rank(group=group) == 0:
        get_logger(__name__).info(
            "chunkflow gate before forward: predicted_gain_pct=%.4f threshold_pct=%g selected=%s",
            100 * (speedup - 1),
            threshold,
            "chunkflow" if selected else "baseline",
        )
    return selected
