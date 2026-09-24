# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure-Python cost proxies for a ChunkFlow execution plan.

These metrics describe the plan, not elapsed time.  They deliberately avoid
CUDA events, tensor values, and collectives so collecting them cannot perturb
the schedule that they are meant to explain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from relax.backends.megatron.chunkflow.scheduler import Chunk
from relax.backends.megatron.chunkflow.units import Unit, classify_units


@dataclass(frozen=True)
class PlanTelemetry:
    """Cost proxies for one parent micro-batch."""

    input_rows: int
    source_rows: int
    parent_padding_rows: int
    num_chunks: int
    num_units: int
    num_groups: int
    num_standalone_chunks: int
    num_dependent_chunks: int
    max_group_depth: int
    unit_padded_rows: int
    q_rows: int
    kv_rows: int
    logical_attention_pairs: int
    logical_pairs_match: int
    kernel_rect_pairs: int
    recompute_units: int
    recompute_rows: int
    recompute_kernel_rect_pairs: int
    warmup_min: int
    warmup_max: int
    p2p_messages: int
    p2p_row_transfers: int


def summarize_plan(
    seq_lengths: Sequence[int],
    chunks: Sequence[Chunk],
    units: Sequence[Unit],
    keep_last_k: int,
    pp_size: int,
    warmup_depths: Sequence[int],
    source_rows: int | None = None,
) -> PlanTelemetry:
    """Summarize one plan without inspecting tensors or changing the plan."""
    if pp_size < 1:
        raise ValueError(f"pp_size must be positive, got {pp_size}")
    if len(warmup_depths) != pp_size:
        raise ValueError(f"expected {pp_size} warmup depths, got {len(warmup_depths)}")

    _, need_fr = classify_units(units, chunks, keep_last_k)
    padded_rows = {unit.unit_id: math.ceil(unit.num_tokens / 128) * 128 for unit in units}

    q_rows = 0
    kv_rows = 0
    logical_attention_pairs = 0
    kernel_rect_pairs = 0
    kernel_rect_pairs_by_chunk: dict[int, int] = {}
    for chunk in chunks:
        chunk_kernel_rect_pairs = 0
        for segment in chunk.segments:
            q_len = segment.length
            kv_len = q_len if not chunk.is_dependent else segment.prefix + q_len
            q_rows += q_len
            kv_rows += kv_len
            logical_attention_pairs += q_len * (kv_len - q_len) + q_len * (q_len + 1) // 2
            kernel_rect_pairs += q_len * kv_len
            chunk_kernel_rect_pairs += q_len * kv_len
        kernel_rect_pairs_by_chunk[chunk.chunk_id] = chunk_kernel_rect_pairs

    input_rows = sum(int(length) for length in seq_lengths)
    source_rows = input_rows if source_rows is None else int(source_rows)
    expected_pairs = sum(int(length) * (int(length) + 1) // 2 for length in seq_lengths)
    unit_kernel_rect_pairs = {
        unit.unit_id: sum(kernel_rect_pairs_by_chunk[chunk.chunk_id] for chunk in unit.chunks) for unit in units
    }

    num_units = len(units)
    unit_padded_rows = sum(padded_rows.values())
    p2p_links = pp_size - 1
    return PlanTelemetry(
        input_rows=input_rows,
        source_rows=source_rows,
        parent_padding_rows=max(0, input_rows - source_rows),
        num_chunks=len(chunks),
        num_units=num_units,
        num_groups=len({chunk.group for chunk in chunks if chunk.is_dependent}),
        num_standalone_chunks=sum(not chunk.is_dependent for chunk in chunks),
        num_dependent_chunks=sum(chunk.is_dependent for chunk in chunks),
        max_group_depth=max((chunk.group_size for chunk in chunks if chunk.is_dependent), default=0),
        unit_padded_rows=unit_padded_rows,
        q_rows=q_rows,
        kv_rows=kv_rows,
        logical_attention_pairs=logical_attention_pairs,
        logical_pairs_match=int(logical_attention_pairs == expected_pairs),
        kernel_rect_pairs=kernel_rect_pairs,
        recompute_units=len(need_fr),
        recompute_rows=sum(padded_rows[unit_id] for unit_id in need_fr),
        recompute_kernel_rect_pairs=sum(unit_kernel_rect_pairs[unit_id] for unit_id in need_fr),
        warmup_min=min(warmup_depths, default=0),
        warmup_max=max(warmup_depths, default=0),
        p2p_messages=2 * p2p_links * num_units,
        p2p_row_transfers=2 * p2p_links * unit_padded_rows,
    )


@dataclass
class StepPlanTelemetry:
    """Aggregate plans produced by one optimizer step on one rank."""

    plans: list[PlanTelemetry] = field(default_factory=list)

    def add(self, plan: PlanTelemetry) -> None:
        self.plans.append(plan)

    def to_metrics(self, pp_size: int) -> dict[str, int | float]:
        if not self.plans:
            return {}

        num_microbatches = len(self.plans)
        input_rows = sum(plan.input_rows for plan in self.plans)
        source_rows = sum(plan.source_rows for plan in self.plans)
        padded_rows = sum(plan.unit_padded_rows for plan in self.plans)
        recompute_rows = sum(plan.recompute_rows for plan in self.plans)
        kernel_rect_pairs = sum(plan.kernel_rect_pairs for plan in self.plans)
        recompute_kernel_rect_pairs = sum(plan.recompute_kernel_rect_pairs for plan in self.plans)
        num_units = sum(plan.num_units for plan in self.plans)

        baseline_fill_drain = 1.0 + (pp_size - 1) / num_microbatches
        chunkflow_fill_drain = sum(
            plan.unit_padded_rows * (1.0 + (pp_size - 1) / max(plan.num_units, 1)) for plan in self.plans
        ) / max(padded_rows, 1)

        return {
            "num_microbatches": num_microbatches,
            "input_rows": input_rows,
            "source_rows": source_rows,
            "parent_padding_rows": sum(plan.parent_padding_rows for plan in self.plans),
            "num_chunks": sum(plan.num_chunks for plan in self.plans),
            "num_units": num_units,
            "min_units_per_microbatch": min(plan.num_units for plan in self.plans),
            "max_units_per_microbatch": max(plan.num_units for plan in self.plans),
            "num_groups": sum(plan.num_groups for plan in self.plans),
            "num_standalone_chunks": sum(plan.num_standalone_chunks for plan in self.plans),
            "num_dependent_chunks": sum(plan.num_dependent_chunks for plan in self.plans),
            "max_group_depth": max(plan.max_group_depth for plan in self.plans),
            "unit_padded_rows": padded_rows,
            "fragmentation_amplification": padded_rows / max(input_rows, 1),
            "total_padding_amplification": padded_rows / max(source_rows, 1),
            "q_rows": sum(plan.q_rows for plan in self.plans),
            "kv_rows": sum(plan.kv_rows for plan in self.plans),
            "logical_attention_pairs": sum(plan.logical_attention_pairs for plan in self.plans),
            "logical_pairs_match": min(plan.logical_pairs_match for plan in self.plans),
            "kernel_rect_pairs": kernel_rect_pairs,
            "recompute_units": sum(plan.recompute_units for plan in self.plans),
            "recompute_rows": recompute_rows,
            "recompute_kernel_rect_pairs": recompute_kernel_rect_pairs,
            "token_forward_recompute_multiplier": 1.0 + recompute_rows / max(padded_rows, 1),
            "attention_forward_recompute_multiplier": 1.0 + recompute_kernel_rect_pairs / max(kernel_rect_pairs, 1),
            "launch_amplification": num_units / num_microbatches,
            "warmup_min": min(plan.warmup_min for plan in self.plans),
            "warmup_max": max(plan.warmup_max for plan in self.plans),
            "min_steady_unit_fraction": min(
                (plan.num_units - plan.warmup_max) / max(plan.num_units, 1) for plan in self.plans
            ),
            "equal_unit_fill_drain_baseline_proxy": baseline_fill_drain,
            "equal_unit_fill_drain_chunkflow_proxy": chunkflow_fill_drain,
            "equal_unit_fill_drain_delta_proxy": baseline_fill_drain - chunkflow_fill_drain,
            "p2p_messages": sum(plan.p2p_messages for plan in self.plans),
            "p2p_row_transfers": sum(plan.p2p_row_transfers for plan in self.plans),
        }
