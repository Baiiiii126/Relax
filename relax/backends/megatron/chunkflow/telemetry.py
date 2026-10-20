# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from relax.backends.megatron.chunkflow.schedule_model import ModelPlan, make_model_plan, predict_step
from relax.backends.megatron.chunkflow.scheduler import Chunk
from relax.backends.megatron.chunkflow.units import Unit, classify_units


@dataclass(frozen=True)
class PlanTelemetry:
    input_rows: int
    source_rows: int
    parent_padding_rows: int
    num_chunks: int
    num_units: int
    unit_padded_rows: int
    recompute_units: int
    model_plan: ModelPlan | None = None


def summarize_plan(
    seq_lengths: Sequence[int],
    chunks: Sequence[Chunk],
    units: Sequence[Unit],
    keep_last_k: int,
    pp_size: int,
    warmup_depths: Sequence[int],
    source_rows: int | None = None,
    include_schedule: bool = False,
) -> PlanTelemetry:
    if pp_size < 1:
        raise ValueError(f"pp_size must be positive, got {pp_size}")
    if len(warmup_depths) != pp_size:
        raise ValueError(f"expected {pp_size} warmup depths, got {len(warmup_depths)}")

    _, need_fr = classify_units(units, chunks, keep_last_k)
    input_rows = sum(int(length) for length in seq_lengths)
    source_rows = input_rows if source_rows is None else int(source_rows)
    return PlanTelemetry(
        input_rows=input_rows,
        source_rows=source_rows,
        parent_padding_rows=max(0, input_rows - source_rows),
        num_chunks=len(chunks),
        num_units=len(units),
        unit_padded_rows=sum((unit.num_tokens + 127) // 128 * 128 for unit in units),
        recompute_units=len(need_fr),
        model_plan=make_model_plan(units, chunks, keep_last_k, pp_size) if include_schedule else None,
    )


@dataclass
class StepPlanTelemetry:
    plans: list[PlanTelemetry] = field(default_factory=list)

    def add(self, plan: PlanTelemetry) -> None:
        self.plans.append(plan)

    def to_metrics(self, pp_size: int, *, simulate_schedule: bool = False) -> dict[str, int | float]:
        if not self.plans:
            return {}
        metrics = {
            "num_microbatches": len(self.plans),
            "input_rows": sum(plan.input_rows for plan in self.plans),
            "source_rows": sum(plan.source_rows for plan in self.plans),
            "parent_padding_rows": sum(plan.parent_padding_rows for plan in self.plans),
            "num_chunks": sum(plan.num_chunks for plan in self.plans),
            "num_units": sum(plan.num_units for plan in self.plans),
            "unit_padded_rows": sum(plan.unit_padded_rows for plan in self.plans),
            "recompute_units": sum(plan.recompute_units for plan in self.plans),
        }
        if simulate_schedule:
            started = time.perf_counter()
            prediction = predict_step(self.plans, pp_size)
            metrics.update(
                {
                    "schedule_model_baseline_makespan": prediction["baseline"]["makespan"],
                    "schedule_model_chunkflow_makespan": prediction["chunkflow"]["makespan"],
                    "schedule_model_speedup": prediction["speedup"],
                    "schedule_model_baseline_bubble_fraction": prediction["baseline"]["bubble_fraction"],
                    "schedule_model_chunkflow_bubble_fraction": prediction["chunkflow"]["bubble_fraction"],
                    "schedule_model_analysis_time_s": time.perf_counter() - started,
                    "schedule_model_version": 1,
                }
            )
        return metrics


def remote_metrics(metrics: dict[str, Any], chunkflow_prefix: str) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith(chunkflow_prefix)}
