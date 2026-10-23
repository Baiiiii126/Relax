# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import os

from relax.backends.megatron.chunkflow.schedule_model import ModelPlan
from relax.backends.megatron.chunkflow.state_store import StateStore
from relax.backends.megatron.chunkflow.telemetry import PlanTelemetry, StepPlanTelemetry


RUNTIME_ATTR = "chunkflow_runtime"
UNIT_ATTR = "chunkflow_unit"
RECOMPUTE_ATTR = "chunkflow_recompute"


class ChunkflowRuntime:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.schedule_analysis = os.environ.get("RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS") == "1"
        self._stores: dict[tuple[int, int], StateStore] = {}
        self._step_telemetry: StepPlanTelemetry | None = None

    def begin_step(self) -> None:
        self._step_telemetry = StepPlanTelemetry()

    def record_plan(self, plan: PlanTelemetry) -> None:
        if self._step_telemetry is None:
            self._step_telemetry = StepPlanTelemetry()
        self._step_telemetry.add(plan)

    def consume_step_metrics(self, pp_size: int) -> dict[str, int | float]:
        telemetry, self._step_telemetry = self._step_telemetry, None
        return telemetry.to_metrics(pp_size, simulate_schedule=self.schedule_analysis) if telemetry is not None else {}

    def record_window(self, plan: ModelPlan, pooled_units: int) -> None:
        if self._step_telemetry is None:
            self._step_telemetry = StepPlanTelemetry()
        self._step_telemetry.windows.append(plan)
        self._step_telemetry.pooled_units += pooled_units

    def store(self, layer_number: int, group_id: int) -> StateStore:
        key = (layer_number, group_id)
        store = self._stores.get(key)
        if store is None:
            store = StateStore()
            self._stores[key] = store
        return store

    def release_group(self, group_id: int) -> int:
        keys = [k for k in self._stores if k[1] == group_id]
        for key in keys:
            del self._stores[key]
        return len(keys)

    def live_groups(self) -> list[int]:
        return sorted({group_id for _, group_id in self._stores})

    def num_stores(self) -> int:
        return len(self._stores)

    def assert_empty(self, where: str) -> None:
        if self._stores:
            groups = self.live_groups()
            raise RuntimeError(
                f"chunkflow: {len(self._stores)} store(s) still live after {where} "
                f"(groups {groups}). A dependency group's last backward must call release_group()."
            )


def attach_runtime(model, chunk_size: int) -> ChunkflowRuntime:
    runtime = getattr(model, RUNTIME_ATTR, None)
    if runtime is None:
        runtime = ChunkflowRuntime(chunk_size)
        setattr(model, RUNTIME_ATTR, runtime)
    return runtime


def find_runtime(model) -> ChunkflowRuntime | None:
    seen = 0
    while model is not None and seen < 8:
        runtime = getattr(model, RUNTIME_ATTR, None)
        if runtime is not None:
            return runtime
        model = getattr(model, "module", None)
        seen += 1
    return None


def require_runtime(model) -> ChunkflowRuntime:
    runtime = find_runtime(model)
    if runtime is None:
        raise RuntimeError("chunkflow runtime is missing; install ChunkFlow before selecting its schedule")
    return runtime
