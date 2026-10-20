# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-rank KV stores and step telemetry attached to the built model.

Stores live from a group's first forward through its last backward. The active
unit travels on packed_seq_params, not through global state.
"""

from __future__ import annotations

import os
from typing import Any

from relax.backends.megatron.chunkflow.state_store import StateStore
from relax.backends.megatron.chunkflow.telemetry import PlanTelemetry, StepPlanTelemetry


RUNTIME_ATTR = "chunkflow_runtime"
UNIT_ATTR = "chunkflow_unit"
# Fr reuses the K/V leaves stored by Fp; a second put would break ascending order.
RECOMPUTE_ATTR = "chunkflow_recompute"


class ChunkflowRuntime:
    """``(layer_number, group_id) -> StateStore`` for one model on one rank."""

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.schedule_analysis = os.environ.get("RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS") == "1"
        self._stores: dict[tuple[int, int], StateStore] = {}
        self._live_group_ids: set[int] = set()
        self._step_telemetry: StepPlanTelemetry | None = None
        self._max_live_stores = 0
        self._max_live_groups = 0
        self.gate_metrics: dict[str, int | float] | None = None
        self.baseline_rope_fusion: list[tuple[Any, bool]] = []

    def begin_step(self) -> None:
        """Reset plan telemetry and peak counters for the next step."""
        self._step_telemetry = StepPlanTelemetry()
        self._max_live_stores = len(self._stores)
        self._max_live_groups = len(self._live_group_ids)

    def record_plan(self, plan: PlanTelemetry) -> None:
        if self._step_telemetry is None:
            self._step_telemetry = StepPlanTelemetry()
        self._step_telemetry.add(plan)

    def consume_step_metrics(self, pp_size: int) -> dict[str, int | float]:
        """Return and clear the latest step's CPU-only telemetry."""
        telemetry = self._step_telemetry
        self._step_telemetry = None
        gate_metrics, self.gate_metrics = self.gate_metrics, None
        if gate_metrics is not None:
            # Candidate telemetry is counterfactual when the gate selects baseline.
            metrics = dict(gate_metrics)
        elif telemetry is not None:
            metrics = telemetry.to_metrics(pp_size, simulate_schedule=self.schedule_analysis)
        else:
            return {}
        if gate_metrics is not None or metrics:
            metrics["pp_last_max_live_stores"] = self._max_live_stores
            metrics["pp_last_max_live_groups"] = self._max_live_groups
        return metrics

    def store(self, layer_number: int, group_id: int) -> StateStore:
        """The store for one layer of one group, created on first use."""
        key = (layer_number, group_id)
        store = self._stores.get(key)
        if store is None:
            store = StateStore()
            self._stores[key] = store
            self._live_group_ids.add(group_id)
            self._max_live_stores = max(self._max_live_stores, len(self._stores))
            self._max_live_groups = max(self._max_live_groups, len(self._live_group_ids))
        return store

    def release_group(self, group_id: int) -> int:
        """Drop every layer's store for a finished group.

        Returns how many.
        """
        keys = [k for k in self._stores if k[1] == group_id]
        for key in keys:
            del self._stores[key]
        self._live_group_ids.discard(group_id)
        return len(keys)

    def live_groups(self) -> list[int]:
        return sorted(self._live_group_ids)

    def num_stores(self) -> int:
        return len(self._stores)

    def num_elements(self) -> int:
        """Total live state elements -- the K/V this rank is holding."""
        return sum(store.num_elements for store in self._stores.values())

    def assert_empty(self, where: str) -> None:
        """Reject stores leaked by an incomplete backward schedule."""
        if self._stores:
            groups = self.live_groups()
            raise RuntimeError(
                f"chunkflow: {len(self._stores)} store(s) still live after {where} "
                f"(groups {groups}). A dependency group's last backward must call release_group()."
            )


def attach_runtime(model, chunk_size: int) -> ChunkflowRuntime:
    """Create (or return) the runtime for a built model."""
    runtime = getattr(model, RUNTIME_ATTR, None)
    if runtime is None:
        runtime = ChunkflowRuntime(chunk_size)
        setattr(model, RUNTIME_ATTR, runtime)
    return runtime


def find_runtime(model) -> ChunkflowRuntime | None:
    """Find the runtime through Megatron's DDP/Float16 module wrappers."""
    seen = 0
    while model is not None and seen < 8:
        runtime = getattr(model, RUNTIME_ATTR, None)
        if runtime is not None:
            return runtime
        model = getattr(model, "module", None)
        seen += 1
    return None


def require_runtime(model) -> ChunkflowRuntime:
    """Require the installed runtime; creating one on a wrapper splits KV
    state."""
    runtime = find_runtime(model)
    if runtime is None:
        raise RuntimeError("chunkflow runtime is missing; install ChunkFlow before selecting its schedule")
    return runtime
