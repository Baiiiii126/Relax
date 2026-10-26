# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dependency-free checks for chunk construction and prediction."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parents[3]


def _load_chunkflow() -> SimpleNamespace:
    package = "relax.backends.megatron.chunkflow"
    folder = ROOT / "relax/backends/megatron/chunkflow"
    modules = {}
    with patch.dict(sys.modules):
        for name in ("relax", "relax.backends", "relax.backends.megatron", package):
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        for leaf in ("scheduler", "units", "schedule_model", "telemetry", "gating"):
            name = f"{package}.{leaf}"
            spec = importlib.util.spec_from_file_location(name, folder / f"{leaf}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[leaf] = module
    return SimpleNamespace(**modules)


class ChunkflowPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cf = _load_chunkflow()

    def test_gate_forecast_peeks_current_parents_and_excludes_padding_from_units(self):
        from dataclasses import replace

        lengths = [999, 2050, 300, 4097]
        iterator = SimpleNamespace(
            offset=1, micro_batch_indices=[[0], [1, 2], [3]], rollout_data={"total_lengths": lengths}
        )
        plans = []
        for source in ([2050, 300], [4097]):
            chunks = self.cf.scheduler.build_chunks(source, 1024, repack_tail=False)
            units = self.cf.units.build_micro_batches(chunks, "staggered")
            plans.append(
                replace(
                    self.cf.telemetry.summarize_plan(
                        [*source, (-sum(source)) % 256],
                        chunks,
                        units,
                        1 << 30,
                        4,
                        [0] * 4,
                        source_rows=sum(source),
                        include_schedule=True,
                    ),
                    parent_window=2,
                )
            )
        expected = self.cf.telemetry.StepPlanTelemetry(plans).to_metrics(4, simulate_schedule=True)
        got, fingerprint = self.cf.gating.forecast_step(
            iterator, 2, chunk_size=1024, pp_size=4, tp_size=2, pad_multiplier=128
        )
        for key in expected.keys() - {"schedule_model_analysis_time_s"}:
            self.assertEqual(got[key], expected[key], key)
        self.assertEqual(iterator.offset, 1)
        self.assertEqual(got["recompute_units"], 0)
        other, other_hash = self.cf.gating.forecast_step(
            iterator, 2, chunk_size=1024, pp_size=4, tp_size=2, pad_multiplier=128, parent_window=1
        )
        self.assertNotEqual(fingerprint, other_hash)
        self.assertNotEqual(got["schedule_model_speedup"], other["schedule_model_speedup"])
        iterator.dummy_micro_batch_offsets = {2}
        with self.assertRaisesRegex(ValueError, "dummy"):
            self.cf.gating.forecast_step(iterator, 2, chunk_size=1024, pp_size=4, tp_size=2, pad_multiplier=128)

    def test_plan_prediction_preserves_parent_padding(self):
        chunks = self.cf.scheduler.build_chunks([1100, 180], 512, repack_tail=False)
        units = self.cf.units.build_micro_batches(chunks, "staggered")
        plan = self.cf.telemetry.summarize_plan(
            [1100, 180], chunks, units, 1 << 30, 4, [4, 4, 3, 2], source_rows=1100, include_schedule=True
        )
        step = self.cf.telemetry.StepPlanTelemetry([plan])
        metrics = step.to_metrics(4, simulate_schedule=True)
        expected = {"source_rows": 1100, "input_rows": 1280, "parent_padding_rows": 180, "num_units": 4}
        for key, value in expected.items():
            self.assertEqual(metrics[key], value)
        self.assertAlmostEqual(metrics["schedule_model_speedup"], 12 / 7)

    def test_remote_metrics_preserves_training_fields_and_full_local_diagnostics(self):
        for prefix in ("train/chunkflow/", "train/critic-chunkflow/"):
            local = {
                "train/step": 12,
                "train/loss": 0.2,
                "train/grad_norm": 1.5,
                "perf/actor_train_time": 3.0,
                prefix + "num_units": 4,
                prefix + "schedule_model_speedup": 2.3,
            }
            before = dict(local)
            remote = self.cf.telemetry.remote_metrics(local, prefix)
            self.assertEqual(
                remote, {"train/step": 12, "train/loss": 0.2, "train/grad_norm": 1.5, "perf/actor_train_time": 3.0}
            )
            self.assertEqual(local, before)

    def test_equal_cost_single_parent_matches_fill_drain(self):
        model = self.cf.schedule_model
        for pp in (1, 4):
            for n in (1, 4):
                plan = model.ModelPlan(
                    {i: 128 for i in range(n)},
                    tuple(range(n)),
                    tuple(reversed(range(n))),
                    frozenset(),
                    tuple(min(n - 1 + pp - s - 1, n) for s in range(pp)),
                )
                result = model.simulate([plan], pp)
                self.assertAlmostEqual(result["makespan"], 3 * (n + pp - 1))

    def test_multiple_parents_restart_chunkflow_pipeline(self):
        chunks = self.cf.scheduler.build_chunks([512], 128, repack_tail=False)
        units = self.cf.units.build_micro_batches(chunks, "staggered")
        plan = self.cf.telemetry.summarize_plan([512], chunks, units, 1 << 30, 4, [4, 4, 4, 3], include_schedule=True)
        prediction = self.cf.schedule_model.predict_step([plan, plan], 4)
        self.assertAlmostEqual(prediction["baseline"]["makespan"], 60)
        self.assertAlmostEqual(prediction["chunkflow"]["makespan"], 42)

    def test_parent_window_preserves_orders_and_reduces_pipeline_restarts(self):
        from dataclasses import replace

        model = self.cf.schedule_model
        for pp in (2, 4):
            for keep in (1, 1 << 30):
                plans = []
                for lengths in ([2048], [2048], [513, 128], [1800], [513]):
                    chunks = self.cf.scheduler.build_chunks(lengths, 512, repack_tail=False)
                    units = self.cf.units.build_micro_batches(chunks, "staggered")
                    plans.append(
                        self.cf.telemetry.summarize_plan(
                            lengths, chunks, units, keep, pp, [0] * pp, include_schedule=True
                        )
                    )
                old = model.predict_step(plans, pp)
                for window in (2, 4):
                    new = model.predict_step([replace(p, parent_window=window) for p in plans], pp)
                    self.assertLess(new["chunkflow"]["makespan"], old["chunkflow"]["makespan"])
                    self.assertEqual(new["baseline"], old["baseline"])
                merged = model.merge_model_plans([p.model_plan for p in plans], pp)
                offset = 0
                for plan in plans:
                    local = plan.model_plan
                    n = len(local.rows)
                    self.assertEqual(merged.forward[offset : offset + n], tuple(u + offset for u in local.forward))
                    self.assertEqual(merged.backward[offset : offset + n], tuple(u + offset for u in local.backward))
                    offset += n
                self.assertEqual(sum(merged.rows.values()), sum(sum(p.model_plan.rows.values()) for p in plans))
                model.simulate([merged], pp)  # Blocking transfers must remain deadlock-free.

    def test_chunk_split_and_packing_preserve_tokens_and_prefixes(self):
        lengths = [2050, 300, 200]
        chunks = self.cf.scheduler.build_chunks(lengths, 1024, repack_tail=False)
        dependent = [chunk for chunk in chunks if chunk.is_dependent]
        self.assertEqual([c.segments[0].prefix for c in dependent], [0, 1024, 2048])
        self.assertEqual([c.num_tokens for c in dependent], [1024, 1024, 2])
        self.assertEqual(sum(c.num_tokens for c in chunks), sum(lengths))
        standalone = [chunk for chunk in chunks if not chunk.is_dependent]
        self.assertEqual([c.num_tokens for c in standalone], [500])
        units = self.cf.units.build_micro_batches(chunks, "staggered")
        self.assertEqual([u.chunk_ids for u in units], [[c.chunk_id] for c in chunks])


if __name__ == "__main__":
    unittest.main()
