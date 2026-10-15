# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only regression checks, also runnable with python -m unittest."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "scripts" / "training" / "sft" / "tools"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.align = load_script("chunkflow_align")
        cls.benefit = load_script("chunkflow_benefit_model")
        cls.cf = cls.benefit._load_chunkflow(ROOT)

    def test_all_four_direction_cells(self):
        rows = [{"theory_gain_pct": x, "actual_gain_pct": y} for x, y in ((20, 10), (-20, 10), (20, -10), (-20, -10))]
        report = self.align._attribution(rows)
        self.assertEqual(report["direction_agreement"], 0.5)
        self.assertTrue(all(n == 1 for row in report["counts"].values() for n in row.values()))
        self.assertAlmostEqual(report["pearson"], 0.0)

    def test_neutral_band_and_undefined_constant_correlation(self):
        rows = [{"theory_gain_pct": 10.0, "actual_gain_pct": y} for y in (0.5, 2, -2)]
        report = self.align._attribution(rows, tolerance_pct=1)
        self.assertEqual(report["neutral_steps"], 1)
        self.assertIsNone(report["pearson"])
        self.assertIsNone(report["spearman"])
        self.assertEqual(self.align._ranks([2, 1, 2, 3]), [1.5, 0.0, 1.5, 3.0])

    def test_scalar_repr_and_ansi_do_not_drop_valid_metrics(self):
        rows = list(
            self.align._iter_dicts(
                "\x1b[32mperf   7:\x1b[0m {'perf/actor_train_time': np.float64(3.5), 'other': tensor(2.)}",
                "perf",
            )
        )
        self.assertEqual(rows, [(7, {"perf/actor_train_time": 3.5, "other": 2.0})])
        self.assertEqual(self.align._metric_dict("{'x': __import__('os'), 'y': 2}"), {"y": 2})

    def test_equal_cost_single_parent_matches_fill_drain(self):
        model = self.cf.schedule_model
        for pp in (1, 2, 4):
            for n in (1, 4, 16):
                plan = model.ModelPlan(
                    {i: 128 for i in range(n)},
                    tuple(range(n)),
                    tuple(reversed(range(n))),
                    frozenset(),
                    tuple(min(n - 1 + pp - s - 1, n) for s in range(pp)),
                )
                result = model.simulate([plan], pp)
                self.assertAlmostEqual(result["makespan"], 3 * (n + pp - 1))

    def test_variable_length_timeline_covers_dependencies_and_stage_exclusivity(self):
        for lengths in ([2600], [2048, 700, 300, 1500], [96, 96, 96, 96]):
            for policy in ("staggered", "lockstep"):
                for keep in (1, 2, 1 << 30):
                    chunks = self.cf.scheduler.build_chunks(lengths, 512, repack_tail=False)
                    units = self.cf.units.build_micro_batches(chunks, policy)
                    plan = self.cf.schedule_model.make_model_plan(units, chunks, keep, 4)
                    trace = self.cf.schedule_model.simulate([plan], 4)["timeline"]
                    for stage in range(4):
                        events = sorted((e for e in trace if e["stage"] == stage), key=lambda e: e["start"])
                        for a, b in zip(events, events[1:]):
                            self.assertLessEqual(a["end"], b["start"])
                        forwarded = {e["unit"]: e["end"] for e in events if e["kind"] in ("F", "Fp")}
                        self.assertEqual(set(forwarded), set(plan.forward))
                        for event in events:
                            peers = [e for e in trace if e["unit"] == event["unit"] and e["parent"] == event["parent"]]
                            if event["kind"] in ("F", "Fp") and stage > 0:
                                prior = next(
                                    e for e in peers if e["stage"] == stage - 1 and e["kind"] == event["kind"]
                                )
                                self.assertLessEqual(prior["end"], event["start"])
                            if event["kind"] == "B":
                                if stage < 3:
                                    downstream = next(e for e in peers if e["stage"] == stage + 1 and e["kind"] == "B")
                                    self.assertLessEqual(downstream["end"], event["start"])
                                self.assertLessEqual(forwarded[event["unit"]], event["start"])
                                unit = next(u for u in units if u.unit_id == event["unit"])
                                for other in units:
                                    if unit.groups & other.groups:
                                        self.assertLessEqual(forwarded[other.unit_id], event["start"])

    def test_multiple_parents_restart_chunkflow_pipeline(self):
        chunks = self.cf.scheduler.build_chunks([512], 128, repack_tail=False)
        units = self.cf.units.build_micro_batches(chunks, "staggered")
        plan = self.cf.telemetry.summarize_plan([512], chunks, units, 1 << 30, 4, [4, 4, 4, 3], include_schedule=True)
        prediction = self.cf.schedule_model.predict_step([plan, plan], 4)
        self.assertAlmostEqual(prediction["baseline"]["makespan"], 60)
        self.assertAlmostEqual(prediction["chunkflow"]["makespan"], 42)

    def test_default_telemetry_does_not_enable_timeline_simulation(self):
        chunks = self.cf.scheduler.build_chunks([512], 128, repack_tail=False)
        units = self.cf.units.build_micro_batches(chunks, "staggered")
        plan = self.cf.telemetry.summarize_plan([512], chunks, units, 1 << 30, 4, [4, 4, 4, 3])
        self.assertIsNone(plan.model_plan)
        metrics = self.cf.telemetry.StepPlanTelemetry([plan]).to_metrics(4)
        self.assertNotIn("schedule_model_speedup", metrics)

    def test_schedule_prediction_requires_its_own_metrics(self):
        with TemporaryDirectory() as folder:
            base, cf = Path(folder) / "base.log", Path(folder) / "cf.log"
            timing = "perf 0: {'perf/actor_train_time': 4, 'perf/actor_train_tokens': 100}\n"
            base.write_text(timing)
            theory = (
                "step 0: {'train/chunkflow/equal_unit_fill_drain_baseline_proxy': 4, "
                "'train/chunkflow/equal_unit_fill_drain_chunkflow_proxy': 2}\n"
            )
            cf.write_text(theory + timing)
            with self.assertRaisesRegex(ValueError, "schedule prediction missing"):
                self.align._pair(base, cf, "actor_train_time", "schedule")
            self.assertEqual(self.align._pair(base, cf, "actor_train_time")[0]["theory_gain_pct"], 100)
            with_schedule = theory.replace("proxy': 2}", "proxy': 2, 'train/chunkflow/schedule_model_speedup': 3}")
            cf.write_text(with_schedule + timing)
            self.assertEqual(self.align._pair(base, cf, "actor_train_time", "schedule")[0]["theory_x"], 3)

    def test_prediction_is_available_without_any_gpu_timing(self):
        metrics = self.benefit.step_metrics([28197], 1024, 32768, 4, "staggered", 0, self.cf)
        self.assertGreater(metrics["schedule_model_speedup"], 1)
        self.assertTrue(math.isfinite(metrics["schedule_model_baseline_makespan"]))

    def test_multi_pair_cli_exports_counts_and_scatter(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            base, cf = folder / "base.log", folder / "cf.log"
            base_lines, cf_lines = [], []
            for step, (prediction, actual) in enumerate(((2, 2), (0.5, 2), (2, 0.5), (0.5, 0.5))):
                base_lines.append(f"perf {step}: {{'perf/actor_train_time': 10, 'perf/actor_train_tokens': 100}}")
                cf_lines.append(
                    f"perf {step}: {{'perf/actor_train_time': {10 / actual}, 'perf/actor_train_tokens': 100}}"
                )
                cf_lines.append(
                    f"step {step}: {{'train/chunkflow/equal_unit_fill_drain_baseline_proxy': {prediction}, "
                    "'train/chunkflow/equal_unit_fill_drain_chunkflow_proxy': 1}"
                )
            base.write_text("\n".join(base_lines))
            cf.write_text("\n".join(cf_lines))
            prefix = folder / "report"
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "chunkflow_align.py"),
                    "--baseline",
                    str(base),
                    "--chunkflow",
                    str(cf),
                    "--baseline",
                    str(base),
                    "--chunkflow",
                    str(cf),
                    "--bootstrap-samples",
                    "10",
                    "--output-prefix",
                    str(prefix),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(prefix.with_suffix(".json").read_text())
            self.assertEqual(report["pooled"]["paired_steps"], 8)
            self.assertEqual(report["per_case"]["0"]["counts"]["actual_positive"]["predicted_negative"], 1)
            with prefix.with_suffix(".csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 8)
            self.assertIn("<circle", prefix.with_suffix(".svg").read_text())
            markdown = prefix.with_suffix(".md").read_text()
            self.assertIn("| 实际正收益 | 2 | 2 |", markdown)
            self.assertIn("| 实际负收益 | 2 | 2 |", markdown)

    def test_tail_sweep_selection_does_not_pair_with_older_chunkflow_run(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            for gbs, steps, chunks in ((1, 35, (0, 512, 1024, 2048, 4096)), (4, 20, (0, 1024)), (16, 10, (0, 1024))):
                for i, chunk in enumerate(chunks):
                    name = f"cfbenefit-gbs{gbs}-cap32768-chunk{chunk}-s{steps}-gpu8-2026-10-15-10:00:{i:02d}.log"
                    (folder / name).write_text("")
            pairs = self.align._tail_sweep_pairs(folder)
            self.assertEqual(len(pairs), 6)
            self.assertEqual(len({base for base, _ in pairs}), 3)
            self.assertEqual(len({cf for _, cf in pairs}), 6)
            (folder / "cfbenefit-gbs1-cap32768-chunk0-s35-gpu8-2026-10-15-11:00:00.log").write_text("")
            with self.assertRaisesRegex(ValueError, "missing CF"):
                self.align._tail_sweep_pairs(folder)


if __name__ == "__main__":
    unittest.main()
