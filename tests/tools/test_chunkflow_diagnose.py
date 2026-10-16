# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for observational diagnosis and model negative controls."""

from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "scripts/training/sft/tools"


def load_tool(name: str):
    with patch.object(sys, "path", [str(TOOLS), *sys.path]):
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool("chunkflow_diagnose")

    def rows(self, values):
        rows = []
        for step, (prediction, actual, m) in enumerate(values):
            rows.append(
                {
                    "step": step,
                    "case": 0,
                    "theory_x": prediction,
                    "actual_x": actual,
                    "theory_gain_pct": 100 * (prediction - 1),
                    "actual_gain_pct": 100 * (actual - 1),
                    "base_time": 10,
                    "cf_time": 10 / actual,
                    "ideal_cf_time": 10 / prediction,
                    "tokens": 1000 * (step + 1),
                    "M": m,
                    "U": 4 * m,
                    "excess_fraction": 1 / actual - 1 / prediction,
                }
            )
        return rows

    def test_constant_positive_sign_does_not_hide_ranking_signal(self):
        rows = self.rows([(1.1, 0.7, 1), (1.2, 0.8, 1), (1.3, 1.1, 1), (1.4, 1.2, 1)])
        result = self.tool.summarize(rows, 0)
        self.assertIsNone(result["sign_correlation"])
        self.assertEqual(result["predicted_positive_precision"], 0.5)
        self.assertEqual(result["actual_win_auroc"], 1)
        self.assertGreater(result["pearson"], 0)
        self.assertEqual(self.tool.auc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5)
        self.assertIsNone(self.tool.auc([1, 2], [1, 1]))

    def test_grouping_exposes_m_confounding_and_keeps_ties(self):
        rows = self.rows([(1.1, 1.1, 1), (1.2, 1.2, 1), (1.3, 1.3, 1), (2.1, 0.7, 2), (2.2, 0.8, 2), (2.3, 0.9, 2)])
        result = self.tool.diagnose(rows, 0, 20, 2, 1234)["cases"]["0"]
        self.assertLess(result["summary"]["pearson"], 0)
        for group in result["by_M"].values():
            self.assertAlmostEqual(group["pearson"], 1)
        tied = self.rows([(1.2, 0.9, 1)] * 8)
        self.assertEqual(len(self.tool.grouped(tied, "theory_gain_pct", 0)), 1)

    def test_block_bootstrap_handles_constants_and_short_or_gapped_series(self):
        rows = self.rows([(1.2, 0.8, 1)] * 20)
        result = self.tool.block_intervals(rows, 20, 3, 1)
        self.assertTrue(result["available"])
        self.assertIsNone(result["pearson"])
        self.assertAlmostEqual(result["actual_x"][0], 0.8)
        self.assertFalse(self.tool.block_intervals(rows[:3], 20, 3, 1)["available"])
        rows[-1]["step"] += 1
        self.assertFalse(self.tool.block_intervals(rows, 20, 3, 1)["available"])

    def test_linear_token_control_removes_common_length_trend(self):
        rows = self.rows([(1 + (3 * t + t % 2) / 100, 1 + (-4 * t + 2 * (t % 2)) / 100, 1) for t in range(10)])
        self.assertLess(self.tool.summarize(rows, 0)["pearson"], 0)
        self.assertAlmostEqual(self.tool.token_adjusted_correlation(rows), 1)

    def test_csv_rejects_duplicate_steps_and_inconsistent_timings(self):
        rows = self.rows([(1.2, 0.8, 1)] * 4)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "attribution.csv"

            def write():
                with path.open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            write()
            self.assertEqual(len(self.tool.read_rows(path)), 4)
            rows[1]["step"] = 0
            write()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.tool.read_rows(path)
            rows[1]["step"] = 1
            rows[1]["actual_x"] = 1
            write()
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                self.tool.read_rows(path)

    def test_model_negative_control_and_paper_nonconformance_are_reported(self):
        report = load_tool("chunkflow_model_audit").audit(ROOT)
        controls = {row["name"]: row["model_speedup"] for row in report["runtime_model_controls"]}
        self.assertGreater(controls["single_long_parent"], 1)
        self.assertLess(controls["many_unsplit_parents"], 1)
        self.assertLess(report["paper_length_examples"][-1]["model_speedup"], 1)
        self.assertFalse(report["paper_length_examples"][0]["bubble_matches_to_0_002"])


if __name__ == "__main__":
    unittest.main()
