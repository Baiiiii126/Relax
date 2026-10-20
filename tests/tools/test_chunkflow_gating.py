# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only gate checks: no training, framework imports, or GPU required."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[2]


class GatingTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "benefit_model", ROOT / "scripts/training/sft/tools/chunkflow_benefit_model.py"
        )
        self.benefit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.benefit)
        self.cf = self.benefit._load_chunkflow(ROOT)
        name = "relax.backends.megatron.chunkflow.gating"
        spec = importlib.util.spec_from_file_location(name, ROOT / "relax/backends/megatron/chunkflow/gating.py")
        self.gate = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.gate
        spec.loader.exec_module(self.gate)

    def test_forecast_peeks_current_step_without_consuming_data(self):
        lengths = [700, 1100, 28160]
        iterator = SimpleNamespace(
            rollout_data={"total_lengths": lengths},
            micro_batch_indices=[[0], [1], [2]],
            offset=1,
            dummy_micro_batch_offsets=set(),
        )
        metrics, fingerprint = self.gate.forecast_step(
            iterator, 1, chunk_size=512, pp_size=4, tp_size=2, pad_multiplier=128
        )
        reference = self.benefit.step_metrics([1100], 512, 32768, 4, "staggered", 0, self.cf)
        for key in ("schedule_model_speedup", "source_rows", "input_rows", "parent_padding_rows", "num_units"):
            self.assertEqual(metrics[key], reference[key])
        self.assertEqual(iterator.offset, 1)
        self.assertEqual(iterator.rollout_data["total_lengths"], lengths)
        again, next_fingerprint = self.gate.forecast_step(
            iterator, 2, chunk_size=512, pp_size=4, tp_size=2, pad_multiplier=128
        )
        self.assertEqual(again["num_microbatches"], 2)
        self.assertNotEqual(fingerprint, next_fingerprint)
        self.assertEqual(iterator.offset, 1)

    def test_140_percent_gain_means_strictly_more_than_2_point_4_x(self):
        threshold = self.gate.gain_threshold("140")
        boundary = 1 + threshold / 100
        self.assertEqual(boundary, 2.4)
        self.assertFalse(2.4 > boundary)
        self.assertFalse(1.4 > boundary)
        self.assertTrue(2.4001 > boundary)
        self.assertIsNone(self.gate.gain_threshold(""))
        for value in ("nan", "inf", "-1", "oops"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gate.gain_threshold(value)

    def test_rejects_unsupported_or_incomplete_inputs(self):
        for iterator in (
            SimpleNamespace(),
            SimpleNamespace(micro_batch_indices=[[0]], rollout_data={"total_lengths": [8]}, offset=1),
            SimpleNamespace(micro_batch_indices=[[0]], rollout_data={"total_lengths": [0]}, offset=0),
            SimpleNamespace(
                micro_batch_indices=[[0]], rollout_data={"total_lengths": [8]}, offset=0, dummy_micro_batch_offsets={0}
            ),
        ):
            with self.subTest(iterator=iterator), self.assertRaises(ValueError):
                self.gate.forecast_step(iterator, 1, chunk_size=4096, pp_size=4, tp_size=2, pad_multiplier=128)


if __name__ == "__main__":
    unittest.main()
