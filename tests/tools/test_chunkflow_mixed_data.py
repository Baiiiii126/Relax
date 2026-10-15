# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check synthetic length control without downloading a tokenizer."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


BUILDER = Path(__file__).parents[2] / "scripts/training/sft/tools/build_chunkflow_longtail_sft.py"


class MixedDataTests(unittest.TestCase):
    def test_ultralong_samples_are_a_subset_of_tail_with_reproducible_lengths(self):
        spec = importlib.util.spec_from_file_location("builder", BUILDER)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        targets = dict(zip((bucket[0] for bucket in builder.BUCKETS), (7, 2, 1, 4)))
        with patch.object(
            builder, "_token_count", side_effect=lambda _, question, answer: len(question) + len(answer)
        ):
            rows = builder._build_synthetic_rows(targets, None, 1234, ultralong_count=2)
            repeated = builder._build_synthetic_rows(targets, None, 1234, ultralong_count=2)
        self.assertEqual(rows, repeated)
        for bucket, lower, upper in builder.BUCKETS:
            self.assertEqual(len(rows[bucket]), targets[bucket])
            self.assertTrue(all(lower <= row["token_count"] < upper for row in rows[bucket]))
        tail = rows["tail_16k_32k"]
        self.assertEqual(sum(row["token_count"] >= 28_672 for row in tail), 2)
        with self.assertRaises(ValueError):
            builder._build_synthetic_rows(targets, None, 1234, ultralong_count=5)


if __name__ == "__main__":
    unittest.main()
