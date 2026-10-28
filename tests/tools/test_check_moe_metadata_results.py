# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class MoEMetadataResultTests(unittest.TestCase):
    def test_check_rejects_numerical_drift_memory_growth_and_missing_ranks(self):
        directory = Path(__file__).resolve().parents[2] / "scripts/training/sft/tools"
        with patch.object(sys, "path", [str(directory), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "check_metadata", directory / "check_moe_metadata_results.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            base, candidate = Path(root) / "off", Path(root) / "b"
            for folder, mode in ((base, "off"), (candidate, "b")):
                folder.mkdir()
                for arm in ("baseline", "chunkflow"):
                    (folder / f"{arm}.log").write_text(
                        "step 0: {'train/loss': 1.0, 'train/grad_norm': 2, 'perf/actor_train_tokens': 8192}\n"
                    )
                    memory = folder / arm / "memory"
                    memory.mkdir(parents=True)
                    row = dict(
                        rank="rank0",
                        rollout_id=0,
                        step_id=0,
                        moe_metadata_mode=mode,
                        peak_allocated=100,
                        peak_reserved=200,
                        before=dict(peak_allocated_since_previous_reset=110, peak_reserved_since_previous_reset=210),
                    )
                    (memory / "memory_rank0.jsonl").write_text(json.dumps(row) + "\n")

            def run(ranks=1):
                return module.check(base, candidate, "b", 1, ranks, 0, 0)

            self.assertEqual(run(), [])
            self.assertTrue(run(ranks=8))
            path = candidate / "chunkflow/memory/memory_rank0.jsonl"
            row = json.loads(path.read_text())
            row["peak_allocated"] = 101
            path.write_text(json.dumps(row) + "\n")
            self.assertTrue(any("peak_allocated" in issue for issue in run()))
            log = candidate / "baseline.log"
            log.write_text(log.read_text().replace("1.0", "1.01"))
            self.assertTrue(any("train/loss" in issue for issue in run()))


if __name__ == "__main__":
    unittest.main()
