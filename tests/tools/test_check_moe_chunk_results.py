# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


class MoEChunkReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[2] / "scripts/training/sft/tools"
        with patch.object(sys, "path", [str(directory), *sys.path]):
            spec = importlib.util.spec_from_file_location("moe_chunk_report", directory / "check_moe_chunk_results.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("off", "off_repeat", "dedup", "expert", "both"):
            mode = "off" if name == "off_repeat" else name
            folder = self.root / "pilot/repeat0" / name
            folder.mkdir(parents=True)
            (folder / "comparison.md").write_text(
                f"MoE chunk optimization: {mode} (test).\nbaseline: complete\nchunkflow: complete\n"
            )
            for arm in ("baseline", "chunkflow"):
                (folder / f"{arm}.log").write_text(
                    "".join(
                        f"step {step}: {{'train/loss': 1.0, 'train/grad_norm': 2.0, 'perf/actor_train_tokens': 8192, "
                        "'perf/actor_train_time': 1.0, 'perf/step_time': 2.0}\n"
                        for step in range(2)
                    )
                )
                memory = folder / arm / "memory"
                memory.mkdir(parents=True)
                for rank in range(8):
                    records = [
                        dict(
                            rank=f"rank{rank}_dp0_tp{rank % 2}_pp{rank // 2}",
                            rollout_id=step,
                            step_id=0,
                            arm=arm,
                            moe_metadata_mode="off",
                            moe_chunk_mode=mode,
                            peak_allocated=100,
                            peak_reserved=200,
                            before=dict(
                                peak_allocated_since_previous_reset=100, peak_reserved_since_previous_reset=200
                            ),
                        )
                        for step in range(2)
                    ]
                    (memory / f"memory_rank{rank}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                start = datetime(2026, 10, 29, 12).timestamp()
                (folder / arm / "device-memory.start").write_text(str(start + 0.1))
                (folder / arm / "device-memory.end").write_text(str(start + 0.9))
                (folder / arm / "device-memory.csv").write_text(
                    "".join(f"2026/10/29 12:00:0{second}.000, {rank}, 500\n" for second in (0, 1) for rank in range(8))
                )

    def run_report(self):
        return self.module.report(self.root, "pilot", 2, 1)

    def test_complete_identical_evidence_passes(self):
        issues, text = self.run_report()
        self.assertEqual(issues, [])
        self.assertIn("recorded checks passed", text)

    def test_off_repeat_drift_is_not_attributed_to_candidate(self):
        path = self.root / "pilot/repeat0/off_repeat/baseline.log"
        path.write_text(path.read_text().replace("'train/loss': 1.0", "'train/loss': 1.01", 1))
        issues, _ = self.run_report()
        self.assertTrue(any("off_repeat vs off/baseline: train/loss" in i for i in issues))

    def test_same_last_loss_does_not_hide_earlier_drift(self):
        path = self.root / "pilot/repeat0/both/chunkflow.log"
        path.write_text(path.read_text().replace("'train/loss': 1.0", "'train/loss': 1.02", 1))
        issues, _ = self.run_report()
        self.assertTrue(any("both vs off/chunkflow: train/loss" in i for i in issues))

    def test_per_rank_memory_increase_is_rejected(self):
        path = self.root / "pilot/repeat0/dedup/chunkflow/memory/memory_rank3.jsonl"
        path.write_text(path.read_text().replace('"peak_allocated": 100', '"peak_allocated": 101', 1))
        issues, _ = self.run_report()
        self.assertTrue(any("dedup vs off/chunkflow" in i and "allocator increases" in i for i in issues))

    def test_missing_rank_or_wrong_mode_fails_closed(self):
        path = self.root / "pilot/repeat0/expert/chunkflow/memory/memory_rank7.jsonl"
        path.unlink()
        issues, _ = self.run_report()
        self.assertTrue(any("incomplete per-rank memory" in i for i in issues))

    def test_sampled_external_memory_growth_is_reported(self):
        path = self.root / "pilot/repeat0/dedup/chunkflow/device-memory.csv"
        path.write_text(path.read_text().replace(", 3, 500", ", 3, 501"))
        issues, _ = self.run_report()
        self.assertTrue(any("GPU 3 sampled memory" in i for i in issues))

    def test_missing_repeat_control_fails(self):
        (self.root / "pilot/repeat0/off_repeat/comparison.md").unlink()
        issues, _ = self.run_report()
        self.assertTrue(issues)

    def test_sampler_stopping_early_is_rejected(self):
        path = self.root / "pilot/repeat0/dedup/chunkflow/device-memory.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:8]) + "\n")
        issues, _ = self.run_report()
        self.assertTrue(any("incomplete device-memory time coverage" in i for i in issues))

    def make_formal_runs(self):
        folder = self.root / "formal/repeat0"
        shutil.copytree(self.root / "pilot/repeat0", folder)
        for mode in self.module.MODES:
            for arm in self.module.ARMS:
                path = folder / mode / f"{arm}.log"
                row = path.read_text().splitlines()[0].split(": ", 1)[1]
                step_time = 2.0 if mode == "off" else (1.9 if arm == "baseline" else 1.8)
                row = row.replace("'perf/step_time': 2.0", f"'perf/step_time': {step_time}")
                path.write_text("".join(f"step {step}: {row}\n" for step in range(31)))
                for path in (folder / mode / arm / "memory").glob("*.jsonl"):
                    row = json.loads(path.read_text().splitlines()[0])
                    path.write_text("".join(json.dumps(dict(row, rollout_id=s)) + "\n" for s in range(31)))
        return folder

    def test_formal_positive_end_to_end_comparisons_pass(self):
        self.make_formal_runs()
        issues, _ = self.module.report(self.root, "formal", 31, 1)
        self.assertEqual(issues, [])

    def test_training_time_gain_cannot_hide_step_time_regression(self):
        folder = self.make_formal_runs()
        path = folder / "both/chunkflow.log"
        path.write_text(path.read_text().replace("'perf/step_time': 1.8", "'perf/step_time': 2.1"))
        issues, _ = self.module.report(self.root, "formal", 31, 1)
        self.assertTrue(any("both: positive end-to-end gain not reproduced" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
