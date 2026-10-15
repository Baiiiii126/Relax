# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify the experiment driver with a local log producer, without Ray or
GPUs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "scripts/training/sft/tools"


class AttributionRunnerTests(unittest.TestCase):
    def test_focused_run_exports_only_current_paired_steps(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tools = folder / "scripts/training/sft/tools"
            tools.mkdir(parents=True)
            for name in ("run_chunkflow_attribution.sh", "chunkflow_align.py"):
                shutil.copyfile(TOOLS / name, tools / name)
            entrypoint = folder / "scripts/entrypoint/ray-job.sh"
            entrypoint.parent.mkdir()
            # This replaces the launcher only in the temporary fixture.
            entrypoint.write_text(
                "#!/bin/bash\nset -eu\npython3 - <<'PY'\n"
                "import os\nfrom pathlib import Path\n"
                "assert os.environ['GBS'] == os.environ.get('EXPECTED_GBS', '8')\n"
                "assert os.environ['RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS'] == '1'\n"
                "assert os.environ['TP_SIZE'] == '2' and os.environ['PP_SIZE'] == '4'\n"
                "lines = []\n"
                "for step in range(int(os.environ['NUM_ROLLOUT'])):\n"
                "    cf = os.environ['CHUNK_SIZE'] == '2048'\n"
                "    time = (4 if step % 2 else 6) if cf else 5\n"
                "    lines.append(f\"perf {step}: {{'perf/actor_train_time': {time}, "
                "'perf/actor_train_tokens': 100}}\")\n"
                "    if cf:\n"
                "        lines.append(f\"step {step}: {{'train/chunkflow/schedule_model_speedup': 1.2, "
                "'train/chunkflow/equal_unit_fill_drain_baseline_proxy': 2, "
                "'train/chunkflow/equal_unit_fill_drain_chunkflow_proxy': 1}}\")\n"
                "Path(os.environ['LOG_FILE']).write_text('\\n'.join(lines))\nPY\n"
            )
            model = folder / "models/Qwen3-4B"
            model.mkdir(parents=True)
            data = folder / "data/chunkflow-mixed/train.parquet"
            data.parent.mkdir(parents=True)
            data.touch()
            environment = {
                **os.environ,
                "MODEL_DIR": str(model.parent),
                "DATA_DIR": str(data.parents[1]),
                "STEPS": "14",
                "DROP_FIRST": "10",
                "REPORT_DIR": str(folder / "reports/current"),
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            }
            for key in ("DRY_RUN", "RAY_NO_WAIT", "PROMPT_DATA", "GBS", "EXPECTED_GBS"):
                environment.pop(key, None)
            command = ["bash", str(tools / "run_chunkflow_attribution.sh")]
            dry = subprocess.run(command, env={**environment, "DRY_RUN": "1"}, capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse((folder / "reports").exists())
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report_dir = folder / "reports/current"
            report = json.loads((report_dir / "attribution.json").read_text())
            self.assertEqual(report["theory_source"], "schedule")
            self.assertEqual(report["pooled"]["paired_steps"], 4)
            self.assertIn("| 实际正收益 | 2 | 0 |", (report_dir / "attribution.md").read_text())
            self.assertIn("| 实际负收益 | 2 | 0 |", (report_dir / "attribution.md").read_text())
            for suffix in ("csv", "svg", "json", "md"):
                self.assertTrue((report_dir / f"attribution.{suffix}").exists())
            rerun = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("REPORT_DIR must be a new directory", rerun.stderr)
            changed = {
                **environment,
                "GBS": "4",
                "EXPECTED_GBS": "4",
                "REPORT_DIR": str(folder / "reports/gbs4"),
            }
            override = subprocess.run(command, env=changed, capture_output=True, text=True)
            self.assertEqual(override.returncode, 0, override.stderr)
            self.assertIn("GBS=4", override.stdout)


if __name__ == "__main__":
    unittest.main()
