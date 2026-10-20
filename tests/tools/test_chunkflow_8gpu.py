# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real experiment driver with local logs, never Ray or GPUs."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "scripts/training/sft/tools"


class EightGpuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tools = self.root / "scripts/training/sft/tools"
        tools.mkdir(parents=True)
        for name in ("chunkflow_8gpu.py", "chunkflow_align.py", "chunkflow_diagnose.py"):
            shutil.copyfile(TOOLS / name, tools / name)
        (tools.parent / "run-qwen3-4B-8xgpu-chunkflow.sh").touch()
        entry = self.root / "scripts/entrypoint/ray-job.sh"
        entry.parent.mkdir()
        entry.write_text(
            '#!/bin/bash\nset -eu\nps -p "$PPID" -o command= >"${LOG_FILE}.parent"\npython3 - <<\'PY\'\n'
            "import os\nfrom pathlib import Path\n"
            "assert os.environ['TP_SIZE'] == '2' and os.environ['PP_SIZE'] == '4'\n"
            "assert os.environ['KEEP_LAST_K'] == '0'\n"
            "chunk = int(os.environ['CHUNK_SIZE'])\n"
            "profile = os.environ['CHUNKFLOW_PROFILE'] == '1'\n"
            "assert os.environ['RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS'] == ('0' if profile else '1')\n"
            "backend = os.environ['RELAX_CHUNKFLOW_FLASH_BACKEND']\n"
            "assert os.environ['RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY'] == '1'\n"
            "gated = bool(os.environ['GATE_GAIN_PCT'])\n"
            "gate_phase = 'cf8-gate-' in os.environ['RUN_NAME']\n"
            "lines = ['chunked attention installed'] if chunk else []\n"
            "if chunk and not os.environ.get('DROP_BACKEND_MARKER'):\n"
            "    lines.append('chunkflow: flash attention backend=' + backend)\n"
            "count = int(os.environ['NUM_ROLLOUT']) - (int(os.environ.get('MISS_LAST', '0')) if chunk else 0)\n"
            "for step in range(count):\n"
            "    seconds = {0: 5, 4096: 4, 8192: 6}[chunk]\n"
            "    if chunk and backend == 'fa3': seconds = 3\n"
            "    prediction = (2.5 if step % 2 == 0 else 2.4) if gate_phase else 1.2\n"
            "    selected = int(prediction > 1 + float(os.environ.get('GATE_GAIN_PCT') or 140) / 100)\n"
            "    if gated and not selected: seconds = 5\n"
            "    tokens = 1000 + step + (1 if chunk and os.environ.get('WRONG_TOKENS') else 0)\n"
            "    lines.append(f\"perf {step}: {{'perf/actor_train_time': {seconds}, "
            "'perf/step_time': {seconds + 2}, 'perf/actor_train_tokens': {tokens}}}\")\n"
            "    if chunk and not profile:\n"
            "        gate = {}\n"
            "        if gated and not os.environ.get('DROP_GATE_MARKER'):\n"
            "            gate = {'train/chunkflow/gate_selected': selected, "
            "'train/chunkflow/gate_threshold_gain_pct': float(os.environ['GATE_GAIN_PCT']), "
            "'train/chunkflow/gate_prediction_time_s': .001, 'train/chunkflow/gate_consensus_time_s': .002}\n"
            "        metrics = {'train/chunkflow/schedule_model_speedup': prediction, "
            "'train/chunkflow/equal_unit_fill_drain_baseline_proxy': 2, "
            "'train/chunkflow/equal_unit_fill_drain_chunkflow_proxy': 1}\n"
            "        metrics.update(gate)\n"
            "        lines.append(f'step {step}: {metrics!r}')\n"
            "Path(os.environ['LOG_FILE']).write_text('\\n'.join(lines))\n"
            "if os.environ.get('MUTATE_DATA'):\n"
            "    Path(os.environ['PROMPT_DATA']).write_text('changed')\n"
            "PY\n"
        )
        model = self.root / "models/Qwen3-4B"
        model.mkdir(parents=True)
        data = self.root / "mixed.parquet"
        data.write_text("fixture")
        self.env = {
            "MODEL_DIR": str(model.parent),
            "PROMPT_DATA": str(data),
            "REPORT_DIR": str(self.root / "report"),
            "STEPS": "30",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        }
        with patch.object(sys, "path", [str(tools), *sys.path]):
            spec = importlib.util.spec_from_file_location("chunkflow_8gpu", tools / "chunkflow_8gpu.py")
            self.tool = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.tool)

    def run_phase(self, phase="screen", **extra):
        env = {**self.env, **extra}
        config = self.tool.configuration(phase, env)

        def execute_shell(executable, argv):
            self.assertEqual(executable, "/bin/bash")
            result = subprocess.run(argv, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)

        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(self.tool, "git_output", return_value="fixture"),
            patch.object(self.tool.os, "execv", side_effect=execute_shell),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.tool.run(config)
        return Path(config["report_dir"])

    def test_screen_generates_two_separate_comparisons_and_rejects_overwrite(self):
        folder = self.run_phase()
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual([a["environment"]["CHUNK_SIZE"] for a in manifest["arms"]], ["0", "4096", "8192"])
        summary = json.loads((folder / "summary.json").read_text())
        results = summary["results"]
        self.assertEqual([r["summary"]["paired_steps"] for r in results], [20, 20])
        self.assertAlmostEqual(results[0]["summary"]["aggregate_actual_x"], 1.25)
        self.assertAlmostEqual(results[0]["step_time_speedup"], 7 / 6)
        self.assertAlmostEqual(results[1]["summary"]["aggregate_actual_x"], 5 / 6)
        self.assertIn("| 实际正收益 | 20 | 0 |", (folder / "summary.md").read_text())
        self.assertIn("| 实际负收益 | 20 | 0 |", (folder / "summary.md").read_text())
        self.assertTrue((folder / "repeat1/chunk4096/diagnosis.json").exists())
        # The actual launcher kills residual Python processes. The waiting
        # orchestrator must already have been replaced with bash execute.sh.
        parent = (folder / "repeat1/baseline.log.parent").read_text().strip()
        self.assertTrue(parent.startswith("bash "), parent)
        self.assertIn("execute.sh", parent)
        with self.assertRaises(FileExistsError):
            self.run_phase()

    def test_confirm_alternates_order_and_pairs_seeds(self):
        folder = self.run_phase("confirm", CHUNK_SIZE="4096")
        manifest = json.loads((folder / "manifest.json").read_text())
        arms = [a["environment"] for a in manifest["arms"]]
        self.assertEqual([a["CHUNK_SIZE"] for a in arms], ["0", "4096", "4096", "0", "0", "4096"])
        self.assertEqual([a["TRAIN_SEED"] for a in arms], ["1235", "1235", "1236", "1236", "1237", "1237"])
        self.assertEqual(len(json.loads((folder / "summary.json").read_text())["results"]), 3)

    def test_profile_does_not_generate_performance_claims(self):
        folder = self.run_phase("profile", CHUNK_SIZE="4096", STEPS="14")
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["arms"]), 2)
        self.assertTrue((folder / "profile.md").exists())
        self.assertFalse((folder / "summary.json").exists())
        self.assertFalse(list(folder.glob("**/attribution.csv")))

    def test_profile_gate_collects_baseline_and_real_adaptive_path(self):
        folder = self.run_phase(
            "profile",
            CHUNK_SIZE="4096",
            STEPS="16",
            PROFILE_START="12",
            PROFILE_END="14",
            PROFILE_GATE="1",
            TRAIN_SEED="1235",
        )
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual([a["gated"] for a in manifest["arms"]], [False, True])
        self.assertEqual(manifest["arms"][1]["environment"]["GATE_GAIN_PCT"], "140.0")
        self.assertEqual(manifest["arms"][1]["backend"], "fa3")
        self.assertIn("chunk4096-fa3-gate140", (folder / "profile.md").read_text())
        self.assertFalse((folder / "summary.json").exists())
        with self.assertRaisesRegex(ValueError, "only supported"):
            self.tool.configuration("gate", {**self.env, "CHUNK_SIZE": "4096", "PROFILE_GATE": "1"})

    def test_backend_phase_shares_baseline_and_reports_direct_fa3_gain(self):
        folder = self.run_phase("backend", CHUNK_SIZE="4096")
        manifest = json.loads((folder / "manifest.json").read_text())
        arms = [a["environment"] for a in manifest["arms"]]
        self.assertEqual([a["CHUNK_SIZE"] for a in arms], ["0", "4096", "4096"])
        self.assertEqual([a["RELAX_CHUNKFLOW_FLASH_BACKEND"] for a in arms], ["fa2", "fa2", "fa3"])
        self.assertEqual({a["TRAIN_SEED"] for a in arms}, {"1234"})
        report = json.loads((folder / "summary.json").read_text())
        self.assertEqual([r["flash_backend"] for r in report["results"]], ["fa2", "fa3"])
        self.assertAlmostEqual(report["backend_comparison"]["fa3_vs_fa2_x"], 4 / 3)
        for backend in ("fa2", "fa3"):
            self.assertTrue((folder / f"repeat1/chunk4096-{backend}/attribution.csv").exists())
        with self.assertRaisesRegex(RuntimeError, "not confirmed"):
            self.run_phase(
                "backend", CHUNK_SIZE="4096", REPORT_DIR=str(self.root / "missing-backend"), DROP_BACKEND_MARKER="1"
            )

    def test_training_script_passes_seed_and_profiler_window_to_ray(self):
        binary = self.root / "bin"
        binary.mkdir()
        capture = self.root / "ray-arguments.json"
        ray = binary / "ray"
        ray.write_text(
            f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
            "Path(os.environ['CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
            "print('chunked attention installed')\n"
        )
        ray.chmod(0o755)
        configs = self.root / "model-configs"
        configs.mkdir()
        (configs / "qwen3-4B.sh").write_text("MODEL_ARGS=()\n")
        env = {
            **self.env,
            "PATH": str(binary) + os.pathsep + self.env["PATH"],
            "RELAX_ENTRYPOINT_MODE": "ray-job",
            "MODEL_CONFIG_DIR": str(configs),
            "RUNTIME_ENV_JSON": "{}",
            "CAPTURE_ARGS": str(capture),
            "CHUNK_SIZE": "4096",
            "RELAX_CHUNKFLOW_FLASH_BACKEND": "fa3",
            "RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY": "1",
            "CHUNKFLOW_PROFILE": "1",
            "PROFILE_START": "2",
            "PROFILE_END": "4",
            "NUM_ROLLOUT": "6",
            "TRAIN_SEED": "1237",
            "RUN_NAME": "fixture-profile",
            "LOG_FILE": str(self.root / "train.log"),
        }
        command = ["bash", str(TOOLS.parent / "run-qwen3-4B-8xgpu-chunkflow.sh"), "chunkflow"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(capture.read_text())
        runtime = (
            json.loads(args[args.index("--runtime-env-json") + 1])
            if "--runtime-env-json" in args
            else json.loads(next(a.split("=", 1)[1] for a in args if a.startswith("--runtime-env-json=")))
        )
        self.assertEqual(runtime["env_vars"]["RELAX_CHUNKFLOW_FLASH_BACKEND"], "fa3")
        self.assertEqual(runtime["env_vars"]["RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY"], "1")
        self.assertIn("--use-pytorch-profiler", args)
        for flag, value in {
            "--profile-target": "train_overall",
            "--profile-step-start": "2",
            "--profile-step-end": "4",
            "--seed": "1237",
            "--tb-experiment-name": "fixture-profile",
            "--tensor-model-parallel-size": "2",
            "--pipeline-model-parallel-size": "4",
        }.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        plain = subprocess.run(command, env={**env, "CHUNKFLOW_PROFILE": "0"}, capture_output=True, text=True)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertNotIn("--use-pytorch-profiler", json.loads(capture.read_text()))
        adaptive = subprocess.run(
            [*command[:-1], "adaptive"],
            env={**env, "CHUNKFLOW_PROFILE": "0", "GATE_GAIN_PCT": "140"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(adaptive.returncode, 0, adaptive.stderr)
        args = json.loads(capture.read_text())
        runtime = (
            json.loads(args[args.index("--runtime-env-json") + 1])
            if "--runtime-env-json" in args
            else json.loads(next(a.split("=", 1)[1] for a in args if a.startswith("--runtime-env-json=")))
        )
        self.assertEqual(runtime["env_vars"]["RELAX_CHUNKFLOW_GATE_GAIN_PCT"], "140")

    def test_gate_phase_runs_baseline_always_and_online_selected_with_new_seed(self):
        folder = self.run_phase("gate", CHUNK_SIZE="4096")
        manifest = json.loads((folder / "manifest.json").read_text())
        arms = manifest["arms"]
        self.assertEqual([a["gated"] for a in arms], [False, False, True])
        self.assertEqual({a["environment"]["TRAIN_SEED"] for a in arms}, {"1235"})
        report = json.loads((folder / "summary.json").read_text())
        self.assertEqual(len(report["results"]), 2)
        gated = report["results"][1]
        self.assertEqual(gated["gate"]["selected_steps"], 10)
        self.assertEqual(gated["gate"]["fallback_steps"], 10)
        self.assertAlmostEqual(gated["summary"]["aggregate_actual_x"], 1.25)
        self.assertTrue((folder / "repeat1/chunk4096-fa3-gate140/attribution.csv").exists())
        with self.assertRaisesRegex(RuntimeError, "missing pre-step"):
            self.run_phase("gate", CHUNK_SIZE="4096", REPORT_DIR=str(self.root / "missing-gate"), DROP_GATE_MARKER="1")

    def test_rejects_partial_runs_and_token_mismatch_before_next_arm(self):
        for key, error in (
            ("MISS_LAST", "expected step IDs"),
            ("WRONG_TOKENS", "different token counts"),
            ("MUTATE_DATA", "dataset changed"),
        ):
            with self.subTest(key=key):
                folder = self.root / key
                with self.assertRaisesRegex(RuntimeError, error):
                    self.run_phase(REPORT_DIR=str(folder), **{key: "1"})
                manifest = json.loads((folder / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["arms"][-1]["status"], "pending")
                self.assertFalse((folder / "summary.md").exists())

    def test_plan_is_read_only_and_unsafe_configuration_is_rejected(self):
        folder = self.run_phase(DRY_RUN="1", MODEL_DIR="/nonexistent/models", PROMPT_DATA="/nonexistent/data")
        self.assertFalse(folder.exists())
        for extra in (
            {"RAY_NO_WAIT": "1"},
            {"PP_SIZE": "2"},
            {"KEEP_LAST_K": "1"},
            {"CHUNKFLOW_PROFILE": "1"},
            {"CHUNK_SIZES": "4096 4096"},
            {"STEPS": "11"},
            {"RELAX_CHUNKFLOW_FLASH_BACKEND": "auto"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.tool.configuration("screen", {**self.env, **extra})
        with self.assertRaisesRegex(ValueError, "requires CHUNK_SIZE"):
            self.tool.configuration("confirm", self.env)
        with self.assertRaisesRegex(ValueError, "flush traces"):
            self.tool.configuration("profile", {**self.env, "CHUNK_SIZE": "4096", "STEPS": "13"})


if __name__ == "__main__":
    unittest.main()
