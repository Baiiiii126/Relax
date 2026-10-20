# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for core planning, gating and FA3 adapters; no training
imports."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).parents[3]


def _load_chunkflow() -> SimpleNamespace:
    """Load CPU modules in a temporary namespace, restoring existing
    imports."""
    package = "relax.backends.megatron.chunkflow"
    folder = ROOT / "relax/backends/megatron/chunkflow"
    modules = {}
    with patch.dict(sys.modules):
        for name in ("relax", "relax.backends", "relax.backends.megatron", package):
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        for leaf in ("scheduler", "units", "schedule_model", "telemetry", "gating", "flash_backends"):
            name = f"{package}.{leaf}"
            spec = importlib.util.spec_from_file_location(name, folder / f"{leaf}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[leaf] = module
    return SimpleNamespace(**modules)


class GatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = _load_chunkflow().gating

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
        expected = {"source_rows": 1100, "input_rows": 1280, "parent_padding_rows": 180, "num_units": 4}
        for key, value in expected.items():
            self.assertEqual(metrics[key], value)
        self.assertAlmostEqual(metrics["schedule_model_speedup"], 12 / 7)
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

    def test_remote_metrics_preserves_training_fields_and_full_local_diagnostics(self):
        for prefix in ("train/chunkflow/", "train/critic-chunkflow/"):
            local = {
                "train/step": 12,
                "train/loss": 0.2,
                "train/grad_norm": 1.5,
                "perf/actor_train_time": 3.0,
                prefix + "gate_selected": 0,
                prefix + "schedule_model_speedup": 2.3,
            }
            before = dict(local)
            remote = self.gate.remote_metrics(local, prefix)
            self.assertEqual(
                remote, {"train/step": 12, "train/loss": 0.2, "train/grad_norm": 1.5, "perf/actor_train_time": 3.0}
            )
            self.assertEqual(local, before)


class ScheduleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cf = _load_chunkflow()

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


class FlashBackendTests(unittest.TestCase):
    def setUp(self):
        self.backends = _load_chunkflow().flash_backends

    def tearDown(self):
        self.backends.flash3_varlen_ops.cache_clear()

    def test_selector_defaults_to_fa2_and_rejects_ambiguous_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.backends.flash_backend_name(), "fa2")
        with patch.dict(os.environ, {"RELAX_CHUNKFLOW_FLASH_BACKEND": "auto"}):
            with self.assertRaises(ValueError):
                self.backends.flash_backend_name()

    def test_adapter_preserves_lse_maxima_causal_and_gradient_buffers(self):
        native = ModuleType("flash_attn_3.flash_attn_interface")
        output, lse = object(), object()
        native._flash_attn_forward = Mock(return_value=(output, lse, None, None))
        native._flash_attn_backward = Mock()
        with patch.dict(sys.modules, {"flash_attn_3.flash_attn_interface": native}):
            forward, backward = self.backends.flash3_varlen_ops()
        q, k, v, cu_q, cu_kv = (object() for _ in range(5))
        got = forward(q, k, v, cu_q, cu_kv, 7, 11, 0, 0.125, True)
        self.assertIs(got[0], output)
        self.assertIs(got[1], lse)
        f = native._flash_attn_forward.call_args
        self.assertEqual(f.args, (q, k, v))
        self.assertEqual((f.kwargs["max_seqlen_q"], f.kwargs["max_seqlen_k"]), (7, 11))
        self.assertIs(f.kwargs["cu_seqlens_q"], cu_q)
        self.assertIs(f.kwargs["cu_seqlens_k"], cu_kv)
        self.assertTrue(f.kwargs["causal"])
        dout, dq, dk, dv = (object() for _ in range(4))
        backward(dout, q, k, v, output, lse, dq, dk, dv, cu_q, cu_kv, 7, 11, 0, 0.125, True, -1, -1, 0, None, False)
        b = native._flash_attn_backward.call_args
        self.assertEqual(b.args, (dout, q, k, v, output, lse))
        self.assertTrue(b.kwargs["is_causal"])
        self.assertEqual((b.kwargs["max_seqlen_q"], b.kwargs["max_seqlen_k"]), (7, 11))
        for key, value in (("dq", dq), ("dk", dk), ("dv", dv)):
            self.assertIs(b.kwargs[key], value)
        with self.assertRaises(ValueError):
            forward(q, k, v, cu_q, cu_kv, 7, 11, 0.1, 0.125, True)

    def test_explicit_fa3_missing_module_fails_without_top_level_fallback(self):
        with patch.dict(
            sys.modules,
            {"flash_attn_3.flash_attn_interface": None, "flash_attn_interface": ModuleType("flash_attn_interface")},
        ):
            with self.assertRaisesRegex(RuntimeError, "explicitly requested fa3"):
                self.backends.flash3_varlen_ops()


class LaunchScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        (self.folder / "models with spaces/Qwen3-4B").mkdir(parents=True)
        (self.folder / "train data.parquet").touch()
        binary = self.folder / "bin"
        binary.mkdir()
        ray = binary / "ray"
        ray.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "Path(os.environ['CF_CAPTURE']).write_text(json.dumps(args))\n"
            "if '--use-chunkflow' in args:\n"
            "    print('chunkflow: chunked attention installed on 9 layer(s)')\n"
            "    print('chunkflow: flash attention backend=' + os.environ['RELAX_CHUNKFLOW_FLASH_BACKEND'])\n"
            "if os.environ.get('RELAX_CHUNKFLOW_GATE_GAIN_PCT'):\n"
            "    print('chunkflow gate before forward: selected=baseline')\n"
            "if not os.environ.get('CF_INCOMPLETE'):\n"
            "    steps = int(args[args.index('--num-rollout') + 1])\n"
            "    print(f'Actor training completed step {steps - 1}/{steps}')\n"
        )
        ray.chmod(0o755)
        self.env = {
            "PATH": f"{binary}:{Path(sys.executable).parent}:{os.defpath}",
            "MODEL_DIR": str(self.folder / "models with spaces"),
            "PROMPT_DATA": str(self.folder / "train data.parquet"),
            "RELAX_ENTRYPOINT_MODE": "test",
            "RUNTIME_ENV_JSON": '{"env_vars":{"EXISTING":"kept"}}',
            "LOG_FILE": str(self.folder / "train.log"),
            "CF_CAPTURE": str(self.folder / "args.json"),
            "RELAX_CHUNKFLOW_GATE_GAIN_PCT": "999",
        }

    def launch(self, *modes, **overrides):
        return subprocess.run(
            ["bash", str(ROOT / "scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh"), *modes],
            env=self.env | overrides,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_modes_preserve_layout_and_forward_worker_settings(self):
        for mode in ("baseline", "chunkflow", "adaptive"):
            with self.subTest(mode=mode):
                result = self.launch(mode, RAY_NO_WAIT="0", WORKING_DIR=str(self.folder / "work dir"))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = json.loads(Path(self.env["CF_CAPTURE"]).read_text())
                self.assertNotIn("--no-wait", args)
                self.assertEqual(args[args.index("--working-dir") + 1], str(self.folder / "work dir"))
                self.assertEqual(args[args.index("--prompt-data") + 1], self.env["PROMPT_DATA"])
                self.assertEqual(args[args.index("--num-rollout") + 1], "100")
                self.assertEqual(args[args.index("--global-batch-size") + 1], "8")
                self.assertEqual(args[args.index("--tensor-model-parallel-size") + 1], "2")
                self.assertEqual(args[args.index("--pipeline-model-parallel-size") + 1], "4")
                self.assertEqual(json.loads(args[args.index("--resource") + 1])["actor"], [1, 8])
                self.assertEqual("--use-chunkflow" in args, mode != "baseline")
                if mode != "baseline":
                    self.assertEqual(args[args.index("--chunkflow-chunk-size") + 1], "4096")
                runtime = json.loads(next(a.split("=", 1)[1] for a in args if a.startswith("--runtime-env-json=")))
                worker = runtime["env_vars"]
                self.assertEqual(worker["EXISTING"], "kept")
                self.assertEqual(worker["RELAX_CHUNKFLOW_FLASH_BACKEND"], "fa3")
                self.assertEqual(worker["RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY"], "1")
                self.assertEqual(worker["RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS"], "1")
                self.assertEqual(worker["RELAX_CHUNKFLOW_GATE_GAIN_PCT"], "140" if mode == "adaptive" else "")

    def test_invalid_configuration_never_submits(self):
        cases = [
            (("chunkflow",), {"PP_SIZE": "0"}),
            (("chunkflow",), {"TP_SIZE": "1"}),
            (("chunkflow",), {"CHUNK_SIZE": "0"}),
            (("adaptive",), {"KEEP_LAST_K": "1"}),
            (("adaptive",), {"GATE_GAIN_PCT": "nan"}),
            (("chunkflow",), {"RUNTIME_ENV_JSON": "invalid"}),
            (("chunkflow",), {"CHUNKFLOW_PROFILE": "1", "PROFILE_END": "08"}),
            (("chunkflow", "extra"), {}),
        ]
        for modes, overrides in cases:
            with self.subTest(modes=modes, overrides=overrides):
                result = self.launch(*modes, **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(Path(self.env["CF_CAPTURE"]).exists())

    def test_incomplete_run_fails_but_async_only_checks_submission(self):
        result = self.launch("chunkflow", CF_INCOMPLETE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected training steps not completed", result.stderr)
        result = self.launch("chunkflow", CF_INCOMPLETE="1", RAY_NO_WAIT="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--no-wait", json.loads(Path(self.env["CF_CAPTURE"]).read_text()))

    def test_explicit_backend_recompute_and_profiler_are_preserved(self):
        result = self.launch(
            "chunkflow",
            RELAX_CHUNKFLOW_FLASH_BACKEND="fa2",
            KEEP_LAST_K="1",
            CHUNKFLOW_PROFILE="1",
            PROFILE_START="12",
            PROFILE_END="14",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = json.loads(Path(self.env["CF_CAPTURE"]).read_text())
        self.assertEqual(args[args.index("--chunkflow-keep-last-k") + 1], "1")
        self.assertIn("--use-pytorch-profiler", args)
        self.assertEqual(args[args.index("--profile-target") + 1], "train_overall")
        self.assertEqual(args[args.index("--profile-step-start") + 1], "12")
        self.assertEqual(args[args.index("--profile-step-end") + 1], "14")


if __name__ == "__main__":
    unittest.main()
