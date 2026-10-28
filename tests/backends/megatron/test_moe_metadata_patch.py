# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the patched dispatcher methods on CPU, including async-copy
ordering.

Set RELAX_MOE_DISPATCHER_SOURCE to a patched token_dispatcher.py when Megatron
is not installed. No CUDA kernels or Megatron imports are needed for these
checks.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


class Tensor:
    def __init__(self, data, cuda=True):
        self.data = np.asarray(data)
        self.is_cuda = cuda

    def sum(self, axis=None, dim=None):
        return Tensor(self.data.sum(axis=axis if dim is None else dim), self.is_cuda)

    def long(self):
        return Tensor(self.data.astype(np.int64), self.is_cuda)

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), self.is_cuda)

    view = reshape

    def transpose(self, a, b):
        return Tensor(self.data.swapaxes(a, b), self.is_cuda)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.is_cuda)

    def __getitem__(self, key):
        return Tensor(self.data[key], self.is_cuda)

    def size(self, axis):
        return self.data.shape[axis]


class CPUCounts(np.ndarray):
    """Reject reads before the event completes, just like an unfinished
    DtoH."""

    def sum(self, *args, **kwargs):
        assert self.event.synced, "CPU metadata read before DtoH completion"
        return np.asarray(self).sum(*args, **kwargs)


class Event:
    def __init__(self):
        self.synced = False
        self.calls = 0

    def synchronize(self):
        self.synced = True
        self.calls += 1


def dispatcher_source():
    explicit = os.environ.get("RELAX_MOE_DISPATCHER_SOURCE")
    if explicit:
        return Path(explicit)
    try:
        spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        spec = None
    if spec and spec.origin:
        return Path(spec.origin).parent / "transformer/moe/token_dispatcher.py"
    raise unittest.SkipTest("Set RELAX_MOE_DISPATCHER_SOURCE or install patched Megatron")


def load_dispatcher():
    source = dispatcher_source().read_text()
    if "_relax_moe_metadata_patch_version = 1" not in source:
        raise AssertionError("Apply scripts/tools/apply_moe_metadata_patch.py first")
    cls = next(
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "MoEAlltoAllTokenDispatcher"
    )
    names = {"_init_relax_metadata_mode", "preprocess", "_maybe_dtoh_and_synchronize", "_maybe_update_cuda_sync_point"}
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    namespace = {"os": os, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module), str(dispatcher_source()), "exec"), namespace)
    return namespace[cls.name], namespace


class MoEMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dispatcher_cls, cls.namespace = load_dispatcher()

    def make_dispatcher(self, mode, rank=0, **overrides):
        d = self.dispatcher_cls()
        d.config = SimpleNamespace(
            cuda_graph_impl="none",
            moe_expert_capacity_factor=None,
            moe_router_padding_for_quantization=False,
            moe_permute_fusion=True,
            moe_router_topk=8,
        )
        d.tp_size, d.ep_size, d.tp_rank = 1, 2, 0
        d.num_experts, d.num_local_experts = 256, 128
        d.local_expert_indices = list(range(rank * 128, (rank + 1) * 128))
        d.drop_and_pad = False
        d.tp_ep_group = object()
        d.cuda_sync_point = "no_sync"
        d.cuda_dtoh_point = "before_permutation_1"
        d.cuda_sync_point_priority = dict(
            zip(
                ("before_permutation_1", "before_ep_alltoall", "before_permutation_2", "before_finish", "no_sync"),
                range(5),
            )
        )
        d.output_splits_tp = None
        for key, value in overrides.items():
            if hasattr(d.config, key):
                setattr(d.config, key, value)
            else:
                setattr(d, key, value)
        with patch.dict(os.environ, RELAX_MOE_METADATA_MODE=mode):
            d._init_relax_metadata_mode()
        return d

    def run_metadata(self, d, routes, rank):
        event = Event()
        copies = []
        stream = SimpleNamespace(wait_stream=lambda _: None, record_event=lambda: event)
        d.cuda_dtoh_stream = stream

        def copy(t, as_numpy=False, record_stream=False):
            if not isinstance(t, Tensor) or not t.is_cuda:
                return t
            self.assertTrue(record_stream)
            copies.append(t.data.shape)
            data = t.data.copy()
            if as_numpy:
                result = data.view(CPUCounts)
                result.event = event
                return result
            return Tensor(data, cuda=False)

        torch = SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda: None, stream=lambda _: nullcontext()),
            from_numpy=lambda x: Tensor(x, cuda=False),
        )
        counts = np.stack([r.sum(axis=0, dtype=np.int64) for r in routes])

        def gather(local, group):
            self.assertIs(group, d.tp_ep_group)
            np.testing.assert_array_equal(local.data, counts[rank])
            return Tensor(counts.reshape(-1))

        with patch.dict(
            self.namespace, torch=torch, maybe_move_tensor_to_cpu=copy, gather_from_sequence_parallel_region=gather
        ):
            tokens = d.preprocess(Tensor(routes[rank]))
            gpu_counts = d.num_global_tokens_per_local_expert
            tokens = d._maybe_dtoh_and_synchronize("before_permutation_1", tokens)
            self.assertFalse(event.synced)
            tokens = d._maybe_dtoh_and_synchronize("before_ep_alltoall", tokens)
            output = d.output_splits
            for point in ("before_permutation_2", "before_finish"):
                self.assertIs(tokens, d._maybe_dtoh_and_synchronize(point, tokens))
                self.assertIs(output, d.output_splits)
        self.assertEqual(event.calls, 1)
        self.assertIs(gpu_counts, d.num_global_tokens_per_local_expert)
        self.assertTrue(gpu_counts.is_cuda)
        self.assertFalse(tokens.is_cuda)
        self.assertEqual(tokens.data.dtype, np.int64)
        self.assertEqual(d.output_splits.dtype, np.int64)
        self.assertIsNone(d._relax_counts_cpu)
        if d.relax_metadata_mode != "off":
            self.assertIsNone(d.output_splits_tp)
        start = rank * 128
        expected = counts[:, start : start + 128]
        np.testing.assert_array_equal(d.output_splits, expected.sum(axis=1))
        np.testing.assert_array_equal(tokens.data, expected.sum(axis=0))
        np.testing.assert_array_equal(d.input_splits, counts[rank].reshape(2, 128).sum(axis=1))
        return copies, (d.input_splits, d.output_splits, tokens.data)

    def test_modes_preserve_counts_copy_budget_and_prior_forward_storage(self):
        rng = np.random.default_rng(7)
        random = [np.zeros((8192, 256), dtype=bool) for _ in range(2)]
        for r in random:
            indices = np.argsort(rng.random(r.shape), axis=1)[:, :8]
            np.put_along_axis(r, indices, True, axis=1)
        skewed = [np.zeros((17, 256), dtype=bool) for _ in range(2)]
        for r in skewed:
            r[:, :8] = True  # Rank 1 receives nothing; most experts receive nothing.
        empty = [np.zeros((0, 256), dtype=bool) for _ in range(2)]
        for mode, expected_copies in (("off", 4), ("a", 3), ("b", 2)):
            for rank in (0, 1):
                with self.subTest(mode=mode, rank=rank):
                    d = self.make_dispatcher(mode, rank)
                    previous = []
                    for routes in (random, skewed, empty):
                        copies, arrays = self.run_metadata(d, routes, rank)
                        self.assertEqual(len(copies), expected_copies)
                        previous.append((arrays, [np.asarray(a).copy() for a in arrays]))
                    for arrays, snapshots in reversed(previous):
                        for a, snapshot in zip(arrays, snapshots):
                            np.testing.assert_array_equal(a, snapshot)

    def test_guards_reject_unsupported_modes_and_configs(self):
        for mode, overrides in (
            ("invalid", {}),
            ("a", {"tp_size": 2}),
            ("a", {"cuda_graph_impl": "local"}),
            ("b", {"ep_size": 1}),
            ("b", {"moe_permute_fusion": False}),
            ("b", {"moe_expert_capacity_factor": 1.0}),
            ("b", {"drop_and_pad": True}),
            ("b", {"moe_router_padding_for_quantization": True}),
            ("b", {"num_local_experts": 64}),
        ):
            with self.subTest(mode=mode, overrides=overrides), self.assertRaises(ValueError):
                self.make_dispatcher(mode, **overrides)
        self.make_dispatcher("off", tp_size=2, cuda_graph_impl="local")

    def test_patch_application_is_idempotent_and_rejects_mismatch(self):
        spec = importlib.util.spec_from_file_location(
            "apply_metadata", ROOT / "scripts/tools/apply_moe_metadata_patch.py"
        )
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / tool.TARGET
            target.parent.mkdir(parents=True)
            target.write_text(dispatcher_source().read_text())
            self.assertEqual(tool.apply_patch(root), "already applied")
            import subprocess

            subprocess.run(
                ["git", "apply", "--reverse", f"--include={tool.TARGET}", str(tool.PATCH)], cwd=root, check=True
            )
            before = target.read_bytes()
            self.assertEqual(tool.apply_patch(root, check_only=True), "applicable (check only)")
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(tool.apply_patch(root), "applied")
            self.assertEqual(tool.apply_patch(root), "already applied")
            target.write_text("# incompatible source\n")
            with self.assertRaises(RuntimeError):
                tool.apply_patch(root)
            self.assertEqual(target.read_text(), "# incompatible source\n")


class AllocatorMemoryTests(unittest.TestCase):
    def test_memory_probe_is_opt_in_and_records_peaks_without_cuda_sync(self):
        import json
        import sys
        from contextlib import contextmanager
        from types import ModuleType

        source = ROOT / "relax/backends/megatron/training_profile.py"
        nodes = [
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_record_allocator_memory"
        ]
        state = {"allocated": 10, "reserved": 20, "peak_allocated": 30, "peak_reserved": 40}
        resets = []
        cuda = SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            memory_allocated=lambda _: state["allocated"],
            memory_reserved=lambda _: state["reserved"],
            max_memory_allocated=lambda _: state["peak_allocated"],
            max_memory_reserved=lambda _: state["peak_reserved"],
            reset_peak_memory_stats=lambda _: resets.append(True),
        )
        namespace = dict(contextmanager=contextmanager, os=os, torch=SimpleNamespace(cuda=cuda), Path=Path, json=json)
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        record = namespace["_record_allocator_memory"]
        with patch.dict(os.environ, {}, clear=True), record(SimpleNamespace(), 0, 0):
            pass
        self.assertEqual(resets, [])
        profile_utils = ModuleType("relax.utils.profile_utils")
        profile_utils._get_rank_tag = lambda: "rank0"
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(sys.modules, {"relax.utils.profile_utils": profile_utils}),
                patch.dict(os.environ, RELAX_TRAIN_MEMORY_DIR=directory, RELAX_MOE_METADATA_MODE="b"),
            ):
                with record(SimpleNamespace(use_chunkflow=True), 7, 0):
                    state.update(peak_allocated=50, peak_reserved=60)
            row = json.loads((Path(directory) / "memory_rank0.jsonl").read_text())
        self.assertEqual(len(resets), 1)
        self.assertEqual(row["before"]["peak_allocated_since_previous_reset"], 30)
        self.assertEqual(row["peak_allocated"], 50)
        self.assertEqual(row["peak_reserved"], 60)
        self.assertEqual(row["moe_metadata_mode"], "b")


if __name__ == "__main__":
    unittest.main()
