# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


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
                patch.dict(os.environ, RELAX_TRAIN_MEMORY_DIR=directory),
            ):
                with record(SimpleNamespace(use_chunkflow=True), 7, 0):
                    state.update(peak_allocated=50, peak_reserved=60)
            row = json.loads((Path(directory) / "memory_rank0.jsonl").read_text())
        self.assertEqual(len(resets), 1)
        self.assertEqual(row["before"]["peak_allocated_since_previous_reset"], 30)
        self.assertEqual(row["peak_allocated"], 50)
        self.assertEqual(row["peak_reserved"], 60)
        self.assertEqual(row["arm"], "chunkflow")
