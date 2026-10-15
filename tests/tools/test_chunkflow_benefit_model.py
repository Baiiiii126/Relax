# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "training" / "sft" / "tools" / "chunkflow_benefit_model.py"
_SPEC = importlib.util.spec_from_file_location("chunkflow_benefit_model_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model)


def test_step_metrics_includes_runtime_parent_padding():
    cf = model._load_chunkflow(Path(__file__).parents[2])

    metrics = model.step_metrics([1100], 512, 1600, 4, "staggered", 0, cf)

    assert metrics["num_microbatches"] == 1
    assert metrics["source_rows"] == 1100
    assert metrics["parent_padding_rows"] == 180
    assert metrics["num_units"] == 4


def test_step_metrics_uses_runtime_balanced_parent_partitions():
    cf = model._load_chunkflow(Path(__file__).parents[2])
    lengths = [1000, 900, 800, 700, 600]
    first_fit = model._first_fit_partitions(lengths, 1600)
    balanced = cf.balancing.get_seqlen_balanced_partitions(lengths, len(first_fit), equal_size=False)

    assert sorted(sorted(group) for group in first_fit) != sorted(sorted(group) for group in balanced)
    metrics = model.step_metrics(lengths, 512, 1600, 4, "staggered", 0, cf)
    assert metrics["num_microbatches"] == len(first_fit)
    assert metrics["num_units"] == 13
