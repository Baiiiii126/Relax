# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import runpy
from pathlib import Path

import pytest


read_counts = runpy.run_path(str(Path(__file__).parents[2] / "scripts/tools/benchmark_expert_tiles.py"))["read_counts"]


def test_counts_replay_keeps_captured_shape_and_selects_record(tmp_path):
    path = tmp_path / "counts.jsonl"
    records = [
        {"metadata": "start"},
        {"counts": [0, 1, 63], "rows": 64},
        {"counts": [256, 257, 0], "rows": 513, "hidden": 2048, "layer": "decoder.2"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in records))
    assert read_counts(path, 1) == records[-1]
    with pytest.raises(ValueError, match="absent"):
        read_counts(path, 2)


@pytest.mark.parametrize(
    "record",
    [
        {"counts": []},
        {"counts": [-1, 1]},
        {"counts": [True, 1]},
        {"counts": [1.0]},
        {"counts": [0, 0]},
        {"counts": [2, 3], "rows": 4},
    ],
)
def test_counts_replay_rejects_inconsistent_payload(tmp_path, record):
    path = tmp_path / "counts.jsonl"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read_counts(path, 0)
