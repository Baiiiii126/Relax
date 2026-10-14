# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[2] / "scripts" / "training" / "sft" / "tools" / "chunkflow_align.py"
_SPEC = importlib.util.spec_from_file_location("chunkflow_align_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
chunkflow_align = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chunkflow_align)


def _write_logs(tmp_path: Path, *, mismatched_tokens: bool = False) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.log"
    chunkflow = tmp_path / "chunkflow.log"
    baseline.write_text(
        "perf 0: {'perf/actor_train_time': 10.0, 'perf/step_time': 12.0, "
        "'perf/actor_train_tokens': 20000}\n"
        "perf 1: {'perf/actor_train_time': 8.0, 'perf/step_time': 9.0, "
        "'perf/actor_train_tokens': 18000}\n"
    )
    second_tokens = 17_999 if mismatched_tokens else 18_000
    chunkflow.write_text(
        "step 0: {'train/chunkflow/num_microbatches': 1, 'train/chunkflow/num_units': 20, "
        "'train/chunkflow/equal_unit_fill_drain_baseline_proxy': 4.0, "
        "'train/chunkflow/equal_unit_fill_drain_chunkflow_proxy': 1.15}\n"
        "step 1: {'train/chunkflow/num_microbatches': 1, 'train/chunkflow/num_units': 18, "
        "'train/chunkflow/equal_unit_fill_drain_baseline_proxy': 4.0, "
        "'train/chunkflow/equal_unit_fill_drain_chunkflow_proxy': 1.17}\n"
        "perf 0: {'perf/actor_train_time': 4.0, 'perf/step_time': 5.0, "
        "'perf/actor_train_tokens': 20000}\n"
        f"perf 1: {{'perf/actor_train_time': 4.0, 'perf/step_time': 5.0, "
        f"'perf/actor_train_tokens': {second_tokens}}}\n"
    )
    return baseline, chunkflow


def test_aggregate_speedup_uses_ratio_of_total_paired_times(tmp_path: Path):
    baseline, chunkflow = _write_logs(tmp_path)

    rows = chunkflow_align._pair(baseline, chunkflow, "actor_train_time")

    assert len(rows) == 2
    assert chunkflow_align._aggregate_speedup(rows) == pytest.approx((10.0 + 8.0) / (4.0 + 4.0))


def test_pair_rejects_different_token_counts(tmp_path: Path):
    baseline, chunkflow = _write_logs(tmp_path, mismatched_tokens=True)

    with pytest.raises(ValueError, match="different token counts"):
        chunkflow_align._pair(baseline, chunkflow, "actor_train_time")
