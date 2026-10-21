# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from relax.backends.megatron.chunkflow import diagnostics


def test_diagnostics_record_actual_calls_and_restore_on_exception(tmp_path, monkeypatch):
    class GatedDeltaNet(torch.nn.Module):
        def gated_delta_rule(self, q, initial_state=None, output_final_state=False):
            return q * 2

    model = GatedDeltaNet()
    monkeypatch.setattr(diagnostics, "_rank", lambda: 0)
    monkeypatch.setenv("RELAX_CHUNKFLOW_DIAGNOSTIC_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="stop"):
        with diagnostics.trace_gdn_calls([model], 0):
            q = torch.ones(1, 2, 3)
            torch.testing.assert_close(model.gated_delta_rule(q), q * 2)
            model.gated_delta_rule(q, initial_state=q, output_final_state=True)
            raise RuntimeError("stop")
    assert "gated_delta_rule" not in vars(model)
    assert diagnostics._ACTIVE.get() is None
    rows = [json.loads(line) for line in (tmp_path / "backend_rank0.jsonl").read_text().splitlines()]
    assert [row["initial_state"] for row in rows] == [False, True]
    assert rows[1]["output_final_state"] is True
    assert rows[1]["shapes"]["q"] == [1, 2, 3]
    with diagnostics.trace_gdn_calls([model], 28):
        assert "gated_delta_rule" not in vars(model)


def test_gradient_comparison_detects_direction_error_at_equal_norm(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/training/sft/tools/compare_qwen36_gradients.py"
    spec = importlib.util.spec_from_file_location("compare_gradients", path)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    model = torch.nn.Linear(3, 2, bias=False)
    original = model.weight.detach().clone()
    monkeypatch.setattr(diagnostics, "_rank", lambda: 0)
    monkeypatch.setenv("RELAX_CHUNKFLOW_GRAD_ROLLOUTS", "0,1,2,8")
    for arm, sign in (("baseline", 1), ("chunkflow", -1)):
        folder = tmp_path / arm / "diagnostics"
        monkeypatch.setenv("RELAX_CHUNKFLOW_DIAGNOSTIC_DIR", str(folder))
        model.weight.main_grad = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3) * sign
        model.weight.grad = torch.zeros_like(model.weight)
        diagnostics.capture_gradients([model], 0, 0)
        diagnostics.capture_gradients([model], 3, 0)
        assert not (folder / "grad_r3_s0_rank0.pt").exists()
        torch.testing.assert_close(model.weight, original)
    comparison.compare(tmp_path)
    row = next(csv.DictReader((tmp_path / "gradients.csv").open()))
    assert float(row["norm_ratio"]) == pytest.approx(1)
    assert float(row["sample_cosine"]) == pytest.approx(-1)
    assert float(row["sample_relative_l2"]) == pytest.approx(2)
    assert float(row["max_weight_sample_diff"]) == 0
    assert "Expected snapshot absent" in (tmp_path / "gradients.md").read_text()
    monkeypatch.delenv("RELAX_CHUNKFLOW_DIAGNOSTIC_DIR")
    diagnostics.capture_gradients(None, 0, 0)
    with diagnostics.trace_gdn_calls(None, 0):
        pass
