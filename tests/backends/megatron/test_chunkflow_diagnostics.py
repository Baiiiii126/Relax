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


def test_layer_trace_pairs_repacked_tokens_and_captures_router_changes(tmp_path):
    from relax.backends.megatron.chunkflow import layer_diagnostics as ld

    class TopKRouter(torch.nn.Module):
        def gating(self, input):
            return input[:, 0, :2].double()

        def forward(self, input):
            probs = self.gating(input).softmax(-1)
            return probs, probs == probs.max(dim=-1, keepdim=True).values

    class TransformerLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 1
            self.router = TopKRouter()
            self.self_attention = torch.nn.Identity()
            self.mlp = torch.nn.Identity()

        def forward(self, hidden_states):
            hidden_states = self.self_attention(hidden_states)
            self.router(hidden_states)
            return self.mlp(hidden_states) * 2

    parent = {"tokens": torch.arange(64).reshape(1, 64), "total_lengths": [64]}
    model = TransformerLayer()
    for arm in ("baseline", "chunkflow"):
        x = torch.arange(256, dtype=torch.float32).reshape(64, 1, 4).requires_grad_()
        with ld.trace_layers([model], tmp_path / arm / "diagnostics", 0, 0):
            if arm == "baseline":
                ld.set_batch(parent)
                model(hidden_states=x).sum().backward()
            else:
                for start, end in ((32, 64), (0, 32)):
                    batch = {"tokens": parent["tokens"][:, start:end]}
                    ld.unit_layout(parent, batch, [(start, end)])
                    ld.set_batch(batch)
                    model(x[start:end]).sum().backward()
    assert ld._TRACE.get() is None and ld._BATCH.get() is None
    assert "gating" not in vars(model.router)
    path = Path(__file__).resolve().parents[3] / "scripts/training/sft/tools/compare_qwen36_layers.py"
    spec = importlib.util.spec_from_file_location("compare_layers", path)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    comparison.compare(tmp_path)
    rows = list(csv.DictReader((tmp_path / "layers.csv").open()))
    assert {r["event"] for r in rows} >= {"input", "output", "input_grad", "output_grad", "router_map"}
    assert {r["event"] for r in rows} >= {"self_attention_output", "mlp_output", "router_logits"}
    assert all(float(r["max_abs"]) == 0 and int(r["tokens"]) == 64 for r in rows)
    file = tmp_path / "chunkflow/diagnostics/layers_r0_rank0.pt"
    events = torch.load(file, weights_only=True)
    assert next(e for e in events if e["kind"] == "router_logits")["values"].dtype == torch.float64
    next(e for e in events if e["kind"] == "router_map")["values"][0].logical_not_()
    torch.save(events, file)
    comparison.compare(tmp_path)
    rows = list(csv.DictReader((tmp_path / "layers.csv").open()))
    assert next(r for r in rows if r["event"] == "router_map")["changed_tokens"] == "1"


def test_layer_trace_tracks_sequence_parallel_row_placement(monkeypatch):
    from megatron.core import parallel_state as ps

    from relax.backends.megatron.chunkflow import layer_diagnostics as ld

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 1)
    token = ld._BATCH.set({"rows": [6, 7, 0, 1, 2, 3, 4, 5], "selected": {0, 2, 4, 6}, "parent": "test"})
    try:
        _, indices, rows = ld._selection(torch.ones(4, 1, 8))
        assert indices == [0, 2] and rows == [2, 4]
        _, indices, rows = ld._selection(torch.ones(4, 1, 8), all_tokens=True)
        assert indices == [0, 1, 2, 3] and rows == [2, 3, 4, 5]
    finally:
        ld._BATCH.reset(token)


def test_first_update_captures_clipping_master_and_restores_methods(tmp_path, monkeypatch):
    from relax.backends.megatron.chunkflow import update_diagnostics as ud

    model = torch.nn.Linear(3, 2, bias=False)
    parameter = model.weight
    parameter.grad = torch.ones_like(parameter)
    master = parameter.detach().float().clone()
    shard = torch.nn.Parameter(parameter.detach().clone())
    shard.decoupled_grad = torch.ones_like(shard)

    class Hybrid:
        def __init__(self):
            self.param_to_inner_param = {shard: master}
            self.param_groups = [{"params": [shard], "lr": 0.1}]
            self.state = {}

        def step(self):
            master.sub_(shard.decoupled_grad * 0.1)
            self.state[shard] = {"exp_avg": shard.decoupled_grad.clone(), "exp_avg_sq": shard.decoupled_grad.square()}

    class Outer:
        def __init__(self):
            self.optimizer = Hybrid()
            self.model_param_group_index_map = {parameter: (0, 0)}

        def _get_model_param_range_map(self, parameter):
            from types import SimpleNamespace

            return {"param": SimpleNamespace(start=0, end=parameter.numel())}

        def prepare_grads(self):
            shard.decoupled_grad.mul_(2)

        def step(self):
            self.prepare_grads()
            shard.decoupled_grad.mul_(0.25)
            self.optimizer.step()
            with torch.no_grad():
                parameter.copy_(master)

    optimizer = Outer()
    monkeypatch.setenv("RELAX_CHUNKFLOW_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_CHUNKFLOW_GRAD_ROLLOUTS", "0")
    monkeypatch.setenv("RELAX_CHUNKFLOW_UPDATE_TRACE", "1")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    before = master.clone()
    with ud.trace_update([model], optimizer, 0, 0):
        optimizer.step()
    data = torch.load(tmp_path / "update_r0_s0_rank0.pt", weights_only=True)
    assert not data["issues"]
    record = data["optimizers"][0]
    torch.testing.assert_close(data["prepared_grad"]["0.weight"], torch.full((64,), 2.0))
    torch.testing.assert_close(record["clipped_grad"]["0.weight"], torch.full((64,), 0.5))
    torch.testing.assert_close(
        record["after_master"]["0.weight"] - record["before_master"]["0.weight"], torch.full((64,), -0.05)
    )
    torch.testing.assert_close(master, before - 0.05)
    assert "step" not in vars(optimizer.optimizer) and "prepare_grads" not in vars(optimizer)
    import runpy

    for arm in ("baseline", "chunkflow"):
        folder = tmp_path / arm / "diagnostics"
        folder.mkdir(parents=True)
        if arm == "chunkflow":
            record["after_master"]["0.weight"].add_(0.01)
        torch.save(data, folder / "update_r0_s0_rank0.pt")
    path = Path(__file__).resolve().parents[3] / "scripts/training/sft/tools/compare_qwen36_updates.py"
    runpy.run_path(str(path))["compare"](tmp_path)
    rows = list(csv.DictReader((tmp_path / "updates.csv").open()))
    delta = next(r for r in rows if r["field"] == "master_delta")
    assert float(delta["relative_l2"]) == pytest.approx(0.2, abs=1e-5)
    with pytest.raises(RuntimeError, match="stop"):
        with ud.trace_update([model], optimizer, 0, 0):
            raise RuntimeError("stop")
    assert "step" not in vars(optimizer.optimizer) and "prepare_grads" not in vars(optimizer)

    from types import SimpleNamespace

    bad_owner = Outer()
    bad_owner.model_param_group_index_map = {parameter: (99, 0)}
    chain = SimpleNamespace(chained_optimizers=[optimizer, bad_owner])
    with pytest.raises(IndexError):
        with ud.trace_update([model], chain, 0, 0):
            pass
    assert "step" not in vars(optimizer.optimizer)
