# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import csv
import runpy
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from relax.backends.megatron.chunkflow import layer_diagnostics as ld
from relax.backends.megatron.chunkflow import tail_diagnostics as td


class TransformerLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer_number = layer
        self.self_attention = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Linear(4, 4)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attention(hidden_states)
        return hidden_states + self.mlp(hidden_states)


@pytest.mark.parametrize("recompute", [False, True])
def test_tail_probes_pair_repacked_head_rows_and_preserve_backward(tmp_path, monkeypatch, recompute):
    from megatron.core import parallel_state as ps

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
    torch.manual_seed(1)
    model = torch.nn.Module()
    model.layers = torch.nn.Sequential(TransformerLayer(39), TransformerLayer(40))
    model.final_layernorm = torch.nn.LayerNorm(4)
    head = torch.nn.Linear(4, 8)
    x = torch.randn(64, 1, 4)
    parent = {"tokens": torch.arange(64).view(1, 64), "total_lengths": [64]}
    gradients = []
    for arm in ("plain", "baseline", "baseline_repeat", "chunkflow"):
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        data = x.clone().requires_grad_()
        folder = tmp_path / arm / "diagnostics"
        with td.trace_tail([model], folder, 0, 0):
            if arm == "plain":
                active = td._TRACE.set(None)
            for start, end in [(32, 64), (0, 32)] if arm == "chunkflow" else [(0, 64)]:
                batch = {"tokens": parent["tokens"][:, start:end]}
                ld.unit_layout(parent, batch, [(start, end)])
                ld.set_batch(batch)
                y = model.final_layernorm(model.layers(data[start:end]))
                source = y.transpose(0, 1)

                def loss(source):
                    outputs = []
                    for offset in range(0, end - start, 7):
                        h = source[:, offset : offset + 7].transpose(0, 1)
                        logits = head(h).squeeze(1)
                        targets = (torch.arange(logits.shape[0]) + start + offset) % 8
                        lp = logits.log_softmax(-1).gather(1, targets[:, None])
                        td.tap_head(source, h, logits, lp, targets)
                        outputs.append(-lp.sum())
                    return sum(outputs)

                measured_loss = td.bind_loss_probe(loss, batch)
                loss_value = (
                    checkpoint(measured_loss, source, use_reentrant=True) if recompute else measured_loss(source)
                )
                ld.set_batch({"tokens": torch.tensor([[99]]), "total_lengths": [1]})
                loss_value.backward()
            if arm == "plain":
                td._TRACE.reset(active)
        gradients.append([p.grad.clone() for p in model.parameters()] + [data.grad])
        assert td._TRACE.get() is None and ld._TRACE.get() is None and ld._BATCH.get() is None
        assert not model.final_layernorm._forward_hooks and not model.layers[1]._forward_pre_hooks
    for plain, measured in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(plain, measured, rtol=0, atol=0)
    compare = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/training/sft/tools/compare_qwen36_layers.py")
    )["compare"]
    compare(tmp_path, candidate="baseline_repeat", prefix="tail")
    rows = list(csv.DictReader((tmp_path / "tail-baseline-repeat.csv").open()))
    assert all(float(r["max_abs"]) == 0 for r in rows)
    assert {r["event"] for r in rows} >= {
        "lm_head_input_tp0_grad",
        "lm_head_logits_tp0_grad",
        "log_probs_grad",
        "final_norm_input_grad",
        "attention_input_grad",
        "moe_input_grad",
    }
    compare(tmp_path, prefix="tail")
    rows = list(csv.DictReader((tmp_path / "tail.csv").open()))
    assert all(float(r["max_abs"]) < 2e-6 for r in rows)
    for plain, measured in zip(gradients[0], gradients[3], strict=True):
        torch.testing.assert_close(plain, measured, rtol=2e-5, atol=1e-6)
    captured = torch.load(tmp_path / "chunkflow/diagnostics/tail_r0_rank0.pt", weights_only=True)
    assert {row for e in captured if e["kind"] == "lm_head_input_tp0" for row in e["rows"]} == ld._layout(parent)[
        "selected"
    ]
    assert next(e for e in captured if e["kind"] == "lm_head_logits_tp0")["values"].shape[1] == 9


def test_tail_sequence_parallel_coordinates_and_cleanup_on_error(tmp_path, monkeypatch):
    from megatron.core import parallel_state as ps

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 1)
    model = torch.nn.Module()
    model.final_layernorm = torch.nn.LayerNorm(4)
    with pytest.raises(RuntimeError, match="stop"):
        with td.trace_tail([model], tmp_path, 0, 7):
            ld.set_batch({"tokens": torch.arange(8).view(1, 8), "total_lengths": [8]})
            model.final_layernorm(torch.randn(4, 1, 4, requires_grad=True)).sum().backward()
            events = td._TRACE.get()["events"]
            assert all(e["rows"] == [4, 5, 6, 7] for e in events)
            source = torch.randn(1, 8, 4, requires_grad=True)
            with pytest.raises(ValueError, match="row-aligned view"):
                td.tap_head(
                    source,
                    source.transpose(0, 1).clone(),
                    torch.randn(8, 8),
                    torch.randn(8, 1),
                    torch.zeros(8, dtype=torch.long),
                )
            raise RuntimeError("stop")
    assert td._TRACE.get() is None and ld._BATCH.get() is None
    assert not model.final_layernorm._forward_hooks and not model.final_layernorm._forward_pre_hooks
