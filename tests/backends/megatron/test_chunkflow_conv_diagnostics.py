# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import csv
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from relax.backends.megatron.chunkflow import conv_diagnostics as cd
from relax.backends.megatron.chunkflow import gdn
from relax.backends.megatron.chunkflow import layer_diagnostics as ld
from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR
from relax.backends.megatron.chunkflow.scheduler import Chunk, Segment
from relax.backends.megatron.chunkflow.units import Unit


def _kernel(
    x,
    weight,
    bias=None,
    initial_state=None,
    output_final_state=False,
    activation="silu",
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    cp_context=None,
):
    cu = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    outputs, states = [], []
    for sequence, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=True)):
        state = (
            x.new_zeros(1, x.shape[-1], weight.shape[1])
            if initial_state is None
            else initial_state[sequence : sequence + 1]
        )
        for t in range(start, end):
            state = torch.cat((state[:, :, 1:], x[:, t, :, None]), -1)
            pre = (state * weight).sum(-1)
            if bias is not None:
                pre = pre + bias
            outputs.append(F.silu(pre) if activation else pre)
        states.append(state)
    return torch.stack(outputs, 1), torch.cat(states) if output_final_state else None


@pytest.mark.parametrize("recompute", [False, True])
def test_convolution_probes_pair_chunks_capture_state_and_check_reference(tmp_path, monkeypatch, recompute):
    ssm = ModuleType("megatron.core.ssm")
    kernels = ModuleType("megatron.core.ssm.gated_delta_net")
    kernels.causal_conv1d = _kernel
    ssm.gated_delta_net = kernels
    monkeypatch.setitem(sys.modules, ssm.__name__, ssm)
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setenv("RELAX_CHUNKFLOW_CONV_TRACE", "1")
    monkeypatch.setenv("RELAX_CHUNKFLOW_CONV_CHUNK_SIZE", "3")

    class GatedDeltaNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 15
            self.qk_dim_local_tp = self.v_dim_local_tp = 1
            self.conv1d = torch.nn.Conv1d(3, 3, 3, groups=3, dtype=torch.float64)

        def _forward_compute(self, x, packed_seq_params):
            packed = packed_seq_params
            context = None
            if hasattr(packed, UNIT_ATTR):
                context = gdn._ACTIVE.set(
                    gdn._Context(
                        self,
                        packed,
                        packed.previous,
                        packed.current if getattr(packed, UNIT_ATTR).is_dependent else None,
                    )
                )
            try:
                y, _ = kernels.causal_conv1d(
                    x=F.pad(x, (0, 0, 0, packed.pad)),
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    cu_seqlens=packed.cu_seqlens_q,
                )
                return y[:, : x.shape[1]]
            finally:
                if context is not None:
                    gdn._ACTIVE.reset(context)

        def forward(self, x, packed):
            return (
                checkpoint(self._forward_compute, x, packed, use_reentrant=True)
                if recompute
                else self._forward_compute(x, packed)
            )

    torch.manual_seed(7)
    model = GatedDeltaNet()
    x = torch.randn(1, 6, 3, dtype=torch.float64)
    packed = SimpleNamespace(cu_seqlens_q_cpu=[0, 8], cu_seqlens_q=torch.tensor([0, 8]), pad=2)
    parent = {"tokens": torch.arange(8).reshape(1, 8), "total_lengths": [6], "packed_seq_params": packed}
    for arm in ("baseline", "chunkflow"):
        model.zero_grad(set_to_none=True)
        data = x.clone().requires_grad_()
        if arm == "chunkflow":
            kernels.causal_conv1d = gdn._wrap_kernel(_kernel, "conv")
            kernels.causal_conv1d._chunkflow_original = _kernel
        with cd.trace_convolutions([model], tmp_path / arm / "diagnostics", 0, 0):
            if arm == "baseline":
                ld.set_batch(parent)
                out = model(data, packed)
                # Checkpoint replay must restore coordinates, not use the current batch.
                ld.set_batch({"tokens": torch.tensor([[99]]), "total_lengths": [1]})
                out.sum().backward()
            else:
                outputs, previous = [], {}
                for i in range(2):
                    start, end = i * 3, (i + 1) * 3
                    part = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 4]), pad=1, previous=previous, current={})
                    unit = Unit(i, [Chunk(i, [Segment(0, start, 3)], group=0, group_idx=i, group_size=2)])
                    setattr(part, UNIT_ATTR, unit)
                    batch = {"tokens": torch.arange(4).reshape(1, 4), "total_lengths": [3], "packed_seq_params": part}
                    ld.unit_layout(parent, batch, [(start, end)])
                    ld.set_batch(batch)
                    outputs.append(model(data[:, start:end], part))
                    previous = part.current
                part = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2]), pad=0, previous={}, current={})
                part._chunkflow_gdn_lengths = (2, part.cu_seqlens_q, part.cu_seqlens_q.long())
                setattr(part, UNIT_ATTR, Unit(2, [Chunk(2, [Segment(0, 0, 2)])]))
                batch = {
                    "tokens": torch.zeros(1, 2, dtype=torch.long),
                    "total_lengths": [2],
                    "packed_seq_params": part,
                }
                ld.unit_layout(parent, batch, [(6, 8)])
                ld.set_batch(batch)
                padding = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
                outputs.append(model(padding, part) * 0)
                ld.set_batch({"tokens": torch.tensor([[99]]), "total_lengths": [1]})
                for output in reversed(outputs):
                    output.sum().backward()
            cd.capture_final_gradients([model])
        assert cd._TRACE.get() is None and cd._MODULE.get() is None
        assert "_forward_compute" not in vars(model)
    path = Path(__file__).resolve().parents[3] / "scripts/training/sft/tools/compare_qwen36_convolutions.py"
    compare = runpy.run_path(str(path))["compare"]
    compare(tmp_path)
    rows = list(csv.DictReader((tmp_path / "convolutions.csv").open()))
    assert {r["field"] for r in rows} >= {"x", "y", "dy", "dx", "raw_dw_sum", "final_dw", "dw_reference"}
    assert all(float(r["max_abs"]) < 1e-6 for r in rows)
    assert "Checked state edges: 2; max absolute mismatch: 0" in (tmp_path / "convolutions.md").read_text()
    file = tmp_path / "chunkflow/diagnostics/conv_r0_rank0.pt"
    captured = torch.load(file, weights_only=True)
    assert sorted((c["physical_rows"], c["padding_rows"]) for c in captured["calls"]) == [(2, 2), (3, 0), (3, 0)]
    next(c for c in captured["calls"] if c["group"] is not None)["values"]["dw"] *= 2
    torch.save(captured, file)
    compare(tmp_path)
    rows = list(csv.DictReader((tmp_path / "convolutions.csv").open()))
    assert any(r["field"] == "raw_dw_sum" and float(r["max_abs"]) > 0 for r in rows)


@pytest.mark.parametrize("cu,tail", [([0, 4, 8], []), ([0, 4, 8, 10], [99, 99])])
def test_token_coordinates_exclude_padding_between_sequences(cu, tail):
    batch = {
        "tokens": torch.tensor([[1, 2, 99, 99, 3, 4, 99, 99] + tail]),
        "total_lengths": [2, 2],
        "packed_seq_params": SimpleNamespace(cu_seqlens_q_cpu=cu),
    }
    layout = ld._layout(batch)
    assert layout["rows"] == [0, 1, -1, -1, 4, 5, -1, -1] + [-1] * len(tail)
    assert layout["selected"] == {0, 1, 4, 5}
