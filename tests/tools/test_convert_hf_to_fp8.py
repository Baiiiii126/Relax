# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the FP8 HF-checkpoint converter's fused-expert handling.

Qwen3.6 stores routed experts as a single fused 3-D tensor
(``...mlp.experts.gate_up_proj`` / ``.down_proj``). The converter must slice that
into per-expert 2-D projections, split ``gate_up`` into ``gate`` + ``up``, and
quantize each. It must also skip the vision tower and the shared-expert gate.

The real quantizer needs a GPU (triton FP8), so we stub ``quant_fp8`` with a
shape-recording no-op and fake the safetensors I/O. That isolates the pure
naming / slicing logic introduced by the commit.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402


_SOURCE = pathlib.Path(__file__).resolve().parents[2] / "scripts/tools/convert_hf_to_fp8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_hf_to_fp8_under_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class _FakeReader:
    """Minimal stand-in for a ``safetensors.safe_open`` handle."""

    def __init__(self, tensors):
        self._tensors = tensors

    def keys(self):
        return list(self._tensors.keys())

    def get_tensor(self, key):
        return self._tensors[key]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def harness(monkeypatch):
    """Patch heavy deps and run ``_process_file`` over an in-memory shard."""
    quant_shapes = []

    def _fake_quant_fp8(weight, strategy, block_size=None):
        quant_shapes.append(tuple(weight.shape))
        return weight.to(torch.float32), torch.ones(1, dtype=torch.float32)

    saved = {}

    def _fake_save_file(tensors, path, metadata=None):
        saved["tensors"] = dict(tensors)

    monkeypatch.setattr(mod, "quant_fp8", _fake_quant_fp8)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(safetensors.torch, "save_file", _fake_save_file)

    def _run(inputs, block_size=None):
        monkeypatch.setattr(safetensors, "safe_open", lambda *a, **k: _FakeReader(inputs))
        collector = mod.ConversionResult()
        mod._process_file(
            input_path="/in",
            output_path="/out",
            filename="model-00001-of-00001.safetensors",
            strategy="tensor",
            block_size=block_size,
            result_collector=collector,
        )
        return saved["tensors"], collector, quant_shapes

    return _run


class TestStoreQuantizedFp8:
    """Unit-level: scale-suffix selection depends on block_size."""

    def test_per_tensor_scale_suffix(self, monkeypatch):
        monkeypatch.setattr(mod, "quant_fp8", lambda w, s, b=None: (w, torch.ones(1)))
        out = {}
        mod._store_quantized_fp8(out, "a.b", torch.zeros(2, 2), strategy="tensor", block_size=None)
        assert set(out) == {"a.b.weight", "a.b.weight_scale"}

    def test_block_scale_suffix(self, monkeypatch):
        monkeypatch.setattr(mod, "quant_fp8", lambda w, s, b=None: (w, torch.ones(1)))
        out = {}
        mod._store_quantized_fp8(out, "a.b", torch.zeros(2, 2), strategy="block", block_size=[128, 128])
        assert set(out) == {"a.b.weight", "a.b.weight_scale_inv"}


class TestProcessFileFusedExperts:
    def test_fused_gate_up_and_down_are_split_per_expert(self, harness):
        inputs = {
            # 2 experts, gate_up rows = 2 * intermediate(=4), hidden = 3
            "L.mlp.experts.gate_up_proj": torch.randn(2, 8, 3),
            # 2 experts, intermediate = 4, hidden = 3
            "L.mlp.experts.down_proj": torch.randn(2, 3, 4),
        }
        saved, _collector, shapes = harness(inputs)

        expected = set()
        for i in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                expected.add(f"L.mlp.experts.{i}.{proj}.weight")
                expected.add(f"L.mlp.experts.{i}.{proj}.weight_scale")
        assert set(saved) == expected
        # gate_up (rows 8) is chunked into two (4, 3); down keeps (3, 4).
        assert shapes.count((4, 3)) == 4  # gate + up, 2 experts
        assert shapes.count((3, 4)) == 2  # down, 2 experts

    def test_regular_weight_quantized_and_excluded_modules_passthrough(self, harness):
        q = torch.randn(4, 4)
        visual = torch.randn(3, 3)
        gate = torch.randn(2, 2)
        inputs = {
            "L.self_attn.q_proj.weight": q,  # quantized (self_attn not excluded for FP8)
            "L.visual.blocks.0.attn.qkv.weight": visual,  # excluded: vision tower
            "L.mlp.shared_expert_gate.weight": gate,  # excluded: shared-expert gate
        }
        saved, collector, _shapes = harness(inputs)

        # q_proj is quantized -> weight + scale present
        assert "L.self_attn.q_proj.weight" in saved
        assert "L.self_attn.q_proj.weight_scale" in saved
        # excluded modules pass through untouched, no scale tensor emitted
        assert saved["L.visual.blocks.0.attn.qkv.weight"] is visual
        assert saved["L.mlp.shared_expert_gate.weight"] is gate
        assert "L.visual.blocks.0.attn.qkv.weight_scale" not in saved
        # and are recorded so downstream loaders skip them
        assert "L.visual.blocks.0.attn.qkv" in collector.modules_to_not_convert
        assert "L.mlp.shared_expert_gate" in collector.modules_to_not_convert

    def test_block_size_uses_inv_scale_for_fused_experts(self, harness):
        inputs = {"L.mlp.experts.down_proj": torch.randn(1, 3, 4)}
        saved, _collector, _shapes = harness(inputs, block_size=[128, 128])
        assert "L.mlp.experts.0.down_proj.weight" in saved
        assert "L.mlp.experts.0.down_proj.weight_scale_inv" in saved
