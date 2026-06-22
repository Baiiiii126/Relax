# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the FP8 ``modules_to_not_convert`` (ignore-list) quantization
path.

The commit that added Qwen3.6 multimodal QAT support introduced an ignore-list
based FP8 quantizer: when ``quantization_config`` carries ``modules_to_not_convert``
the converter quantizes *every* 2-D float weight except the modules named in that
list. HF checkpoints store routed experts fused (``...experts.<i>.gate_proj``), so
``_checkpoint_module_name`` maps those back to the fused HF module name
(``...experts.gate_up_proj`` / ``.down_proj``) used inside the ignore list.

We stub the triton FP8 kernel and the sglang deps so the module imports without a
GPU, and patch ``_quantize_param`` so the filtering logic is tested in isolation.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

import pytest


torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Import quantizer_fp8 with its heavy ancestors / leaf deps stubbed so it loads
# without triton, sglang, or the real weight_conversion package __init__.
# ---------------------------------------------------------------------------

_MODULE_NAME = "relax.backends.megatron.weight_conversion.processors.quantizer_fp8"
_SOURCE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "relax/backends/megatron/weight_conversion/processors/quantizer_fp8.py"
)

_STUB_PACKAGES = [
    "relax",
    "relax.backends",
    "relax.backends.megatron",
    "relax.backends.megatron.kernels",
    "relax.backends.megatron.kernels.fp8_kernel",
    "relax.backends.megatron.sglang",
    "relax.backends.megatron.weight_conversion",
    "relax.backends.megatron.weight_conversion.processors",
]


def _load_module():
    saved = {name: sys.modules.get(name) for name in _STUB_PACKAGES}
    for name in _STUB_PACKAGES:
        sys.modules[name] = MagicMock()
    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


qf = _load_module()


# ---------------------------------------------------------------------------
# _checkpoint_module_name: HF fused-module name resolution
# ---------------------------------------------------------------------------


class TestCheckpointModuleName:
    @pytest.mark.parametrize(
        "hf_name, expected",
        [
            # split expert projections collapse to the fused HF module name
            ("model.layers.0.mlp.experts.3.gate_proj.weight", "model.layers.0.mlp.experts.gate_up_proj"),
            ("model.layers.0.mlp.experts.3.up_proj.weight", "model.layers.0.mlp.experts.gate_up_proj"),
            ("model.layers.0.mlp.experts.3.down_proj.weight", "model.layers.0.mlp.experts.down_proj"),
            # multi-digit expert index
            ("m.mlp.experts.127.down_proj.weight", "m.mlp.experts.down_proj"),
            # non-expert weights just get the ``.weight`` suffix stripped
            ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj"),
            ("model.embed_tokens.weight", "model.embed_tokens"),
            # names without a ``.weight`` suffix are returned as the module itself
            ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj"),
            # ``.weight`` stripping happens before the expert regex match
            ("m.mlp.experts.1.up_proj", "m.mlp.experts.gate_up_proj"),
        ],
    )
    def test_names(self, hf_name, expected):
        assert qf._checkpoint_module_name(hf_name) == expected

    def test_shared_experts_are_not_treated_as_routed(self):
        # ``shared_experts`` (plural, no numeric index) must not match the routed
        # expert regex; it stays a plain module name.
        name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
        assert qf._checkpoint_module_name(name) == "model.layers.0.mlp.shared_experts.gate_proj"


# ---------------------------------------------------------------------------
# _quantize_params_fp8_by_ignore_list: filtering logic
# ---------------------------------------------------------------------------


class TestQuantizeByIgnoreList:
    @staticmethod
    def _stub_quantize_param(name, param, weight_block_size):
        """Deterministic marker so we can assert *which* params were
        quantized."""
        return [(name, "Q"), (name.replace(".weight", ".weight_scale"), "S")]

    def _run(self, monkeypatch, params, ignore):
        monkeypatch.setattr(qf, "_quantize_param", self._stub_quantize_param)
        return qf._quantize_params_fp8_by_ignore_list(params, set(ignore), weight_block_size=None)

    def test_quantizes_eligible_and_passes_through_rest(self, monkeypatch):
        w2d = torch.zeros(4, 4, dtype=torch.float32)
        bf16 = torch.zeros(4, 4, dtype=torch.bfloat16)
        params = [
            ("m.experts.0.gate_proj.weight", w2d),  # eligible -> quantized
            ("m.self_attn.k_proj.weight", bf16),  # eligible (bf16) -> quantized
            ("m.experts.0.up_proj.weight", w2d),  # ignored via fused name -> passthrough
            ("m.norm.weight", torch.zeros(4)),  # 1-D -> passthrough
            ("m.self_attn.q_proj.bias", w2d),  # not ``.weight`` -> passthrough
            ("m.int.weight", torch.zeros(4, 4, dtype=torch.int8)),  # non-float -> passthrough
        ]
        # ignore the fused gate_up module -> up_proj (and gate_proj) map into it.
        out = self._run(monkeypatch, params, ignore={"m.experts.gate_up_proj"})
        out_map = dict(out)

        # gate_proj is quantized only if NOT ignored; here gate_up_proj IS ignored,
        # so BOTH gate_proj and up_proj pass through untouched.
        assert ("m.experts.0.gate_proj.weight", w2d) in out
        assert ("m.experts.0.up_proj.weight", w2d) in out
        # k_proj is eligible and not ignored -> quantized (marker + scale emitted)
        assert out_map["m.self_attn.k_proj.weight"] == "Q"
        assert out_map["m.self_attn.k_proj.weight_scale"] == "S"
        # passthroughs keep their original tensor objects
        assert ("m.norm.weight", params[3][1]) in out
        assert ("m.self_attn.q_proj.bias", w2d) in out
        assert ("m.int.weight", params[5][1]) in out

    def test_ignore_list_targets_fused_expert_name(self, monkeypatch):
        w2d = torch.zeros(2, 2, dtype=torch.float32)
        params = [
            ("m.experts.0.gate_proj.weight", w2d),
            ("m.experts.0.down_proj.weight", w2d),
        ]
        # Only down_proj is protected; gate_proj should still be quantized.
        out = dict(self._run(monkeypatch, params, ignore={"m.experts.down_proj"}))
        assert out["m.experts.0.gate_proj.weight"] == "Q"
        assert out["m.experts.0.down_proj.weight"] is w2d  # passthrough

    def test_empty_ignore_quantizes_all_eligible(self, monkeypatch):
        w2d = torch.zeros(2, 2, dtype=torch.float32)
        params = [("m.experts.0.gate_proj.weight", w2d), ("m.experts.0.down_proj.weight", w2d)]
        out = dict(self._run(monkeypatch, params, ignore=set()))
        assert out["m.experts.0.gate_proj.weight"] == "Q"
        assert out["m.experts.0.down_proj.weight"] == "Q"


# ---------------------------------------------------------------------------
# quantize_params_fp8: dispatch into the ignore-list path
# ---------------------------------------------------------------------------


class TestDispatch:
    _BASE_CONFIG = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}

    def test_routes_to_ignore_list_when_present(self, monkeypatch):
        recorded = {}

        def _recorder(converted, ignore_set, weight_block_size):
            recorded["args"] = (converted, ignore_set, weight_block_size)
            return "SENTINEL"

        monkeypatch.setattr(qf, "_quantize_params_fp8_by_ignore_list", _recorder)

        converted = [("m.experts.0.gate_proj.weight", torch.zeros(2, 2))]
        config = {**self._BASE_CONFIG, "modules_to_not_convert": ["a.b", "c.d"], "weight_block_size": [128, 128]}

        result = qf.quantize_params_fp8(
            args=None, megatron_name="whatever", converted_named_params=converted, quantization_config=config
        )

        assert result == "SENTINEL"
        got_converted, got_ignore, got_block = recorded["args"]
        assert got_converted is converted
        assert got_ignore == {"a.b", "c.d"}  # list -> set
        assert got_block == [128, 128]

    def test_empty_ignore_list_uses_legacy_path(self, monkeypatch):
        # An empty ``modules_to_not_convert`` is falsy -> legacy per-name matching.
        marker = object()

        def _should_not_run(*_a, **_k):
            raise AssertionError("ignore-list path must not be taken for empty list")

        monkeypatch.setattr(qf, "_quantize_params_fp8_by_ignore_list", _should_not_run)
        config = {**self._BASE_CONFIG, "modules_to_not_convert": []}

        # A non-matching megatron name returns the params unchanged (legacy behaviour).
        converted = [("unmatched", marker)]
        out = qf.quantize_params_fp8(
            args=None,
            megatron_name="not.a.decoder.layer",
            converted_named_params=converted,
            quantization_config=config,
        )
        assert out is converted

    def test_no_key_uses_legacy_path(self, monkeypatch):
        monkeypatch.setattr(
            qf,
            "_quantize_params_fp8_by_ignore_list",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        converted = [("x", object())]
        out = qf.quantize_params_fp8(
            args=None,
            megatron_name="not.a.decoder.layer",
            converted_named_params=converted,
            quantization_config=dict(self._BASE_CONFIG),
        )
        assert out is converted
