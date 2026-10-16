# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify the installed FA3 API mapping without loading CUDA extensions."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


PATH = Path(__file__).parents[2] / "relax/backends/megatron/chunkflow/flash_backends.py"
SPEC = importlib.util.spec_from_file_location("chunkflow_flash_backends_test_target", PATH)
BACKENDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKENDS)


class FlashBackendTests(unittest.TestCase):
    def tearDown(self):
        BACKENDS.flash3_varlen_ops.cache_clear()

    def test_selector_defaults_to_fa2_and_rejects_ambiguous_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(BACKENDS.flash_backend_name(), "fa2")
        with patch.dict(os.environ, {"RELAX_CHUNKFLOW_FLASH_BACKEND": "auto"}):
            with self.assertRaises(ValueError):
                BACKENDS.flash_backend_name()

    def test_adapter_preserves_lse_maxima_causal_and_gradient_buffers(self):
        native = ModuleType("flash_attn_3.flash_attn_interface")
        output, lse = object(), object()
        native._flash_attn_forward = Mock(return_value=(output, lse, None, None))
        native._flash_attn_backward = Mock()
        with patch.dict(sys.modules, {"flash_attn_3.flash_attn_interface": native}):
            forward, backward = BACKENDS.flash3_varlen_ops()
        q, k, v, cu_q, cu_kv = (object() for _ in range(5))
        got = forward(q, k, v, cu_q, cu_kv, 7, 11, 0, 0.125, True)
        self.assertIs(got[0], output)
        self.assertIs(got[1], lse)
        f = native._flash_attn_forward.call_args
        self.assertEqual(f.args, (q, k, v))
        self.assertEqual((f.kwargs["max_seqlen_q"], f.kwargs["max_seqlen_k"]), (7, 11))
        self.assertIs(f.kwargs["cu_seqlens_q"], cu_q)
        self.assertIs(f.kwargs["cu_seqlens_k"], cu_kv)
        self.assertTrue(f.kwargs["causal"])
        dout, dq, dk, dv = (object() for _ in range(4))
        backward(dout, q, k, v, output, lse, dq, dk, dv, cu_q, cu_kv, 7, 11, 0, 0.125, True, -1, -1, 0, None, False)
        b = native._flash_attn_backward.call_args
        self.assertEqual(b.args, (dout, q, k, v, output, lse))
        self.assertTrue(b.kwargs["is_causal"])
        self.assertEqual((b.kwargs["max_seqlen_q"], b.kwargs["max_seqlen_k"]), (7, 11))
        for key, value in (("dq", dq), ("dk", dk), ("dv", dv)):
            self.assertIs(b.kwargs[key], value)
        with self.assertRaises(ValueError):
            forward(q, k, v, cu_q, cu_kv, 7, 11, 0.1, 0.125, True)

    def test_explicit_fa3_missing_module_fails_without_top_level_fallback(self):
        with patch.dict(
            sys.modules,
            {"flash_attn_3.flash_attn_interface": None, "flash_attn_interface": ModuleType("flash_attn_interface")},
        ):
            with self.assertRaisesRegex(RuntimeError, "explicitly requested fa3"):
                BACKENDS.flash3_varlen_ops()


if __name__ == "__main__":
    unittest.main()
