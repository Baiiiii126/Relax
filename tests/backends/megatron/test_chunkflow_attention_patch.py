# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The chunkflow patch must stay off unless asked for, and CLI args must
outrank the ``RELAX_CHUNKFLOW_CHUNK_SIZE`` fallback.

Two halves, deliberately kept apart:

1. ``_chunk_size()`` resolution — pure logic, driven by stubbing
   ``megatron.training.global_vars`` in ``sys.modules`` plus the env var.
2. The probe wrapper — driven by monkeypatching the module globals the
   wrapper closes over (``_ORIGINAL_INSTALL_CP_PROBE`` and ``_chunk_size``).

The wrapper is NOT tested by stubbing ``relax.backends.megatron.model_provider``
in ``sys.modules``: the patch does ``from relax.backends.megatron import
model_provider``, which is an *attribute* lookup on an already-imported package
whenever the real one is present (as it is in the training container), so the
stub would be silently ignored and the test would exercise nothing.
"""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from relax.backends.megatron.chunkflow.integration import ChunkedCoreAttention


try:
    from relax.backends.megatron.megatron_patch import chunkflow_attention_patch as patch
except Exception as exc:  # pragma: no cover - environment without megatron
    pytest.skip(f"chunkflow_attention_patch unavailable: {exc}", allow_module_level=True)


# The wrapper only exists if the patch's own import of model_provider succeeded.
requires_patch = pytest.mark.skipif(
    not hasattr(patch, "_ORIGINAL_INSTALL_CP_PROBE"),
    reason="chunkflow patch did not install (model_provider import failed)",
)


class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.config = SimpleNamespace(context_parallel_size=1)
        self.core_attention = torch.nn.Identity()


def _stub_args(monkeypatch, args):
    """Make the patch's ``from megatron.training.global_vars import get_args``
    resolve to a stub returning ``args`` (or raising, if ``args`` is an
    exception)."""

    def get_args():
        if isinstance(args, Exception):
            raise args
        return args

    global_vars = ModuleType("megatron.training.global_vars")
    global_vars.get_args = get_args
    training = ModuleType("megatron.training")
    training.global_vars = global_vars
    megatron = ModuleType("megatron")
    megatron.training = training

    import sys

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars)


def _set_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RELAX_CHUNKFLOW_CHUNK_SIZE", raising=False)
    else:
        monkeypatch.setenv("RELAX_CHUNKFLOW_CHUNK_SIZE", value)


# ---------------------------------------------------------------------------
# 1. _chunk_size() resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_value", [None, "", "0", "-4", "not-a-number"])
def test_chunk_size_is_zero_without_args_or_usable_env(monkeypatch, env_value):
    """No args and no (usable) env value means off.

    A malformed value must degrade to off rather than crash training.
    """
    _stub_args(monkeypatch, RuntimeError("args not set yet"))
    _set_env(monkeypatch, env_value)
    assert patch._chunk_size() <= 0


def test_chunk_size_falls_back_to_env_when_args_unavailable(monkeypatch):
    """Before ``set_args()`` (e.g. a standalone script) the env var is the only
    switch available."""
    _stub_args(monkeypatch, RuntimeError("args not set yet"))
    _set_env(monkeypatch, "128")
    assert patch._chunk_size() == 128


def test_chunk_size_prefers_args_over_env(monkeypatch):
    _stub_args(monkeypatch, SimpleNamespace(use_chunkflow=True, chunkflow_chunk_size=256))
    _set_env(monkeypatch, "64")
    assert patch._chunk_size() == 256


def test_chunk_size_args_off_beats_env_on(monkeypatch):
    """``--use-chunkflow`` absent means off: a stale env var must not switch
    chunkflow on behind the back of a run that did not ask for it."""
    _stub_args(monkeypatch, SimpleNamespace(use_chunkflow=False, chunkflow_chunk_size=256))
    _set_env(monkeypatch, "64")
    assert patch._chunk_size() == 0


# ---------------------------------------------------------------------------
# 2. The probe wrapper
# ---------------------------------------------------------------------------


def _run_probe(monkeypatch, chunk_size, model):
    """Drive the wrapper with a recording original probe and a fixed chunk
    size; returns what the original probe saw."""
    calls = []
    monkeypatch.setattr(patch, "_ORIGINAL_INSTALL_CP_PROBE", calls.append)
    monkeypatch.setattr(patch, "_chunk_size", lambda: chunk_size)
    patch._install_cp_probe_with_chunkflow(model)
    return calls


@requires_patch
@pytest.mark.parametrize("chunk_size", [0, -4])
def test_probe_leaves_attention_alone_when_off(monkeypatch, chunk_size):
    model = torch.nn.Sequential(_FakeAttn())
    calls = _run_probe(monkeypatch, chunk_size, model)
    assert not isinstance(model[0].core_attention, ChunkedCoreAttention)
    assert calls == [model], "the original cp probe must still run"


@requires_patch
def test_probe_installs_chunked_attention_when_enabled(monkeypatch):
    model = torch.nn.Sequential(_FakeAttn(), _FakeAttn())
    calls = _run_probe(monkeypatch, 512, model)
    assert all(isinstance(m.core_attention, ChunkedCoreAttention) for m in model)
    assert model[0].core_attention.chunk_size == 512
    assert calls == [model]
