# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The chunkflow patch must stay off unless RELAX_CHUNKFLOW_CHUNK_SIZE says
otherwise."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.chunkflow.integration import ChunkedCoreAttention


class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.config = SimpleNamespace(context_parallel_size=1)
        self.core_attention = torch.nn.Identity()


def _patched_probe(monkeypatch, env_value):
    """Import the patch with a stubbed hook target and return (probe,
    calls)."""
    calls = []
    fake_mp = SimpleNamespace(_install_cp_probe=lambda model: calls.append(model))
    monkeypatch.setitem(__import__("sys").modules, "relax.backends.megatron.model_provider", fake_mp)
    if env_value is None:
        monkeypatch.delenv("RELAX_CHUNKFLOW_CHUNK_SIZE", raising=False)
    else:
        monkeypatch.setenv("RELAX_CHUNKFLOW_CHUNK_SIZE", env_value)

    module = importlib.import_module("relax.backends.megatron.megatron_patch.chunkflow_attention_patch")
    importlib.reload(module)
    return fake_mp._install_cp_probe, calls


def _is_chunked(model):
    return isinstance(model[0].core_attention, ChunkedCoreAttention)


@pytest.mark.parametrize("env_value", [None, "", "0", "-4"])
def test_chunkflow_patch_off_by_default_or_nonpositive(monkeypatch, env_value):
    probe, calls = _patched_probe(monkeypatch, env_value)
    model = torch.nn.Sequential(_FakeAttn())
    probe(model)
    assert not _is_chunked(model)
    assert calls == [model], "the original cp probe must still run"


def test_chunkflow_patch_installs_when_enabled(monkeypatch):
    probe, calls = _patched_probe(monkeypatch, "512")
    model = torch.nn.Sequential(_FakeAttn(), _FakeAttn())
    probe(model)
    assert _is_chunked(model)
    assert model[0].core_attention.chunk_size == 512
    assert calls == [model]


def test_chunkflow_patch_ignores_malformed_value(monkeypatch):
    """A typo must not crash training; it degrades to off."""
    probe, _calls = _patched_probe(monkeypatch, "not-a-number")
    model = torch.nn.Sequential(_FakeAttn())
    probe(model)
    assert not _is_chunked(model)
