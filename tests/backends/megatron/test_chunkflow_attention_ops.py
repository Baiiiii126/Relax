# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit attention gradients, backend adapters and CPU length metadata."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime, require_runtime
except ImportError as exc:
    pytest.skip(f"ChunkFlow attention dependencies unavailable: {exc}", allow_module_level=True)


def _offset_causal_mask(ql, kl):
    qpos = torch.arange(ql).unsqueeze(1) + (kl - ql)
    kpos = torch.arange(kl).unsqueeze(0)
    return (kpos <= qpos).unsqueeze(0).unsqueeze(0)


def _unit_attention(q, k, v, cu_q, cu_kv, causal=True, max_seqlens=None):
    from relax.backends.megatron.chunkflow.attention_ops import UnitCacheAttention

    parts = [k[s:e] for s, e in zip(cu_kv[:-1], cu_kv[1:])]
    vparts = [v[s:e] for s, e in zip(cu_kv[:-1], cu_kv[1:])]
    scale = q.shape[-1] ** -0.5
    metadata = () if max_seqlens is None else (max_seqlens,)
    return UnitCacheAttention.apply(q, causal, scale, cu_q, cu_kv, *parts, *vparts, *metadata)[0]


@pytest.mark.parametrize(
    "dtype,heads,kv_heads,max_seqlens",
    [
        (torch.float32, 2, 2, None),
        (torch.float64, 4, 2, (10, 30)),
    ],
)
def test_unit_attention_output_and_gradients_match_sdpa(dtype, heads, kv_heads, max_seqlens):
    torch.manual_seed(0)
    cu_q = torch.tensor([0, 10, 20], dtype=torch.int32)
    cu_kv = torch.tensor([0, 30, 44], dtype=torch.int32)
    q = torch.randn(20, heads, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(44, kv_heads, 8, dtype=dtype, requires_grad=True)
    v = torch.randn(44, kv_heads, 8, dtype=dtype, requires_grad=True)
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    actual = _unit_attention(q, k, v, cu_q, cu_kv, max_seqlens=max_seqlens)
    expected = _varlen_reference(qr, kr, vr, cu_q, cu_kv)
    dout = torch.randn_like(actual)
    dout[10:] = 0
    actual.backward(dout)
    expected.backward(dout)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for got, want in zip((q, k, v), (qr, kr, vr)):
        torch.testing.assert_close(got.grad, want.grad, atol=tolerance, rtol=tolerance)

    assert (
        torch.count_nonzero(q.grad[10:]) == torch.count_nonzero(k.grad[30:]) == torch.count_nonzero(v.grad[30:]) == 0
    )
    assert torch.count_nonzero(k.grad[:30]) > 0


def _varlen_reference(q, k, v, cu_q, cu_kv):
    repeat = q.shape[1] // k.shape[1]
    parts = []
    for qs, qe, ks, ke in zip(cu_q[:-1], cu_q[1:], cu_kv[:-1], cu_kv[1:]):
        out = F.scaled_dot_product_attention(
            q[qs:qe].permute(1, 0, 2).unsqueeze(0),
            k[ks:ke].repeat_interleave(repeat, dim=1).permute(1, 0, 2).unsqueeze(0),
            v[ks:ke].repeat_interleave(repeat, dim=1).permute(1, 0, 2).unsqueeze(0),
            attn_mask=_offset_causal_mask(qe - qs, ke - ks),
            scale=q.shape[-1] ** -0.5,
        )
        parts.append(out.squeeze(0).permute(1, 0, 2))
    return torch.cat(parts)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_unit_flash_reuses_cpu_maxima_in_forward_and_backward(monkeypatch, with_metadata):
    from unittest.mock import Mock

    from relax.backends.megatron.chunkflow import attention_ops as ao

    calls = []

    def forward(q, k, v, cu_q, cu_kv, max_q, max_kv, dropout, scale, causal):
        calls.append(("forward", max_q, max_kv))
        out, lse = ao._manual_varlen_forward(q, k, v, cu_q, cu_kv, causal, scale)
        return out, lse, None, None

    def backward(dout, q, k, v, out, lse, dq, dk, dv, cu_q, cu_kv, max_q, max_kv, dropout, scale, causal, *rest):
        calls.append(("backward", max_q, max_kv))
        gradients = ao._manual_varlen_backward(dout, q, k, v, lse, cu_q, cu_kv, causal, scale)
        for target, gradient in zip((dq, dk, dv), gradients):
            target.copy_(gradient)

    read_max = Mock(wraps=ao._max_seqlen)
    if with_metadata:
        read_max.side_effect = AssertionError("CPU metadata must bypass GPU scalar length reads")
    monkeypatch.setattr(ao, "_max_seqlen", read_max)
    monkeypatch.setattr(ao, "_use_flash", lambda q: True)
    monkeypatch.setattr(ao, "_flash_varlen_ops", lambda: (forward, backward))
    torch.manual_seed(17)
    cu_q, cu_kv = torch.tensor([0, 3, 10]), torch.tensor([0, 11, 20])
    q, k, v = (torch.randn(n, h, 8, dtype=torch.float64, requires_grad=True) for n, h in ((10, 4), (20, 2), (20, 2)))
    dout = torch.randn_like(q)
    result = _unit_attention(q, k, v, cu_q, cu_kv, max_seqlens=(7, 11) if with_metadata else None)
    result.backward(dout)
    assert calls == [("forward", 7, 11), ("backward", 7, 11)]
    assert read_max.call_count == (0 if with_metadata else 2)


@pytest.fixture
def fa3_backend():
    from relax.backends.megatron.chunkflow import flash_backends

    flash_backends.flash3_varlen_ops.cache_clear()
    try:
        yield flash_backends
    finally:
        flash_backends.flash3_varlen_ops.cache_clear()


@pytest.mark.parametrize("chunk_api", [False, True])
def test_fa3_adapter_preserves_gqa_gradients_and_forward_backend(monkeypatch, chunk_api, fa3_backend):
    import sys
    from types import ModuleType

    from relax.backends.megatron.chunkflow import attention_ops as ao

    native = ModuleType("flash_attn_3.flash_attn_interface")
    calls = []
    maxima = (10, 20) if chunk_api else (7, 11)

    def forward(q, k, v, **kwargs):
        assert (kwargs["max_seqlen_q"], kwargs["max_seqlen_k"]) == maxima
        calls.append("fa3-forward")
        out, lse = ao._manual_varlen_forward(
            q, k, v, kwargs["cu_seqlens_q"], kwargs["cu_seqlens_k"], kwargs["causal"], kwargs["softmax_scale"]
        )
        return out, lse, None, None

    def backward(dout, q, k, v, out, lse, **kwargs):
        assert (kwargs["max_seqlen_q"], kwargs["max_seqlen_k"]) == maxima
        calls.append("fa3-backward")
        gradients = ao._manual_varlen_backward(
            dout,
            q,
            k,
            v,
            lse,
            kwargs["cu_seqlens_q"],
            kwargs["cu_seqlens_k"],
            kwargs["is_causal"],
            kwargs["softmax_scale"],
        )
        for name, gradient in zip(("dq", "dk", "dv"), gradients):
            kwargs[name].copy_(gradient)

    native._flash_attn_forward, native._flash_attn_backward = forward, backward
    monkeypatch.setitem(sys.modules, "flash_attn_3.flash_attn_interface", native)
    monkeypatch.setenv("RELAX_CHUNKFLOW_FLASH_BACKEND", "fa3")
    monkeypatch.setattr(ao, "_use_flash", lambda q: True)

    def forbidden(*args):
        raise AssertionError("must reuse CPU maxima and the forward backend")

    monkeypatch.setattr(ao, "_max_seqlen", forbidden)
    q, k, v = (torch.randn(n, h, 8, dtype=torch.float64, requires_grad=True) for n, h in ((10, 4), (20, 2), (20, 2)))
    qr, kr, vr = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    cu_q, cu_kv = torch.tensor([0, 3, 10]), torch.tensor([0, 11, 20])
    if chunk_api:
        result, _ = ao.ChunkCacheAttention.apply(q, True, 8**-0.5, 1, k, v)
    else:
        result = _unit_attention(q, k, v, cu_q, cu_kv, max_seqlens=(7, 11))
    monkeypatch.setenv("RELAX_CHUNKFLOW_FLASH_BACKEND", "fa2")
    monkeypatch.setattr(ao, "_flash_varlen_ops", forbidden)
    dout = torch.randn_like(q)
    result.backward(dout)
    if chunk_api:
        cu_q, cu_kv = torch.tensor([0, 10]), torch.tensor([0, 20])
    expected = _varlen_reference(qr, kr, vr, cu_q, cu_kv)
    expected.backward(dout)
    torch.testing.assert_close(result, expected, atol=1e-9, rtol=1e-9)
    for actual, wanted in zip((q, k, v), (qr, kr, vr)):
        torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-9, rtol=1e-9)
    assert calls == ["fa3-forward", "fa3-backward"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA3 numeric validation requires a CUDA GPU")
@pytest.mark.parametrize(
    "bounds_q,bounds_k,heads,kv_heads,dim",
    [([0, 17, 26], [0, 40, 53], 16, 4, 128), ([0, 6144, 8192], [0, 6144, 8192], 8, 1, 256)],
)
def test_unit_fa3_matches_cuda_reference(monkeypatch, fa3_backend, bounds_q, bounds_k, heads, kv_heads, dim):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires the Hopper FA3 build used by this experiment")
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("FA2 reference kernel is not installed")

    from relax.backends.megatron.chunkflow import flash_backends as fb

    monkeypatch.setenv("RELAX_CHUNKFLOW_FLASH_BACKEND", "fa3")
    try:
        fb.flash3_varlen_ops()
    except RuntimeError as exc:
        if isinstance(exc.__cause__, ImportError):
            pytest.skip("optional FA3 interface is not installed")
        raise
    max_q = max(b - a for a, b in zip(bounds_q[:-1], bounds_q[1:]))
    max_k = max(b - a for a, b in zip(bounds_k[:-1], bounds_k[1:]))
    cu_q = torch.tensor(bounds_q, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(bounds_k, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q, k, v = (
        torch.randn(n, h, dim, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
        for n, h in ((bounds_q[-1], heads), (bounds_k[-1], kv_heads), (bounds_k[-1], kv_heads))
    )
    qr, kr, vr = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    parts = [k[a:b] for a, b in zip(bounds_k[:-1], bounds_k[1:])]
    values = [v[a:b] for a, b in zip(bounds_k[:-1], bounds_k[1:])]
    from relax.backends.megatron.chunkflow.attention_ops import UnitCacheAttention

    result, _ = UnitCacheAttention.apply(q, True, dim**-0.5, cu_q, cu_k, *parts, *values, (max_q, max_k))
    expected = flash_attn_varlen_func(qr, kr, vr, cu_q, cu_k, max_q, max_k, causal=True)
    dout = torch.randn_like(result)
    result.backward(dout)
    expected.backward(dout)
    for actual, wanted in ((result, expected), (q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        torch.testing.assert_close(actual, wanted, atol=0.02, rtol=0.02)
        assert (actual.float() - wanted.float()).norm() / wanted.float().norm() < 0.03


def test_schedule_reuses_runtime_installed_below_model_wrappers():
    class Wrapper:
        def __init__(self, module):
            self.module = module

    bare = Wrapper(None)
    runtime = ChunkflowRuntime(chunk_size=16)
    bare.chunkflow_runtime = runtime
    wrapped = Wrapper(Wrapper(bare))

    assert require_runtime(wrapped) is runtime
    assert not hasattr(wrapped, "chunkflow_runtime")
