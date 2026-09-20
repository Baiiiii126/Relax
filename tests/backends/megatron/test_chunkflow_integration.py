# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ChunkedCoreAttention must match unchunked SDPA for any chunk size."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.backends.megatron.chunkflow.integration import (
    ChunkedCoreAttention,
    install,
    split_sizes,
    uninstall,
)


def _sdpa_reference(q, k, v):
    """q/k/v: [T, H, D] -> [T, H*D], plain causal attention."""
    qh, kh, vh = (t.permute(1, 0, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    return out.permute(1, 0, 2).reshape(q.shape[0], -1)


def _qkv(t, h, d, seed=0, kv_heads=None):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(t, h, d, generator=g, dtype=torch.float64)
    kv_h = kv_heads or h
    k = torch.randn(t, kv_h, d, generator=g, dtype=torch.float64)
    v = torch.randn(t, kv_h, d, generator=g, dtype=torch.float64)
    return q, k, v


def _attn(chunk_size, cp_size=1):
    return ChunkedCoreAttention(
        config=SimpleNamespace(context_parallel_size=cp_size), layer_number=1, chunk_size=chunk_size
    )


def test_chunkflow_split_sizes_covers_total_with_tail():
    assert split_sizes(12, 4) == [4, 4, 4]
    assert split_sizes(10, 4) == [4, 4, 2]
    assert split_sizes(3, 8) == [3]
    assert sum(split_sizes(37, 5)) == 37


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 8, 16])
def test_chunkflow_chunked_attention_matches_sdpa(chunk_size):
    q, k, v = _qkv(t=8, h=2, d=4)
    got = _attn(chunk_size)(q, k, v)
    torch.testing.assert_close(got, _sdpa_reference(q, k, v), rtol=1e-10, atol=1e-10)


def test_chunkflow_chunked_attention_matches_sdpa_with_gqa():
    q, k, v = _qkv(t=8, h=4, d=4, kv_heads=2)
    got = _attn(chunk_size=3)(q, k, v)
    k_wide = k.repeat_interleave(2, dim=1)
    v_wide = v.repeat_interleave(2, dim=1)
    torch.testing.assert_close(got, _sdpa_reference(q, k_wide, v_wide), rtol=1e-10, atol=1e-10)


def test_chunkflow_packed_sequences_are_attended_independently():
    """Two packed sequences must not attend across their boundary."""
    q, k, v = _qkv(t=10, h=2, d=4)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 6, 10]), cu_seqlens_q_cpu=[0, 6, 10])
    got = _attn(chunk_size=4)(q, k, v, packed_seq_params=packed)
    want = torch.cat([_sdpa_reference(q[:6], k[:6], v[:6]), _sdpa_reference(q[6:], k[6:], v[6:])], dim=0)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_chunkflow_backward_matches_sdpa():
    q, k, v = _qkv(t=8, h=2, d=4)
    q_ref, k_ref, v_ref = (t.clone().requires_grad_(True) for t in (q, k, v))
    q_chk, k_chk, v_chk = (t.clone().requires_grad_(True) for t in (q, k, v))

    _sdpa_reference(q_ref, k_ref, v_ref).sum().backward()
    _attn(chunk_size=3)(q_chk, k_chk, v_chk).sum().backward()

    for got, want in ((q_chk, q_ref), (k_chk, k_ref), (v_chk, v_ref)):
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-9, atol=1e-9)


def test_chunkflow_context_parallel_is_rejected():
    q, k, v = _qkv(t=4, h=2, d=4)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _attn(chunk_size=2, cp_size=2)(q, k, v)


def test_chunkflow_sbhd_is_rejected():
    with pytest.raises(NotImplementedError, match="THD"):
        _attn(chunk_size=2)(torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4))


def test_chunkflow_install_swaps_and_uninstall_restores():
    class FakeAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 1
            self.config = SimpleNamespace(context_parallel_size=1)
            self.core_attention = torch.nn.Identity()

    model = torch.nn.Sequential(FakeAttn(), FakeAttn())
    original = model[0].core_attention

    assert install(model, chunk_size=4) == 2
    assert isinstance(model[0].core_attention, ChunkedCoreAttention)
    assert install(model, chunk_size=4) == 0  # idempotent

    assert uninstall(model) == 2
    assert model[0].core_attention is original
