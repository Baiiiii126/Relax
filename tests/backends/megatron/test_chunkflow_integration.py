# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Core attention execution, recomputation and installation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.backends.megatron.chunkflow.integration import (
    ChunkedCoreAttention,
    install,
    uninstall,
)
from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.units import build_micro_batches


@pytest.fixture(autouse=True)
def restore_rope_patch():
    from relax.backends.megatron.chunkflow import rope

    original = rope._ORIGINAL_GET_ROTARY_SEQ_LEN
    if original is not None:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        patched_method = RotaryEmbedding.get_rotary_seq_len
    try:
        yield
    finally:
        if original is None:
            rope.uninstall()
        else:
            rope._ORIGINAL_GET_ROTARY_SEQ_LEN = original
            RotaryEmbedding.get_rotary_seq_len = patched_method


class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.config = SimpleNamespace(context_parallel_size=1)
        self.core_attention = torch.nn.Identity()


def _sdpa_reference(q, k, v):
    repeat = q.shape[1] // k.shape[1]
    qh, kh, vh = (
        t.permute(1, 0, 2) for t in (q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1))
    )
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


@pytest.mark.parametrize("chunk_size,heads,kv_heads", [(size, 2, 2) for size in (1, 3, 16)] + [(3, 4, 2)])
def test_chunkflow_output_and_gradients_match_sdpa(chunk_size, heads, kv_heads):
    q, k, v = _qkv(t=8, h=heads, d=4, kv_heads=kv_heads)
    reference = [t.clone().requires_grad_(True) for t in (q, k, v)]
    chunked = [t.clone().requires_grad_(True) for t in (q, k, v)]
    expected = _sdpa_reference(*reference)
    actual = _attn(chunk_size)(*chunked)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    expected.sum().backward()
    actual.sum().backward()
    for got, want in zip(chunked, reference):
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-9, atol=1e-9)


def test_chunkflow_packed_sequences_are_attended_independently():
    q, k, v = _qkv(t=10, h=2, d=4)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 6, 10]), cu_seqlens_q_cpu=[0, 6, 10])
    got = _attn(chunk_size=4)(q, k, v, packed_seq_params=packed)
    want = torch.cat([_sdpa_reference(q[:6], k[:6], v[:6]), _sdpa_reference(q[6:], k[6:], v[6:])], dim=0)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_chunkflow_context_parallel_is_rejected():
    q, k, v = _qkv(t=4, h=2, d=4)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _attn(chunk_size=2, cp_size=2)(q, k, v)


def test_chunkflow_sbhd_is_rejected():
    with pytest.raises(NotImplementedError, match="THD"):
        _attn(chunk_size=2)(torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunkflow_matches_flash_varlen_on_training_shapes():
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("flash_attn not installed")

    h, kv_h, d = 16, 4, 128
    lengths = [3000, 1096]
    total = sum(lengths)
    bounds = [0, lengths[0], total]

    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(total, h, d, generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, kv_h, d, generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, kv_h, d, generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(total, h * d, generator=g, device="cuda", dtype=torch.bfloat16)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor(bounds, device="cuda", dtype=torch.int32))

    def run(fn):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        fn(*leaves).backward(dout)
        return [leaf.grad for leaf in leaves]

    def chunked(ql, kl, vl):
        return _attn(chunk_size=1024)(ql, kl, vl, packed_seq_params=packed)

    def reference(ql, kl, vl):
        cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
        out = flash_attn_varlen_func(ql, kl, vl, cu, cu, max(lengths), max(lengths), causal=True)
        return out.reshape(total, -1)

    rels = {
        name: ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()
        for name, got, want in zip(("dq", "dk", "dv"), run(chunked), run(reference))
    }
    assert max(rels.values()) < 2e-2, f"chunked differs from flash_attn_varlen_func: {rels}"


def test_chunkflow_install_swaps_and_uninstall_restores(monkeypatch):
    from unittest.mock import Mock

    from relax.backends.megatron.chunkflow import rope

    install_rope = Mock(wraps=rope.install)
    monkeypatch.setattr(rope, "install", install_rope)

    model = torch.nn.Sequential(_FakeAttn(), _FakeAttn())
    original = model[0].core_attention
    model[0].config.apply_rope_fusion = True

    assert install(model, chunk_size=4) == 2
    assert isinstance(model[0].core_attention, ChunkedCoreAttention)
    assert install_rope.call_count == 1
    assert model[0].config.apply_rope_fusion is False
    assert install(model, chunk_size=4) == 0
    assert install_rope.call_count == 2

    assert uninstall(model) == 2
    assert model[0].core_attention is original


def _unit_attn(chunk_size, layer=1):

    return ChunkedCoreAttention(
        config=SimpleNamespace(context_parallel_size=1),
        layer_number=layer,
        chunk_size=chunk_size,
        runtime=ChunkflowRuntime(chunk_size),
    )


@pytest.mark.parametrize("fp32", [False, True])
def test_pooled_tail_attention_preserves_prefix_outputs_and_gradients(fp32, monkeypatch):
    from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR
    from relax.backends.megatron.chunkflow.schedule_model import make_model_plan
    from relax.backends.megatron.chunkflow.tail_pool import pool_terminal_units
    from relax.backends.megatron.chunkflow.units import completed_groups

    monkeypatch.setenv("RELAX_CHUNKFLOW_KV_GRAD_FP32", str(int(fp32)))
    lengths, chunk_size = [5, 8, 10], 4
    reference = [t.requires_grad_(True) for t in _qkv(sum(lengths), 4, 8, kv_heads=2)]
    actual = [t.detach().clone().requires_grad_(True) for t in reference]
    starts = [0, 5, 13]
    expected = torch.cat([_sdpa_reference(*(t[s : s + n] for t in reference)) for s, n in zip(starts, lengths)])
    grad = torch.zeros_like(expected)
    for s, n in zip(starts, lengths):
        grad[s + n - 1] = 1
    expected.backward(grad)
    chunks = build_chunks(lengths, chunk_size, repack_tail=False)
    units = build_micro_batches(chunks)
    plan = make_model_plan(units, chunks, 1 << 30, 1)
    units, members, plan = pool_terminal_units(
        {u.unit_id: u for u in units}, {u.unit_id: 0 for u in units}, plan, chunk_size
    )
    assert any(len(ids) > 1 for ids in members.values())
    attn = _unit_attn(chunk_size)
    outputs, indices = {}, {}
    for uid in plan.forward:
        unit = units[uid]
        ix, cu = [], [0]
        for c in unit.chunks:
            for s in c.segments:
                ix.extend(range(starts[s.seq_id] + s.prefix, starts[s.seq_id] + s.prefix + s.length))
                cu.append(cu[-1] + s.length)
        packed = SimpleNamespace(cu_seqlens_q_cpu=cu, **{UNIT_ATTR: unit})
        outputs[uid] = attn(*(t[ix] for t in actual), packed_seq_params=packed)
        indices[uid] = ix
        torch.testing.assert_close(outputs[uid], expected[ix], atol=1e-9, rtol=1e-9)
    for uid in plan.backward:
        outputs[uid].backward(grad[indices[uid]])
        for group in completed_groups(units[uid]):
            attn.runtime.release_group(group)
    attn.runtime.assert_empty("pooled tails")
    for got, want in zip(actual, reference):
        torch.testing.assert_close(got.grad, want.grad, atol=1e-7 if fp32 else 1e-9, rtol=1e-6 if fp32 else 1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Transformer Engine")
def test_te_empty_expert_preserves_accumulated_gradient():
    te = pytest.importorskip("transformer_engine.pytorch")
    model = te.GroupedLinear(
        2, 16, 16, bias=False, params_dtype=torch.float32, fuse_wgrad_accumulation=True, device="cuda"
    )
    weights = [model.weight0, model.weight1]
    for weight in weights:
        weight.main_grad = torch.zeros_like(weight)
    inputs = [torch.arange(n * 16, device="cuda", dtype=torch.float32).reshape(n, 16) / 16 for n in (2, 3)]
    outputs = [model(inputs[0], [2, 0], is_first_microbatch=None), model(inputs[1], [0, 3], is_first_microbatch=None)]
    outputs[1].sum().backward()
    accumulated = weights[1].main_grad.clone()
    outputs[0].sum().backward()
    torch.testing.assert_close(weights[1].main_grad, accumulated, rtol=0, atol=0)
    for weight, x in zip(weights, inputs):
        expected = x.sum(dim=0).expand_as(weight)
        torch.testing.assert_close(weight.main_grad, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("original_cache_disabled", [False, True])
def test_reversed_backward_preserves_fused_weight_gradients(original_cache_disabled):
    class FusedLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, first):
            ctx.save_for_backward(x, weight)
            ctx.first = first
            return x @ weight.T

        @staticmethod
        def backward(ctx, grad):
            x, weight = ctx.saved_tensors
            wgrad = grad.T @ x
            # Transformer Engine overwrites main_grad when first=True.
            if ctx.first is True:
                weight.main_grad.copy_(wgrad)
            else:
                weight.main_grad.add_(wgrad)
            return grad @ weight, None, None

    model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
    model.disable_parameter_transpose_cache = original_cache_disabled
    model.is_first_microbatch = True
    model.weight.main_grad = torch.zeros_like(model.weight)
    inputs = [torch.arange(12, dtype=torch.float64).reshape(4, 3) + offset for offset in (1, 3)]
    reference = model.weight.detach().clone().requires_grad_()
    sum((x @ reference.T).square().sum() for x in inputs).backward()

    install(model, 4)
    install(model, 4)
    outputs = []
    for x in inputs:
        first = None if model.disable_parameter_transpose_cache else model.is_first_microbatch
        outputs.append(FusedLinear.apply(x, model.weight, first))
        model.is_first_microbatch = False
    for out in reversed(outputs):
        out.square().sum().backward()
    torch.testing.assert_close(model.weight.main_grad, reference.grad)
    uninstall(model)
    assert model.disable_parameter_transpose_cache is original_cache_disabled


def test_unit_mode_passes_cpu_lengths_with_prefix_and_padding(monkeypatch):
    from relax.backends.megatron.chunkflow.attention_ops import UnitCacheAttention
    from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR

    recorded = []
    metadata = []
    original = UnitCacheAttention.apply

    def capture(*args):
        recorded.append((args[0].shape[0], args[-1]))
        metadata.append(args[3:5])
        fragments = args[5:-1]
        assert len(fragments) == (2 if args[0].shape[0] == 8 else 4)
        assert all(t.grad_fn is None or "CatBackward" not in t.grad_fn.name() for t in fragments)
        return original(*args)

    monkeypatch.setattr(UnitCacheAttention, "apply", capture)
    attn, next_layer = _unit_attn(8), _unit_attn(8, layer=2)
    units = build_micro_batches(build_chunks([13], chunk_size=8, repack_tail=False), "staggered")
    for unit in units:
        length = sum(seg.length for chunk in unit.chunks for seg in chunk.segments)
        cu_q = torch.tensor([0, length], dtype=torch.int32)
        packed = SimpleNamespace(cu_seqlens_q_cpu=[0, length], cu_seqlens_q=cu_q, **{UNIT_ATTR: unit})
        q, k, v = (t.requires_grad_() for t in _qkv(length + 3, 4, 8, kv_heads=2))
        output = attn(q, k, v, packed_seq_params=packed)
        other = next_layer(q, k, v, packed_seq_params=packed)
        torch.testing.assert_close(output, other)
        assert metadata[-1][0] is metadata[-2][0] is cu_q
        assert metadata[-1][1] is metadata[-2][1]
        assert output.shape[0] == length + 3
    assert recorded == [(8, (8, 8)), (8, (8, 8)), (5, (5, 13)), (5, (5, 13))]


@pytest.mark.parametrize("keep_last_k", [1 << 30, 1])
@pytest.mark.parametrize("fp32", [False, True])
def test_unit_schedule_outputs_gradients_and_release_match_sdpa(keep_last_k, fp32, monkeypatch):
    monkeypatch.setenv("RELAX_CHUNKFLOW_KV_GRAD_FP32", str(int(fp32)))
    from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR
    from relax.backends.megatron.chunkflow.units import completed_groups, gen_unit_ops

    torch.manual_seed(0)
    total, chunk_size = 64, 16
    reference = [t.requires_grad_(True) for t in _qkv(total, 4, 8, kv_heads=2)]
    chunked = [t.detach().clone().requires_grad_(True) for t in reference]
    expected_output = _sdpa_reference(*reference)
    dout = torch.randn_like(expected_output)
    expected_output.backward(dout)
    attn = _unit_attn(chunk_size)
    chunks = build_chunks([total], chunk_size=chunk_size, repack_tail=False)
    units = build_micro_batches(chunks, "staggered")
    ops = gen_unit_ops(units, chunks, keep_last_k, pp=1, stage=0)
    expected_fr = set(range(len(units) - 1)) if keep_last_k == 1 else set()
    assert {uid for kind, uid in ops if kind == "Fp"} == {uid for kind, uid in ops if kind == "Fr"} == expected_fr

    pending, outputs = {}, {}
    for kind, uid in ops:
        unit = units[uid]
        start = unit.chunks[0].segments[0].prefix
        length = unit.num_tokens
        if kind == "B":
            pending.pop(uid).backward(dout[start : start + length])
            for group in completed_groups(unit):
                assert attn.runtime.release_group(group) == 1
            continue
        packed = SimpleNamespace(cu_seqlens_q_cpu=[0, length], **{UNIT_ATTR: unit, RECOMPUTE_ATTR: kind == "Fr"})
        inputs = [t[start : start + length] for t in chunked]
        if kind == "Fp":
            with torch.no_grad():
                attn(*inputs, packed_seq_params=packed)
        else:
            pending[uid] = attn(*inputs, packed_seq_params=packed)
            outputs[uid] = pending[uid].detach()
    assert not pending
    attn.runtime.assert_empty("unit schedule")
    torch.testing.assert_close(torch.cat([outputs[u.unit_id] for u in units]), expected_output, atol=1e-9, rtol=1e-9)
    for got, want in zip(chunked, reference):
        torch.testing.assert_close(got.grad, want.grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_kv_cache_fp32_accumulates_across_backward_without_changing_values(monkeypatch, dtype):
    from relax.backends.megatron.chunkflow.integration import _cache_leaf, _fold_leaf_grad_into

    monkeypatch.setenv("RELAX_CHUNKFLOW_KV_GRAD_FP32", "1")
    live = torch.zeros(2, 1, 4, dtype=dtype, requires_grad=True)
    leaf = _cache_leaf(live)
    assert leaf.dtype == dtype and torch.equal(leaf, live)
    for coefficient in (256, 1, 1):
        (leaf * coefficient).sum().backward()
        assert leaf.grad is None
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert leaf._chunkflow_grad_sum.dtype == expected_dtype
    torch.testing.assert_close(leaf._chunkflow_grad_sum, torch.full_like(leaf._chunkflow_grad_sum, 258))
    _fold_leaf_grad_into(live, leaf)
    (live * 0).sum().backward()
    torch.testing.assert_close(live.grad, torch.full_like(live, 258), rtol=0, atol=0)
    monkeypatch.setenv("RELAX_CHUNKFLOW_KV_GRAD_FP32", "0")
    old_live = torch.zeros_like(live, requires_grad=True)
    old_leaf = _cache_leaf(old_live)
    for coefficient in (256, 1, 1):
        (old_leaf * coefficient).sum().backward()
    _fold_leaf_grad_into(old_live, old_leaf)
    (old_live * 0).sum().backward()
    torch.testing.assert_close(
        old_live.grad, torch.full_like(old_live, 256 if dtype == torch.bfloat16 else 258), rtol=0, atol=0
    )
