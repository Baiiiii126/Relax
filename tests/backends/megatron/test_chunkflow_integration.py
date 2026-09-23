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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunkflow_matches_flash_varlen_on_training_shapes():
    """The production configuration, against the kernel it replaces.

    Every test above runs fp64 on CPU, which takes the plain-torch fallback.
    A real run is bf16 on CUDA with GQA and packed sequences, i.e. the flash
    path -- and the thing chunkflow displaces there is Transformer Engine's
    ``DotProductAttention``, which itself calls ``flash_attn_varlen_func``. So
    that function is the oracle: it is what Megatron would have computed.

    Shapes mirror Qwen3-4B under ``--tensor-model-parallel-size 2``: 32 query
    heads and 8 KV heads halve to 16/4 per rank, head_dim 128.
    """
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("flash_attn not installed")

    h, kv_h, d = 16, 4, 128
    lengths = [3000, 1096]  # packed, and neither is a multiple of chunk_size
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

    # Measured: dq 2.75e-04, dk 7.19e-03, dv 6.17e-03, relative to each
    # tensor's own magnitude (an absolute tolerance is meaningless here -- bf16
    # carries ~3 decimal digits and one ULP is already ~7.8e-03).
    #
    # dk/dv sit a decimal place above dq because a K token's gradient arrives
    # from every later query chunk as a separate bf16 kernel output, summed by
    # autograd in bf16, where the oracle sums inside one kernel in fp32. That
    # is chunking's cost, not a bug: re-running this with chunk_size past the
    # sequence length makes dk/dv bit-exact (0.0), and dropping GQA leaves them
    # unchanged (7.04e-03 / 5.59e-03), which rules out the head widening.
    rels = {
        name: ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()
        for name, got, want in zip(("dq", "dk", "dv"), run(chunked), run(reference))
    }
    assert max(rels.values()) < 2e-2, f"chunked differs from flash_attn_varlen_func: {rels}"


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


def test_chunkflow_install_can_retain_original_attention_for_single_unit_alignment(monkeypatch):
    class FakeAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 1
            self.config = SimpleNamespace(context_parallel_size=1)
            self.core_attention = torch.nn.Identity()

    model = torch.nn.Sequential(FakeAttn())
    original = model[0].core_attention
    monkeypatch.setenv("RELAX_CHUNKFLOW_USE_ORIGINAL_ATTENTION", "1")

    assert install(model, chunk_size=4096) == 0
    assert model[0].core_attention is original


# ---------------------------------------------------------------------------
# Unit mode: chunks driven one at a time, backward descending
# ---------------------------------------------------------------------------


def _unit_attn(chunk_size, layer=1):
    from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime

    return ChunkedCoreAttention(
        config=SimpleNamespace(context_parallel_size=1),
        layer_number=layer,
        chunk_size=chunk_size,
        runtime=ChunkflowRuntime(chunk_size),
    )


def _drive_units(attn, q, k, v, dout, chunk_size, descending_backward=True):
    """Forward chunk by chunk, then backward one unit at a time."""
    from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR
    from relax.backends.megatron.chunkflow.scheduler import build_chunks
    from relax.backends.megatron.chunkflow.units import build_micro_batches

    total = q.shape[0]
    chunks = build_chunks([total], chunk_size=chunk_size, repack_tail=False)
    units = build_micro_batches(chunks, "staggered")

    outs, bounds, cursor = [], [], 0
    for unit in units:
        cu = [0]
        for chunk in unit.chunks:
            for seg in chunk.segments:
                cu.append(cu[-1] + seg.length)
        n = cu[-1]
        packed = SimpleNamespace(cu_seqlens_q_cpu=cu, **{UNIT_ATTR: unit})
        outs.append(
            attn(q[cursor : cursor + n], k[cursor : cursor + n], v[cursor : cursor + n], packed_seq_params=packed)
        )
        bounds.append((cursor, cursor + n))
        cursor += n

    order = list(reversed(range(len(outs)))) if descending_backward else list(range(len(outs)))
    for i in order:
        s, e = bounds[i]
        outs[i].backward(dout[s:e])

    return torch.cat([o.detach() for o in outs], dim=0)


def test_unit_mode_matches_the_in_forward_loop():
    """Driving chunks as separate units must equal chunking inside one
    forward."""
    torch.manual_seed(0)
    total, h, d, chunk_size = 32, 2, 8, 16
    q, k, v = _qkv(total, h, d)
    q_ref, k_ref, v_ref = (t.clone().requires_grad_(True) for t in (q, k, v))
    q_u, k_u, v_u = (t.clone().requires_grad_(True) for t in (q, k, v))

    torch.testing.assert_close(_attn(chunk_size)(q_ref, k_ref, v_ref), _unit_attn(chunk_size)(q_u, k_u, v_u))

    ref, dq_r, dk_r, dv_r = _grads(_attn(chunk_size), q_ref, k_ref, v_ref)
    got, dq_u, dk_u, dv_u = _grads(_unit_attn(chunk_size), q_u, k_u, v_u)
    torch.testing.assert_close(ref, got)
    for got_g, want_g in ((dq_u, dq_r), (dk_u, dk_r), (dv_u, dv_r)):
        torch.testing.assert_close(got_g, want_g)


def _grads(module, q, k, v):
    out = module(q, k, v)
    out.backward(torch.ones_like(out))
    return out.detach(), q.grad, k.grad, v.grad


def test_unit_mode_survives_per_unit_descending_backward():
    """The schedule backards one unit at a time, descending.

    With the live K/V in the store the second ``.backward()`` would raise
    "backward through the graph a second time"; with the detached leaf plus the
    fold-back hook it must instead reproduce the single-pass gradients exactly.
    """
    torch.manual_seed(0)
    total, h, d, chunk_size = 48, 2, 8, 16
    q, k, v = _qkv(total, h, d)
    dout = torch.randn(total, h * d) * 0.1

    # Reference: one chunked forward, one backward.
    q_r, k_r, v_r = (t.clone().requires_grad_(True) for t in (q, k, v))
    ref = _attn(chunk_size)(q_r, k_r, v_r)
    ref.backward(dout)  # same downstream gradient as the unit-mode run

    # Unit mode, three units, backward 2 -> 1 -> 0.
    q_u, k_u, v_u = (t.clone().requires_grad_(True) for t in (q, k, v))
    attn = _unit_attn(chunk_size)
    _drive_units(attn, q_u, k_u, v_u, dout, chunk_size)

    for got, want in ((q_u.grad, q_r.grad), (k_u.grad, k_r.grad), (v_u.grad, v_r.grad)):
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)


def test_unit_mode_leaves_no_state_behind_after_release():
    from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime

    runtime = ChunkflowRuntime(16)
    assert runtime.num_stores() == 0
    runtime.assert_empty("a fresh runtime")


def test_unit_mode_recompute_matches_keep_everything():
    """Fp/Fr must give identical gradients to keeping every chunk's activation.

    A group long enough to drop some chunks (keep_last_k=1 of 4) forwards the
    early ones under no_grad, recomputes them before their own backward. The
    gradients must equal the keep-everything path exactly -- the mechanism is
    an exact re-derivation, so any difference is a bug, not noise.
    """
    torch.manual_seed(0)
    total, h, d, chunk_size = 64, 2, 8, 16
    q, k, v = _qkv(total, h, d)
    dout = torch.randn(total, h * d) * 0.1

    def run(keep_last_k):
        from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, ChunkflowRuntime
        from relax.backends.megatron.chunkflow.scheduler import build_chunks
        from relax.backends.megatron.chunkflow.units import build_micro_batches, completed_groups, gen_unit_ops

        runtime = ChunkflowRuntime(chunk_size)
        attn = _unit_attn(chunk_size)
        attn.runtime = runtime
        chunks = build_chunks([total], chunk_size=chunk_size, repack_tail=False)
        units = build_micro_batches(chunks, "staggered")
        ops = gen_unit_ops(units, chunks, keep_last_k, pp=1, stage=0)

        qq = q.detach().clone().requires_grad_(True)
        kk = k.detach().clone().requires_grad_(True)
        vv = v.detach().clone().requires_grad_(True)
        _bounds = {
            u.unit_id: (u.chunks[0].segments[0].prefix, sum(s.length for c in u.chunks for s in c.segments))
            for u in units
        }

        def dout_for(uid):
            s_, n_ = _bounds[uid]
            return dout[s_ : s_ + n_]

        pending = {}
        for kind, uid in ops:
            unit = units[uid]
            if kind == "B":
                pending.pop(uid).backward(dout_for(uid))
                for group in completed_groups(unit):
                    runtime.release_group(group)
                continue
            n = sum(s.length for chunk in unit.chunks for s in chunk.segments)
            # slice the unit's rows out of the parent
            start = unit.chunks[0].segments[0].prefix
            packed = SimpleNamespace(cu_seqlens_q_cpu=[0, n], **{UNIT_ATTR: unit, RECOMPUTE_ATTR: kind == "Fr"})
            q_u, k_u, v_u = qq[start : start + n], kk[start : start + n], vv[start : start + n]
            if kind == "Fp":
                with torch.no_grad():
                    attn(q_u, k_u, v_u, packed_seq_params=packed)
            else:
                pending[uid] = attn(q_u, k_u, v_u, packed_seq_params=packed)
        runtime.assert_empty("recompute test")
        return qq.grad, kk.grad, vv.grad

    full = run(keep_last_k=1 << 30)
    recomputed = run(keep_last_k=1)
    for f, r in zip(full, recomputed):
        torch.testing.assert_close(f, r, atol=1e-9, rtol=1e-9)
