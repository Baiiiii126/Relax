# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA tests for the standalone computation prototype, not EP
communication."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton expert GEMMs")


@pytest.fixture(scope="module", autouse=True)
def cuda_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture(scope="module")
def tiles():
    pytest.importorskip("triton")
    # Avoid importing the Megatron package and its model-installation side effects.
    folder = Path(__file__).parents[3] / "relax/backends/megatron/chunkflow"
    package = ModuleType("_expert_tiles_test")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    name = package.__name__ + ".expert_tiles"
    spec = importlib.util.spec_from_file_location(name, folder / "expert_tiles.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    for key in tuple(sys.modules):
        if key.startswith(package.__name__):
            del sys.modules[key]


def _inputs(counts, inputs, outputs):
    torch.manual_seed(1235)
    x = torch.randn(sum(counts), inputs, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(len(counts), outputs, inputs, device="cuda", dtype=torch.bfloat16) * 0.1
    dy = torch.randn(sum(counts), outputs, device="cuda", dtype=torch.bfloat16) * 0.1
    return x, w, dy


def _reference(x, w, dy, counts):
    ys, dxs, dws = [], [], []
    for xe, we, dye in zip(x.split(counts), w.unbind(), dy.split(counts)):
        ys.append(xe.float() @ we.float().T)
        dxs.append(dye.float() @ we.float())
        dws.append(dye.float().T @ xe.float())
    return torch.cat(ys).to(x.dtype), torch.cat(dxs).to(x.dtype), torch.stack(dws)


@pytest.mark.parametrize(
    "counts",
    [
        [0, 1, 63, 64, 65, 129, 0],
        [0, 257, 0],
        [0, 0, 0],
        [(e * 17) % 131 if e % 5 == 0 else 0 for e in range(128)],
        [0, 255, 256, 257, 511, 512, 513, 0],
    ],
)
@pytest.mark.parametrize("block_m", [32, 64, 128, 256])
def test_expert_tiles_forward_and_backward_with_empty_experts_and_tails(tiles, counts, block_m):
    x, w, dy = _inputs(counts, 79, 113)
    offsets = tiles.make_offsets(torch.tensor(counts, dtype=torch.int64, device="cuda"))
    # FP32 reference avoids TF32 reducing reference accuracy on Hopper.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = _reference(x, w, dy, counts)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    actual = (
        tiles.forward(x, w, offsets, block_m=block_m),
        tiles.backward_input(dy, w, offsets, block_m=block_m),
        tiles.backward_weight(x, dy, offsets),
    )
    for got, want in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, want, atol=0.004, rtol=0.02)
    torch.testing.assert_close(actual[2], expected[2], atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("inputs,outputs", [(2048, 1024), (512, 2048)])
@pytest.mark.parametrize("block_m", [128, 256])
def test_expert_tiles_qwen_expert_shapes(tiles, inputs, outputs, block_m):
    counts = [0, 1, 255, 257]
    x, w, dy = _inputs(counts, inputs, outputs)
    offsets = tiles.make_offsets(torch.tensor(counts, device="cuda"))
    actual_y = tiles.forward(x, w, offsets, block_m=block_m)
    actual_dx = tiles.backward_input(dy, w, offsets, block_m=block_m)
    expected_y = torch.cat([xe @ we.T for xe, we in zip(x.split(counts), w.unbind())])
    expected_dx = torch.cat([dye @ we for dye, we in zip(dy.split(counts), w.unbind())])
    torch.testing.assert_close(actual_y, expected_y, atol=0.01, rtol=0.02)
    torch.testing.assert_close(actual_dx, expected_dx, atol=0.01, rtol=0.02)
    expected_dw = torch.stack([dye.float().T @ xe.float() for xe, dye in zip(x.split(counts), dy.split(counts))])
    torch.testing.assert_close(tiles.backward_weight(x, dy, offsets), expected_dw, atol=3e-5, rtol=3e-4)


def test_expert_tiles_accumulates_fp32_without_changing_empty_expert_gradients(tiles):
    counts = [0, 67, 0, 130, 0]
    x, w, dy = _inputs(counts, 73, 85)
    offsets = tiles.make_offsets(torch.tensor(counts, device="cuda"))
    grad = torch.randn_like(w, dtype=torch.float32)
    before = grad.clone()
    expected = tiles.backward_weight(x, dy, offsets)
    assert tiles.backward_weight(x, dy, offsets, out=grad, accumulate=True) is grad
    tiles.backward_weight(x, dy, offsets, out=grad, accumulate=True)
    torch.testing.assert_close(grad, before + 2 * expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(grad[[0, 2, 4]], before[[0, 2, 4]])
    tiles.backward_weight(x, dy, offsets, out=grad, accumulate=False)
    torch.testing.assert_close(grad, expected, atol=0, rtol=0)


@pytest.mark.parametrize("block_m", [64, 256])
def test_expert_tiles_counts_are_read_on_device_during_graph_replay(tiles, block_m):
    counts = torch.tensor([256, 0, 256], device="cuda", dtype=torch.int64)
    x, w, dy = _inputs([256, 0, 256], 64, 64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            offsets = tiles.make_offsets(counts)
            tiles.forward(x, w, offsets, block_m=block_m)
            tiles.backward_input(dy, w, offsets, block_m=block_m)
            tiles.backward_weight(x, dy, offsets)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        offsets = tiles.make_offsets(counts)
        result = tiles.forward(x, w, offsets, block_m=block_m)
        dx = tiles.backward_input(dy, w, offsets, block_m=block_m)
        dw = tiles.backward_weight(x, dy, offsets)
    for layout in ([0, 512, 0], [1, 255, 256], [256, 0, 256]):
        counts.copy_(torch.tensor(layout, device="cuda"))
        graph.replay()
        expected_y, expected_dx, expected_dw = _reference(x, w, dy, layout)
        torch.testing.assert_close(result, expected_y, atol=0.004, rtol=0.02)
        torch.testing.assert_close(dx, expected_dx, atol=0.004, rtol=0.02)
        torch.testing.assert_close(dw, expected_dw, atol=3e-5, rtol=3e-4)


def _activation(h, probs):
    gate, up = h.float().chunk(2, dim=-1)
    # The FP64 multiply is before the BF16 cast, as in the weighted SwiGLU path.
    return (torch.nn.functional.silu(gate) * up * probs[:, None]).to(h.dtype)


@pytest.mark.parametrize("use_mcore_activation", [False, True])
@pytest.mark.parametrize("block_m", [64, 256])
def test_expert_tiles_mlp_preserves_probability_gradients_and_reverse_backward(tiles, use_mcore_activation, block_m):
    activation = _activation
    if use_mcore_activation:
        fused = pytest.importorskip("megatron.core.fusions.fused_bias_swiglu")

        def activation(h, probs):
            return fused.weighted_bias_swiglu_impl(h, None, probs[:, None], False, None)

    x, w1, _ = _inputs([0, 67, 260], 64, 128)
    w2 = torch.randn(3, 64, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    w1.requires_grad_()
    w2.requires_grad_()
    main = (torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
    expected_main = [torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32)]
    contexts = []
    for counts in ([0, 67, 260], [64, 1, 262]):
        xi = x.clone().requires_grad_()
        probs = torch.rand(x.shape[0], device="cuda", dtype=torch.float64, requires_grad=True)
        offsets = tiles.make_offsets(torch.tensor(counts, device="cuda"))
        actual = tiles.expert_mlp(xi, w1, w2, offsets, probs, activation, main_grads=main, block_m=block_m)
        xr, pr = xi.detach().clone().requires_grad_(), probs.detach().clone().requires_grad_()
        wr1, wr2 = w1.detach().clone().requires_grad_(), w2.detach().clone().requires_grad_()
        fc1_inputs, fc1_outputs, fc2_inputs = [], [], []
        outputs = []
        for xe, pe, a, b in zip(xr.split(counts), pr.split(counts), wr1.unbind(), wr2.unbind()):
            h = xe @ a.T
            h.retain_grad()
            z = activation(h, pe)
            z.retain_grad()
            fc1_inputs.append(xe)
            fc1_outputs.append(h)
            fc2_inputs.append(z)
            outputs.append(z @ b.T)
        expected = torch.cat(outputs)
        dout = torch.randn_like(expected) * 0.1
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03)
        contexts.append((actual, expected, dout, xi, xr, probs, pr, fc1_inputs, fc1_outputs, fc2_inputs, counts))
    for actual, expected, dout, xi, xr, p, pr, xs, hs, zs, counts in reversed(contexts):
        actual.backward(dout)
        expected.backward(dout)
        assert p.grad.dtype == torch.float64
        torch.testing.assert_close(xi.grad, xr.grad, atol=0.01, rtol=0.04)
        torch.testing.assert_close(p.grad, pr.grad, atol=0.02, rtol=0.05)
        for e, (xe, h, z, dye) in enumerate(zip(xs, hs, zs, dout.split(counts))):
            expected_main[0][e].add_(h.grad.float().T @ xe.float())
            expected_main[1][e].add_(dye.float().T @ z.float())
    assert w1.grad is None and w2.grad is None
    for actual, expected in zip(main, expected_main):
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.05)


def test_expert_tiles_rejects_incompatible_layouts(tiles):
    x, w, dy = _inputs([2, 3], 64, 64)
    offsets = tiles.make_offsets(torch.tensor([2, 3], device="cuda"))
    with pytest.raises(ValueError, match="contiguous"):
        tiles.forward(x[:, ::2], w[:, :, ::2], offsets)
    with pytest.raises(ValueError, match="offsets"):
        tiles.forward(x, w, offsets.cpu())
    with pytest.raises(ValueError, match="initialized"):
        tiles.backward_weight(x, dy, offsets, accumulate=True)
    with pytest.raises(ValueError, match="weights"):
        tiles.forward(x, w[:0], offsets[:1])
