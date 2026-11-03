# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA GEMM tests; EP2 transport tests additionally require torchrun with two
ranks."""

import importlib.util
import os
import sys
from datetime import timedelta
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
@pytest.mark.parametrize("block_m", [0, 32, 64, 128, 256])
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
@pytest.mark.parametrize("block_m", [0, 128, 256])
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


@pytest.mark.parametrize("block_m", [0, 64, 256])
@pytest.mark.parametrize("indexed", [False, True])
def test_expert_tiles_counts_are_read_on_device_during_graph_replay(tiles, block_m, indexed):
    if indexed and not block_m:
        pytest.skip("Tile plans require a fixed row block")
    counts = torch.tensor([256, 0, 256], device="cuda", dtype=torch.int64)
    x, w, dy = _inputs([256, 0, 256], 64, 64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def execute():
        offsets, plan = (
            tiles._make_plan(counts, block_m, x.shape[0]) if indexed else (tiles.make_offsets(counts), None)
        )
        return (
            tiles._linear(x, w, offsets, block_m, False, tile_offsets=plan),
            tiles._linear(dy, w, offsets, block_m, True, tile_offsets=plan),
            tiles.backward_weight(x, dy, offsets),
        )

    with torch.cuda.stream(stream):
        for _ in range(3):
            execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result, dx, dw = execute()
    for layout in ([0, 512, 0], [1, 255, 256], [256, 0, 256]):
        counts.copy_(torch.tensor(layout, device="cuda"))
        graph.replay()
        expected_y, expected_dx, expected_dw = _reference(x, w, dy, layout)
        torch.testing.assert_close(result, expected_y, atol=0.004, rtol=0.02)
        torch.testing.assert_close(dx, expected_dx, atol=0.004, rtol=0.02)
        torch.testing.assert_close(dw, expected_dw, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("counts", [[0] * 128, [0] * 63 + [8192] + [0] * 64, [0, 1, 255, 256, 257, 0, 0]])
def test_expert_indexed_tiles_cover_skew_and_empty_experts_without_larger_metadata(tiles, counts):
    x, w, dy = _inputs(counts, 79, 113)
    offsets, plan = tiles._make_plan(torch.tensor(counts, device="cuda"), 256, x.shape[0])
    assert offsets.untyped_storage().nbytes() == 8 * (len(counts) + 1)
    assert offsets.untyped_storage().data_ptr() == plan.untyped_storage().data_ptr()
    expected_y, expected_dx, expected_dw = _reference(x, w, dy, counts)
    result = tiles._linear(x, w, offsets, 256, False, tile_offsets=plan)
    dx = tiles._linear(dy, w, offsets, 256, True, tile_offsets=plan)
    dw = tiles.backward_weight(x, dy, offsets)
    torch.testing.assert_close(result, expected_y, atol=0.004, rtol=0.02)
    torch.testing.assert_close(dx, expected_dx, atol=0.004, rtol=0.02)
    torch.testing.assert_close(dw, expected_dw, atol=3e-5, rtol=3e-4)


def _activation(h, probs):
    gate, up = h.float().chunk(2, dim=-1)
    # The FP64 multiply is before the BF16 cast, as in the weighted SwiGLU path.
    return (torch.nn.functional.silu(gate) * up * probs[:, None]).to(h.dtype)


@pytest.mark.parametrize("use_mcore_activation", [False, True])
@pytest.mark.parametrize("block_m", [0, 64, 256])
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


@pytest.fixture(scope="module")
def adapter(tiles):
    pytest.importorskip("transformer_engine")
    name = "_expert_tiles_test.expert_adapter"
    path = Path(__file__).parents[3] / "relax/backends/megatron/chunkflow/expert_adapter.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("block_m", [0, 128, 256])
def test_expert_adapter_preserves_parameters_and_reverse_backward_hooks(adapter, block_m):
    layer = torch.nn.Module()
    layer.num_gemms = 4
    layer.disable_parameter_transpose_cache = True
    layer.is_first_microbatch = True
    hooks = [0] * 4
    for e in range(4):
        weight = torch.nn.Parameter(torch.randn(85, 73, device="cuda", dtype=torch.bfloat16) * 0.1)
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = True
        layer.register_parameter(f"weight{e}", weight)

        def hook(parameter, index=e):
            assert parameter.grad_added_to_main_grad
            hooks[index] += 1
            parameter.grad = None

        weight.register_post_accumulate_grad_hook(hook)
    state = adapter._Linear(layer, block_m)
    weights = tuple(layer.parameters())
    addresses = [w.data_ptr() for w in weights]
    expected = [torch.zeros_like(w, dtype=torch.float32) for w in weights]
    pending = []
    for counts in ([0, 63, 256, 257], [65, 0, 258, 253]):
        x = (torch.randn(sum(counts), 73, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
        dy = torch.randn(sum(counts), 85, device="cuda", dtype=torch.bfloat16) * 0.1
        y, bias = state.forward(x, adapter._DeviceCounts(torch.tensor(counts, device="cuda"), block_m, x.shape[0]))
        reference = torch.cat([xe @ we.T for xe, we in zip(x.split(counts), weights)])
        torch.testing.assert_close(y, reference, atol=0.004, rtol=0.02)
        assert bias is None
        pending.append((x, y, dy, counts))
    for x, y, dy, counts in reversed(pending):
        y.backward(dy)
        reference_dx = torch.cat([de @ we for de, we in zip(dy.split(counts), weights)])
        torch.testing.assert_close(x.grad, reference_dx, atol=0.004, rtol=0.02)
        for e, (xe, de) in enumerate(zip(x.split(counts), dy.split(counts))):
            expected[e].add_(de.float().T @ xe.float())
    assert hooks == [2] * 4
    assert [w.data_ptr() for w in weights] == addresses
    for w, reference in zip(weights, expected):
        torch.testing.assert_close(w.main_grad, reference, atol=3e-5, rtol=3e-4)

    # DDP/optimizer storage rebinding between completed invocations must refresh both tables.
    for weight in weights:
        weight.data = weight.detach().clone()
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    x, _, dy, counts = pending[0]
    y, _ = state.forward(x.detach(), adapter._DeviceCounts(torch.tensor(counts, device="cuda"), block_m, x.shape[0]))
    y.backward(dy)
    for weight, xe, de in zip(weights, x.split(counts), dy.split(counts)):
        torch.testing.assert_close(weight.main_grad, de.float().T @ xe.float(), atol=3e-5, rtol=3e-4)
    assert hooks == [3] * 4


@pytest.fixture(scope="module")
def ep_group():
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("EP2 transport needs torchrun --nproc_per_node=2")
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    group = dist.group.WORLD
    yield group
    if owned:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("layout", ["skewed", "local_only", "remote_only", "one_way", "empty"])
def test_expert_transport_matches_serial_dispatch_and_reverse_backward(tiles, adapter, ep_group, layout, side_stream):
    import torch.distributed as dist
    from _expert_tiles_test.expert_transport import Dispatch, dispatch

    rank = dist.get_rank(ep_group)
    torch.manual_seed(345 + rank)
    # [source, destination, local expert]; include repeated offsets and tile tails.
    counts = [
        [[0, 65, 257, 0], [3, 0, 128, 1]],
        [[1, 255, 0, 129], [0, 2, 63, 257]],
    ]
    if layout == "local_only":
        counts[0][1] = counts[1][0] = [0] * 4
    elif layout == "remote_only":
        counts[0][0] = counts[1][1] = [0] * 4
    elif layout == "one_way":
        counts[0] = [[0] * 4, [0] * 4]
    elif layout == "empty":
        counts = [[[0] * 4 for _ in range(2)] for _ in range(2)]
    layers, states, references, reference_grads = [], [], [], []
    for index, (inputs, outputs) in enumerate(((32, 96), (48, 32))):
        module = torch.nn.Module()
        module.num_gemms = 4
        module.disable_parameter_transpose_cache = True
        module.is_first_microbatch = True
        for expert in range(4):
            w = torch.nn.Parameter(torch.randn(outputs, inputs, device="cuda", dtype=torch.bfloat16) * 0.02)
            w.main_grad = torch.zeros_like(w, dtype=torch.float32)
            w.grad_added_to_main_grad = False
            w.zero_out_wgrad = True
            module.register_parameter(f"weight{expert}", w)
        layers.append(module)
        states.append(adapter._Linear(module, 0, source_first=index == 0))
        references.append(torch.stack([w.detach() for w in module.parameters()]).requires_grad_())
        reference_grads.append(torch.zeros_like(references[-1], dtype=torch.float32))

    def exchange(x, inputs, outputs):
        output = x.new_empty((sum(outputs), *x.shape[1:]))
        dist.all_to_all_single(output, x.contiguous(), outputs, inputs, group=ep_group)
        return output

    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream() if side_stream else current
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        pending = []
        for invocation in range(2):
            table = counts if invocation == 0 else [[list(reversed(c)) for c in source] for source in counts]
            send = tuple(sum(c) for c in table[rank])
            receive = tuple(sum(table[source][rank]) for source in range(2))
            local_counts = [table[source][rank] for source in range(2)]
            combined = [sum(c[e] for c in local_counts) for e in range(4)]
            x = (torch.randn(sum(send), 32, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
            p = torch.rand(sum(send), device="cuda", dtype=torch.float64, requires_grad=True)
            received_x = exchange(x.detach(), list(send), list(receive))
            received_p = exchange(p.detach(), list(send), list(receive))
            order = [source * 4 + e for e in range(4) for source in range(2)]
            flat_counts = [n for c in local_counts for n in c]
            expected_x = torch.cat([received_x.split(flat_counts)[i] for i in order]).requires_grad_()
            expected_p = torch.cat([received_p.split(flat_counts)[i] for i in order]).requires_grad_()
            offsets = tiles.make_offsets(torch.tensor(combined, device="cuda"))
            reference = tiles.expert_mlp(
                expected_x,
                *references,
                offsets,
                expected_p,
                _activation,
                main_grads=tuple(reference_grads),
                block_m=128,
            )
            if rank == 1 and invocation == 0:
                torch.cuda._sleep(100000)
            state = Dispatch(ep_group, send, receive, torch.tensor(local_counts, device="cuda"))
            packed_x, packed_p = dispatch(x, p, state)
            device_counts = torch.tensor(combined, device="cuda")
            device_counts._relax_dispatch = state
            metadata = adapter._DeviceCounts(device_counts)
            h, _ = states[0].forward(packed_x, metadata)
            y, _ = states[1].forward(_activation(h, packed_p), metadata)
            torch.testing.assert_close(packed_x, expected_x, atol=0, rtol=0)
            torch.testing.assert_close(packed_p, expected_p, atol=0, rtol=0)
            torch.testing.assert_close(y, reference, atol=0.0001, rtol=0.02)
            pending.append((x, p, y, expected_x, expected_p, reference, local_counts, send, receive))
        for x, p, y, expected_x, expected_p, reference, local_counts, send, receive in reversed(pending):
            dy = torch.randn_like(y) * 0.1
            y.backward(dy)
            reference.backward(dy)
            canonical_sizes = [local_counts[s][e] for e in range(4) for s in range(2)]
            order = [e * 2 + s for s in range(2) for e in range(4)]
            for actual, expected in ((x.grad, expected_x.grad), (p.grad, expected_p.grad)):
                source_major = torch.cat([expected.split(canonical_sizes)[i] for i in order])
                want = exchange(source_major, list(receive), list(send))
                assert actual.dtype == want.dtype
                torch.testing.assert_close(
                    actual, want, atol=1e-9 if actual.dtype == torch.float64 else 0.0001, rtol=0.03
                )
        for module, expected in zip(layers, reference_grads):
            for w, want in zip(module.parameters(), expected):
                torch.testing.assert_close(w.main_grad, want, atol=3e-5, rtol=3e-4)
    current.wait_stream(stream)


@pytest.mark.parametrize("layout", ["local", "remote", "one_way"])
def test_expert_compact_metadata_matches_real_dispatcher_and_reverse_backward(adapter, ep_group, layout, monkeypatch):
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    monkeypatch.setenv("RELAX_MOE_METADATA_MODE", "off")
    parallel_state.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(["ep", "expt_tp", "tp_ep"])
        rank = dist.get_rank(ep_group)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=2,
            ffn_hidden_size=64,
            num_moe_experts=8,
            moe_router_topk=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            moe_permute_fusion=True,
            moe_router_dtype="fp64",
        )
        dispatchers = [
            MoEAlltoAllTokenDispatcher(4, list(range(rank * 4, rank * 4 + 4)), config, pg) for _ in range(2)
        ]
        assert adapter._install_compact_metadata(dispatchers[1])

        def forward(owner, x, p, routing):
            h, probabilities = owner.dispatch_preprocess(x, routing, p)
            h, probabilities = owner.token_dispatch(h, probabilities)
            h, counts, probabilities = owner.dispatch_postprocess(h, probabilities)
            h = (h * probabilities[:, None]).to(x.dtype)
            result = owner.combine_postprocess(owner.token_combine(owner.combine_preprocess(h)))
            return result, counts

        pending = []
        torch.manual_seed(712 + rank)
        for invocation in range(2):
            rows = 5 + 2 * invocation + rank
            x = (torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
            p = torch.rand(rows, 8, device="cuda", dtype=torch.float64, requires_grad=True)
            rx, rp = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()
            destination = rank if layout == "local" else 1 - rank if layout == "remote" else 0
            routing = torch.zeros(rows, 8, dtype=torch.bool, device="cuda")
            routing[:, destination * 4 + invocation : destination * 4 + invocation + 2] = True
            expected, _ = forward(dispatchers[0], rx, rp, routing)
            result, counts = forward(dispatchers[1], x, p, routing)
            assert counts.is_cuda
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            pending.append((x, p, result, rx, rp, expected))
        for x, p, result, rx, rp, expected in reversed(pending):
            dy = torch.randn_like(result)
            expected.backward(dy)
            result.backward(dy)
            torch.testing.assert_close(x.grad, rx.grad, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, rp.grad, atol=1e-10, rtol=1e-8)
    finally:
        parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("block_m", [0, 256])
def test_expert_adapter_real_ddp_hooks_and_master_optimizer(adapter, ep_group, block_m):
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
    )
    try:
        model_parallel_cuda_manual_seed(1235)
        pg = ProcessGroupCollection.use_mpu_process_groups(
            ["tp", "pp", "dp", "dp_cp", "ep", "expt_tp", "expt_dp", "tp_ep"]
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=64,
            moe_ffn_hidden_size=64,
            num_moe_experts=8,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            bias_activation_fusion=True,
            gradient_accumulation_fusion=True,
            calculate_per_token_loss=True,
            moe_router_dtype="fp64",
            moe_token_dispatcher_type="alltoall",
            disable_parameter_transpose_cache=True,
        )
        submodules = GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        )
        experts = TEGroupedMLP(4, config, submodules, pg_collection=pg).cuda()
        reference = TEGroupedMLP(4, config, submodules, pg_collection=pg).cuda()
        weights = tuple(experts.parameters())
        for linear in (experts.linear_fc1, experts.linear_fc2):
            linear.forward = adapter._Linear(linear, block_m).forward
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                use_distributed_optimizer=False,
                check_for_nan_in_grad=False,
            ),
            experts,
            pg_collection=pg,
        )
        assert all(a is b for a, b in zip(weights, experts.parameters()))
        optimizer = Float16OptimizerWithFloat16Params(
            torch.optim.SGD(weights, lr=0.01),
            OptimizerConfig(bf16=True, clip_grad=0.0, log_num_zeros_in_grad=False),
            grad_scaler=None,
            init_state_fn=None,
        )
        optimizer.grad_stats_parallel_group = pg.expt_dp
        ready = {w: 0 for w in weights}
        for bucket in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
            original = bucket.register_grad_ready

            def record(parameter, *args, target=bucket, function=original, **kwargs):
                if target.is_last_microbatch:
                    ready[parameter] += 1
                return function(parameter, *args, **kwargs)

            bucket.register_grad_ready = record
        for step in range(2):
            optimizer.zero_grad()
            ddp.zero_grad_buffer()
            for actual, expected in zip(weights, reference.parameters()):
                expected.data.copy_(actual.data)
                expected.main_grad = torch.zeros_like(expected, dtype=torch.float32)
            pending = []
            for counts in ([0, 67, 260, 0], [64, 1, 262, 0]):
                x = (torch.randn(sum(counts), 64, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
                p = torch.rand(sum(counts), device="cuda", dtype=torch.float64, requires_grad=True)
                rx, rp = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()
                y, _ = ddp(x, adapter._DeviceCounts(torch.tensor(counts, device="cuda"), block_m, x.shape[0]), p)
                expected, _ = reference(rx, torch.tensor(counts), rp)
                torch.testing.assert_close(y, expected, atol=0.001, rtol=0.02)
                pending.append((x, p, y, rx, rp, expected))
            for index, (x, p, y, rx, rp, expected) in enumerate(reversed(pending)):
                dy = torch.randn_like(y) * 0.1
                if index == 0:
                    with ddp.no_sync():
                        y.backward(dy)
                else:
                    y.backward(dy)
                expected.backward(dy)
                torch.testing.assert_close(x.grad, rx.grad, atol=0.001, rtol=0.02)
                torch.testing.assert_close(p.grad, rp.grad, atol=1e-8, rtol=0.03)
            ddp.finish_grad_sync()
            for actual, expected in zip(weights, reference.parameters()):
                assert actual.grad is None and actual.grad_added_to_main_grad
                assert ready[actual] == step + 1
                torch.testing.assert_close(actual.main_grad, expected.main_grad, atol=3e-5, rtol=3e-4)
            for name, weight in experts.named_parameters():
                if name.endswith("weight3"):
                    assert torch.count_nonzero(weight.main_grad) == 0
            masters = [p for group in optimizer.fp32_from_float16_groups for p in group]
            expected_masters = [m.detach().clone() - 0.01 * w.main_grad for m, w in zip(masters, weights)]
            assert optimizer.step()[0]
            for master, weight, expected in zip(masters, weights, expected_masters):
                torch.testing.assert_close(master, expected, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(weight, master.to(torch.bfloat16), atol=0, rtol=0)
    finally:
        parallel_state.destroy_model_parallel()
