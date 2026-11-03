# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Host cache and metadata lifetimes; no CUDA or collectives are executed."""

import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


@pytest.fixture
def adapter(monkeypatch):
    folder = Path(__file__).parents[3] / "relax/backends/megatron/chunkflow"
    package = ModuleType("_expert_cpu_test")
    package.__path__ = [str(folder)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + ".expert_adapter"
    spec = importlib.util.spec_from_file_location(name, folder / "expert_adapter.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield module
    sys.modules.pop(package.__name__ + ".expert_tiles", None)


@pytest.fixture
def state(adapter, monkeypatch):
    def check(tensor, ndim, name):
        if tensor.dtype != torch.bfloat16 or tensor.ndim != ndim or not tensor.is_contiguous():
            raise ValueError(name)

    monkeypatch.setattr(adapter.expert_tiles, "_check_tensor", check)
    module = torch.nn.Module()
    module.num_gemms = 2
    for i in range(2):
        weight = torch.nn.Parameter(torch.ones(3, 2, dtype=torch.bfloat16))
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        module.register_parameter(f"weight{i}", weight)
    return adapter._Linear(module, 256)


def test_expert_cache_reuses_validation_and_refreshes_rebound_storage(adapter, state, monkeypatch):
    weights = state.parameters()
    table = state.weights.get(weights, state.weight_key)
    original = adapter.expert_tiles._check_tensor
    calls = []

    def record(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(adapter.expert_tiles, "_check_tensor", record)
    assert state.parameters() is weights
    assert state.weights.get(weights, state.weight_key) is table
    assert not calls
    weights[0].data = weights[0].detach().clone()
    rebound = state.parameters()
    assert all(a is b for a, b in zip(rebound, weights))
    assert len(calls) == 2
    assert state.weights.get(rebound, state.weight_key) is not table
    replacement = torch.nn.Parameter(weights[0].detach().clone())
    state.module.weight0 = replacement
    assert state.parameters()[0] is replacement


@pytest.mark.parametrize("change", ["shape", "stride", "dtype", "frozen", "fsdp"])
def test_expert_cache_revalidates_metadata_changes_without_new_parameter_identity(state, change):
    weight = state.parameters()[0]
    if change == "shape":
        weight.data = weight.detach().view(2, 3)
    elif change == "stride":
        weight.data = weight.detach().as_strided((3, 2), (1, 3))
    elif change == "dtype":
        weight.data = weight.detach().float()
    elif change == "frozen":
        weight.requires_grad_(False)
    else:
        weight.__fsdp_param__ = True
    with pytest.raises(ValueError):
        state.parameters()


def test_expert_gradient_cache_releases_old_storage_and_keeps_dynamic_flags(adapter, state, monkeypatch):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append(args[10])

    monkeypatch.setitem(sys.modules, "_expert_cpu_test._expert_tile_kernels", SimpleNamespace(weight_grad=Kernel()))
    monkeypatch.setitem(
        sys.modules, "transformer_engine.pytorch.module.base", SimpleNamespace(get_dummy_wgrad=lambda *a, **k: None)
    )
    weights = state.parameters()
    x, dy = torch.ones(2, 2, dtype=torch.bfloat16), torch.ones(2, 3, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 1, 2])
    state.wgrad(x, dy, offsets, weights, None)
    old = weakref.ref(weights[0].main_grad)
    for weight in weights:
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight.overwrite_main_grad = True
    gc.collect()
    assert old() is None
    state.wgrad(x, dy, offsets, weights, None)
    for weight in weights:
        weight.overwrite_main_grad = False
    state.wgrad(x, dy, offsets, weights, None)
    state.wgrad(x, dy, offsets, weights, True)
    assert calls == [True, False, True, False]
    weights[0].main_grad = torch.zeros_like(weights[0], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        state.wgrad(x, dy, offsets, weights, None)
    weights[0].main_grad = torch.zeros(2, 3).T
    with pytest.raises(ValueError, match="contiguous"):
        state.wgrad(x, dy, offsets, weights, None)


def _dispatcher():
    return SimpleNamespace(
        config=SimpleNamespace(
            moe_expert_capacity_factor=None,
            moe_router_padding_for_quantization=False,
            moe_permute_fusion=True,
            cuda_graph_impl="none",
        ),
        tp_size=1,
        ep_size=2,
        drop_and_pad=False,
        relax_metadata_mode="off",
        cuda_dtoh_point="before_permutation_1",
        cuda_sync_point="before_ep_alltoall",
    )


def test_expert_compact_metadata_preserves_counts_and_pending_backward_splits(adapter, monkeypatch):
    class Device:
        def __init__(self, values):
            self.values = np.array(values, dtype=np.int64)
            self.shape, self.dtype, self.is_cuda = self.values.shape, torch.int64, True

    class PendingHost:
        def __init__(self, values):
            self.values, self.ready = values.copy(), False

        def __getitem__(self, item):
            assert self.ready, "host splits read before the original event wait"
            return self.values[item]

    monkeypatch.setattr(torch, "cat", lambda tensors: Device(np.concatenate([t.values for t in tensors])))
    owner = _dispatcher()
    calls = []

    def original(point, tokens_per_expert):
        assert tokens_per_expert is None
        assert owner.output_splits is None and owner.output_splits_tp is None
        calls.append(point)
        if point == "before_permutation_1":
            owner.input_splits = PendingHost(owner.input_splits.values)
        else:
            owner.input_splits.ready = True

    owner._maybe_dtoh_and_synchronize = original
    assert adapter._install_compact_metadata(owner)
    saved = []
    for inputs, outputs in (([2, 0], [2, 7]), ([0, 3], [4, 3]), ([0, 0], [0, 0])):
        owner.input_splits, owner.output_splits = Device(inputs), Device(outputs)
        owner.output_splits_tp = Device([sum(outputs)])
        counts, sort_metadata = object(), object()
        owner.num_global_tokens_per_local_expert = sort_metadata
        for point in ("before_permutation_1", "before_ep_alltoall", "before_permutation_2", "before_finish"):
            assert owner._maybe_dtoh_and_synchronize(point, counts) is counts
        assert owner.num_global_tokens_per_local_expert is sort_metadata
        saved.append((owner.input_splits, owner.output_splits, inputs, outputs))
    for inputs, outputs, expected_inputs, expected_outputs in reversed(saved):
        np.testing.assert_array_equal(inputs, expected_inputs)
        np.testing.assert_array_equal(outputs, expected_outputs)
    assert calls == ["before_permutation_1", "before_ep_alltoall"] * 3


@pytest.mark.parametrize(
    "attribute,value", [("tp_size", 2), ("ep_size", 1), ("drop_and_pad", True), ("relax_metadata_mode", "b")]
)
def test_expert_compact_metadata_keeps_original_unsupported_dispatcher(adapter, attribute, value):
    owner = _dispatcher()
    owner._maybe_dtoh_and_synchronize = original = lambda *args: None
    setattr(owner, attribute, value)
    assert not adapter._install_compact_metadata(owner)
    assert owner._maybe_dtoh_and_synchronize is original


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("moe_expert_capacity_factor", 1.0),
        ("moe_router_padding_for_quantization", True),
        ("moe_permute_fusion", False),
        ("cuda_graph_impl", "local"),
    ],
)
def test_expert_compact_metadata_keeps_original_dynamic_or_graph_path(adapter, attribute, value):
    owner = _dispatcher()
    owner._maybe_dtoh_and_synchronize = original = lambda *args: None
    setattr(owner.config, attribute, value)
    assert not adapter._install_compact_metadata(owner)
    assert owner._maybe_dtoh_and_synchronize is original
