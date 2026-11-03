# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real Triton host JIT cache with offline SM90 compilation.

Only device discovery is stubbed; warmup compiles but never launches a kernel.
Numerical and distributed execution are covered by test_expert_tiles.py.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def kernels(monkeypatch, tmp_path):
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        driver,
        "_active",
        SimpleNamespace(
            get_current_device=lambda: 0,
            get_current_stream=lambda device: 0,
            get_current_target=lambda: GPUTarget("cuda", 90, 32),
        ),
    )
    path = Path(__file__).parents[2] / "relax/backends/megatron/chunkflow/_expert_tile_kernels.py"
    spec = importlib.util.spec_from_file_location("_expert_kernel_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_rows_reuse_kernel(kernel, **kwargs):
    first = None
    # Include zero/one, alignment classes, tile boundaries and long routed batches.
    for rows in (0, 1, 15, 16, 17, 255, 256, 257, 513, 8191, 8192, 8193, 32768, 32769, 65536):
        compiled = kernel.warmup(ROWS=rows, grid=(1,), **kwargs)
        assert compiled is not None
        if first is None:
            first = compiled
        assert compiled is first, f"{kernel.__name__} compiled another variant for ROWS={rows}"


@pytest.mark.parametrize("pointers", [False, True])
@pytest.mark.parametrize("transpose,source", [(False, -1), (False, 0), (False, 1), (True, -1)])
@pytest.mark.parametrize("indexed", [False, True])
def test_expert_linear_reuses_kernel_across_routed_lengths(kernels, pointers, transpose, source, indexed):
    import triton.language as tl
    from triton.runtime.jit import MockTensor

    if indexed and source >= 0:
        pytest.skip("Source-split FC1 keeps the original scan")
    _assert_rows_reuse_kernel(
        kernels.linear,
        X=MockTensor(tl.bfloat16),
        W=MockTensor(tl.int64 if pointers else tl.bfloat16),
        Y=MockTensor(tl.bfloat16),
        OFFSETS=MockTensor(tl.int32 if indexed else tl.int64),
        EXPERTS=4,
        N=85,
        K=73,
        WE=0 if pointers else 85 * 73,
        WN=1 if transpose else 73,
        WK=85 if transpose else 1,
        BM=256 if indexed else 0,
        BN=64,
        BK=32,
        WORKERS=8,
        POINTERS=pointers,
        SOURCE_COUNTS=MockTensor(tl.int64) if source >= 0 else None,
        SOURCE=source,
        TILE_OFFSETS=MockTensor(tl.int32) if indexed else None,
        num_warps=4,
        num_stages=3,
    )


@pytest.mark.parametrize("counts_dtype", ["int32", "int64"])
def test_expert_tile_plan_compiles_without_device_access(kernels, counts_dtype):
    import triton.language as tl
    from triton.runtime.jit import MockTensor

    assert (
        kernels.tile_metadata.warmup(
            COUNTS=MockTensor(getattr(tl, counts_dtype)),
            METADATA=MockTensor(tl.int32),
            EXPERTS=128,
            BM=256,
            BLOCK=128,
            grid=(1,),
        )
        is not None
    )


@pytest.mark.parametrize("pointers", [False, True])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("offset_dtype", ["int32", "int64"])
def test_expert_weight_grad_reuses_kernel_across_routed_lengths(kernels, pointers, accumulate, offset_dtype):
    import triton.language as tl
    from triton.runtime.jit import MockTensor

    _assert_rows_reuse_kernel(
        kernels.weight_grad,
        X=MockTensor(tl.bfloat16),
        DY=MockTensor(tl.bfloat16),
        DW=MockTensor(tl.int64 if pointers else tl.float32),
        OFFSETS=MockTensor(getattr(tl, offset_dtype)),
        N=85,
        K=73,
        BN=64,
        BK=64,
        BR=32,
        ACCUMULATE=accumulate,
        POINTERS=pointers,
        num_warps=4,
        num_stages=3,
    )


@pytest.mark.parametrize("inputs,outputs", [(2048, 1024), (512, 2048)])
@pytest.mark.parametrize("transpose", [False, True])
def test_expert_indexed_qwen_shapes_compile_without_rows_specialization(kernels, inputs, outputs, transpose):
    import triton.language as tl
    from triton.runtime.jit import MockTensor

    _assert_rows_reuse_kernel(
        kernels.linear,
        X=MockTensor(tl.bfloat16),
        W=MockTensor(tl.int64),
        Y=MockTensor(tl.bfloat16),
        OFFSETS=MockTensor(tl.int32),
        TILE_OFFSETS=MockTensor(tl.int32),
        EXPERTS=128,
        N=inputs if transpose else outputs,
        K=outputs if transpose else inputs,
        WE=0,
        WN=1 if transpose else inputs,
        WK=outputs if transpose else 1,
        BM=256,
        BN=64,
        BK=32,
        WORKERS=78,
        POINTERS=True,
        num_warps=4,
        num_stages=3,
    )


@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_expert_source_copy_reuses_kernel_across_routed_lengths(kernels, source, reverse):
    import triton.language as tl
    from triton.runtime.jit import MockTensor

    _assert_rows_reuse_kernel(
        kernels.source_copy,
        X=MockTensor(tl.bfloat16),
        P=MockTensor(tl.float64),
        Y=MockTensor(tl.bfloat16),
        Q=MockTensor(tl.float64),
        SOURCE_OFFSETS=MockTensor(tl.int64),
        OFFSETS=MockTensor(tl.int64),
        SOURCE_COUNTS=MockTensor(tl.int64),
        H=73,
        EXPERTS=4,
        SOURCE=source,
        REVERSE=reverse,
    )
