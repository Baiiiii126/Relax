# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import pytest
import torch

from relax.backends.megatron.chunkflow.metadata import int32_metadata


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_chunk_metadata_preserves_values_and_owns_storage(monkeypatch, enabled):
    monkeypatch.setenv("RELAX_CHUNKFLOW_EXECUTION_OPT", enabled)
    values = [0, 8192, 16384, 16512]
    result = int32_metadata(values, torch.device("cpu"))
    values[-1] = 0
    torch.testing.assert_close(result, torch.tensor([0, 8192, 16384, 16512], dtype=torch.int32))
    assert result.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to verify asynchronous metadata copies")
def test_chunk_metadata_cuda_uses_current_stream_without_host_sync(monkeypatch):
    monkeypatch.setenv("RELAX_CHUNKFLOW_EXECUTION_OPT", "1")
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream()
    # Warm up allocator/runtime outside the measurement; keep outputs alive through completion.
    with torch.cuda.stream(stream):
        warmup = int32_metadata([0, 8192], device)
    stream.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with torch.cuda.stream(stream):
            outputs = [int32_metadata([0, 8192, 8192 + i], device) + i for i in range(16)]
    stream.synchronize()
    names = {event.name for event in profile.events()}
    assert "cudaStreamSynchronize" not in names
    for i, output in enumerate(outputs):
        torch.testing.assert_close(output.cpu(), torch.tensor([i, 8192 + i, 8192 + 2 * i], dtype=torch.int32))
    assert warmup.numel() == 2
