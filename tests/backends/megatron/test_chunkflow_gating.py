# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pre-step dispatch agreement, with CPU-only mocked collectives."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist
from megatron.core import mpu

from relax.backends.megatron.chunkflow.gating import prepare_step_gate
from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime
from relax.utils import distributed_utils


@pytest.mark.parametrize("threshold,expected", [(0, True), (140, True), (1000, False)])
def test_gate_predicts_before_consumption_and_records_decision(monkeypatch, threshold, expected):
    runtime = ChunkflowRuntime(4096)
    model = SimpleNamespace(chunkflow_runtime=runtime)
    iterator = SimpleNamespace(
        rollout_data={"total_lengths": [28160]},
        micro_batch_indices=[[0]],
        offset=0,
        dummy_micro_batch_offsets=set(),
    )
    args = SimpleNamespace(
        qkv_format="thd",
        loss_type="sft",
        chunkflow_chunk_size=4096,
        data_pad_size_multiplier=128,
        chunkflow_keep_last_k=0,
    )
    monkeypatch.setenv("RELAX_CHUNKFLOW_GATE_GAIN_PCT", str(threshold))
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **kw: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    group = object()
    monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: group)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    calls = []

    def agree(peers, local, group):
        calls.append(group)
        peers[:] = [local] * len(peers)

    monkeypatch.setattr(dist, "all_gather_object", agree)
    assert prepare_step_gate(args, model, iterator, 1) is expected
    assert iterator.offset == 0
    assert calls == [group]
    metrics = runtime.consume_step_metrics(4)
    assert metrics["gate_selected"] == int(expected)
    assert metrics["gate_threshold_gain_pct"] == threshold
    assert metrics["schedule_model_speedup"] > 2.4
    assert runtime.consume_step_metrics(4) == {}

    def disagree(peers, local, group):
        peers[:] = [local] * len(peers)
        peers[-1] = (*local[:3], not local[3], local[4])

    monkeypatch.setattr(dist, "all_gather_object", disagree)
    with pytest.raises(ValueError, match="differs between ranks"):
        prepare_step_gate(args, model, iterator, 1)
    assert iterator.offset == 0
