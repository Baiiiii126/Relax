import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch


def _load_stream_module(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    transfer_queue = types.ModuleType("transfer_queue")
    tq_dataloader = types.ModuleType("transfer_queue.dataloader")
    streaming_dataloader = types.ModuleType("transfer_queue.dataloader.streaming_dataloader")
    streaming_dataset = types.ModuleType("transfer_queue.dataloader.streaming_dataset")
    tensordict = types.ModuleType("tensordict")

    mpu.get_tensor_model_parallel_rank = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_data_parallel_group = lambda with_context_parallel=True: object()
    core.mpu = mpu

    class _StreamingDataLoader:
        pass

    class _StreamingDataset:
        pass

    streaming_dataloader.StreamingDataLoader = _StreamingDataLoader
    streaming_dataset.StreamingDataset = _StreamingDataset
    tensordict.TensorDict = dict

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "transfer_queue": transfer_queue,
        "transfer_queue.dataloader": tq_dataloader,
        "transfer_queue.dataloader.streaming_dataloader": streaming_dataloader,
        "transfer_queue.dataloader.streaming_dataset": streaming_dataset,
        "tensordict": tensordict,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.utils.data.stream_dataloader", None)
    return importlib.import_module("relax.utils.data.stream_dataloader")


def test_streaming_tq_iterator_stops_at_rollout_mini_sample_limit(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    seen_requests = []
    batches = [
        {"tokens": [torch.tensor([0]), torch.tensor([1])]},
        {"tokens": [torch.tensor([2]), torch.tensor([3])]},
    ]

    def fake_get_data_from_transfer_queue(**kwargs):
        seen_requests.append((kwargs["batch_index"], dict(kwargs["sampling_config"])))
        return batches.pop(0), object()

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get_data_from_transfer_queue)

    iterator = stream_module.StreamingTQIterator(
        args=Namespace(),
        tq_client=object(),
        data_fields=["tokens"],
        rollout_id=7,
        token_budget=1024,
        loss_scale=0.5,
        all_consumed_fn=lambda: False,
        dp_rank=0,
        dp_size=1,
        max_samples=4,
        rollout_mini_index=1,
        start_batch_index=100,
    )

    assert len(next(iterator)[0]["tokens"]) == 2
    assert len(next(iterator)[0]["tokens"]) == 2
    with pytest.raises(StopIteration):
        next(iterator)

    assert [batch_index for batch_index, _ in seen_requests] == [100, 101]
    assert [config["remaining_samples"] for _, config in seen_requests] == [4, 2]
    assert [config["rollout_mini_index"] for _, config in seen_requests] == [1, 1]
    assert len(iterator.get_buffer()) == 2


def test_streaming_tq_iterator_splits_sampler_overfill_into_next_mini(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    overflow_buffer = []
    fetch_count = 0

    def fake_get_data_from_transfer_queue(**kwargs):
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count > 1:
            raise AssertionError("overflow samples should be consumed before fetching more data")
        return {"tokens": [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])]}, object()

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get_data_from_transfer_queue)

    iterator = stream_module.StreamingTQIterator(
        args=Namespace(),
        tq_client=object(),
        data_fields=["tokens"],
        rollout_id=7,
        token_budget=1024,
        loss_scale=0.5,
        all_consumed_fn=lambda: False,
        dp_rank=0,
        dp_size=1,
        max_samples=2,
        overflow_buffer=overflow_buffer,
    )

    current_batch, _ = next(iterator)
    assert [token.item() for token in current_batch["tokens"]] == [0, 1]
    with pytest.raises(StopIteration):
        next(iterator)

    next_iterator = stream_module.StreamingTQIterator(
        args=Namespace(),
        tq_client=object(),
        data_fields=["tokens"],
        rollout_id=7,
        token_budget=1024,
        loss_scale=0.5,
        all_consumed_fn=lambda: False,
        dp_rank=0,
        dp_size=1,
        max_samples=1,
        rollout_mini_index=1,
        overflow_buffer=overflow_buffer,
    )

    overflow_batch, _ = next(next_iterator)
    assert [token.item() for token in overflow_batch["tokens"]] == [2]
    assert fetch_count == 1
