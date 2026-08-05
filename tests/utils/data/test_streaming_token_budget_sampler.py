# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils.data.streaming_token_budget_sampler import StreamingTokenBudgetSampler


class _Partition:
    def __init__(self, indexes: list[int]) -> None:
        self._custom_meta = {index: {"total_lengths": 10} for index in indexes}

    def get_custom_meta(self, indexes: list[int]) -> dict[int, dict[str, int]]:
        return {index: self._custom_meta[index] for index in indexes}


@pytest.mark.parametrize(("n_samples_per_prompt", "initial_ready"), [(1, [0]), (2, [0, 1])])
def test_streaming_sampler_waits_for_equal_size_dp_round(n_samples_per_prompt, initial_ready):
    sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=n_samples_per_prompt)
    ready = list(range(4))
    partition = _Partition(ready)
    common = {
        "task_name": "actor_train",
        "partition_id": "sft_0",
        "dp_size": 4,
        "batch_index": 0,
        "partition": partition,
        "token_budget": 10,
        "production_done": False,
    }

    sampled, consumed = sampler.sample(initial_ready, 0, dp_rank=0, **common)

    assert sampled == consumed == []
    assert sampler._assigned_global[("sft_0", "actor_train")] == set()

    per_dp = [sampler.sample(ready, 0, dp_rank=dp_rank, **common)[0] for dp_rank in range(4)]

    assert all(len(indexes) == 1 for indexes in per_dp)
    assert sorted(index for indexes in per_dp for index in indexes) == ready


class _PartitionWithoutMeta:
    def get_custom_meta(self, indexes):
        return {index: {} for index in indexes}


class _PartitionNoMethod:
    pass


def test_streaming_sampler_rejects_missing_custom_meta_keys():
    sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
    common = {
        "task_name": "actor_train",
        "partition_id": "sft_0",
        "dp_size": 1,
        "batch_index": 0,
        "partition": _PartitionWithoutMeta(),
        "token_budget": 10,
        "production_done": False,
    }
    with pytest.raises(RuntimeError, match="custom_meta contract violated"):
        sampler.sample([0], 0, dp_rank=0, **common)


def test_streaming_sampler_rejects_partition_without_get_custom_meta():
    sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
    common = {
        "task_name": "actor_train",
        "partition_id": "sft_0",
        "dp_size": 1,
        "batch_index": 0,
        "partition": _PartitionNoMethod(),
        "token_budget": 10,
        "production_done": False,
    }
    with pytest.raises(RuntimeError, match="expects the partition to expose get_custom_meta"):
        sampler.sample([0], 0, dp_rank=0, **common)
