# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compatibility fixes for TransferQueue's streaming token-budget sampler.

Contract with producers: partitions consumed by this sampler MUST attach a
``custom_meta`` payload containing a ``total_lengths`` integer per sample. The
upstream sampler uses that field to size the token budget; consumers of SFT
partitions (``relax/components/sft.py``) pass ``custom_meta`` explicitly when
calling ``data_system_client.async_put``.
"""

from typing import Any

from transfer_queue import StreamingTokenBudgetSampler as _StreamingTokenBudgetSampler


REQUIRED_CUSTOM_META_KEYS: tuple[str, ...] = ("total_lengths",)


class StreamingTokenBudgetSampler(_StreamingTokenBudgetSampler):
    """Wait for an equal-size DP round before invoking TransferQueue balancing,
    and validate the producer-side ``custom_meta`` contract once per sampler
    instance so a missing field surfaces as a clear error instead of a
    downstream KeyError inside the parent sampler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._custom_meta_validated = False

    def _validate_custom_meta_contract(self, partition: Any, indexes: list[int]) -> None:
        if self._custom_meta_validated or not indexes:
            return
        get_custom_meta = getattr(partition, "get_custom_meta", None)
        if get_custom_meta is None:
            raise RuntimeError(
                "StreamingTokenBudgetSampler expects the partition to expose get_custom_meta(); "
                "producers must set custom_meta when putting samples so token budgets can be resolved."
            )
        sample_meta = get_custom_meta([indexes[0]]).get(indexes[0])
        if sample_meta is None:
            raise RuntimeError(
                "StreamingTokenBudgetSampler: partition.get_custom_meta returned no entry for the "
                f"first ready sample (index={indexes[0]}); producers must attach custom_meta on put."
            )
        missing = [key for key in REQUIRED_CUSTOM_META_KEYS if key not in sample_meta]
        if missing:
            raise RuntimeError(
                "StreamingTokenBudgetSampler custom_meta contract violated: sample "
                f"(index={indexes[0]}) is missing required keys {missing}. "
                "Producers must attach these fields on put."
            )
        self._custom_meta_validated = True

    def _run_balance_round(
        self,
        available_ready: list[int],
        balance_unit: int,
        dp_size: int,
        partition: Any,
        buckets: dict[int, list[int]],
        assigned: set[int],
        resolved_lengths: dict[int, int],
    ) -> bool:
        self._validate_custom_meta_contract(partition, available_ready)
        # TransferQueue balances this round with equal_size=True, which requires
        # at least one sample per DP and a sample count divisible by dp_size.
        # Returning False keeps the batch index pollable until enough samples
        # arrive; production_done tails are still handled by its _flush_tail.
        if balance_unit < dp_size or balance_unit % dp_size != 0:
            return False
        return super()._run_balance_round(
            available_ready,
            balance_unit,
            dp_size,
            partition,
            buckets,
            assigned,
            resolved_lengths,
        )
