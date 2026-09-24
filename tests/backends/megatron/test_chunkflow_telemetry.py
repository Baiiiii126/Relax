# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure-Python tests for ChunkFlow plan telemetry."""

from __future__ import annotations

import math

import pytest


try:
    from relax.backends.megatron.chunkflow.pipeline_schedules import _pipeline_schedule
    from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime
    from relax.backends.megatron.chunkflow.scheduler import build_chunks
    from relax.backends.megatron.chunkflow.telemetry import StepPlanTelemetry, summarize_plan
    from relax.backends.megatron.chunkflow.units import build_micro_batches
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.chunkflow unavailable: {exc}", allow_module_level=True)


def _plan(lengths, *, chunk_size=512, policy="staggered", keep_last_k=1, pp_size=4):
    chunks = build_chunks(lengths, chunk_size=chunk_size, repack_tail=False)
    units = build_micro_batches(chunks, policy)
    warmups = [_pipeline_schedule(units, chunks, keep_last_k, pp_size, stage)[3] for stage in range(pp_size)]
    return summarize_plan(lengths, chunks, units, keep_last_k, pp_size, warmups)


@pytest.mark.parametrize("policy", ["staggered", "lockstep"])
def test_plan_preserves_logical_attention_pairs(policy):
    lengths = [1100, 700, 190]
    plan = _plan(lengths, policy=policy)

    assert plan.q_rows == sum(lengths)
    assert plan.logical_attention_pairs == sum(length * (length + 1) // 2 for length in lengths)
    assert plan.logical_pairs_match == 1
    assert plan.kv_rows > plan.q_rows


def test_plan_counts_padding_and_recompute_work():
    plan = _plan([1100], keep_last_k=1, pp_size=1)

    assert plan.num_chunks == plan.num_units == 3
    assert plan.unit_padded_rows == 512 + 512 + 128
    assert plan.recompute_units == 2
    assert plan.recompute_rows == 1024
    assert plan.p2p_messages == plan.p2p_row_transfers == 0


def test_unbounded_keep_window_has_no_recompute():
    plan = _plan([1100, 300], keep_last_k=1 << 30)

    assert plan.recompute_units == 0
    assert plan.recompute_rows == 0


def test_pair_mismatch_is_observed_without_interrupting_training():
    lengths = [1100]
    chunks = build_chunks(
        lengths,
        chunk_size=512,
        repack_tail=False,
        drop_tail=True,
        drop_tail_max_tokens=128,
    )
    units = build_micro_batches(chunks, "staggered")

    plan = summarize_plan(lengths, chunks, units, 1, 1, [len(units)])

    assert plan.logical_pairs_match == 0


def test_step_aggregation_exposes_separate_benefit_and_cost_proxies():
    first = _plan([1100, 300], keep_last_k=1, pp_size=4)
    second = _plan([700, 200], keep_last_k=1, pp_size=4)
    step = StepPlanTelemetry([first, second])

    metrics = step.to_metrics(pp_size=4)

    assert metrics["num_microbatches"] == 2
    assert metrics["num_units"] == first.num_units + second.num_units
    assert metrics["launch_amplification"] == metrics["num_units"] / 2
    assert metrics["fragmentation_amplification"] >= 1.0
    assert metrics["total_padding_amplification"] >= 1.0
    assert metrics["token_forward_recompute_multiplier"] >= 1.0
    assert metrics["attention_forward_recompute_multiplier"] >= 1.0
    assert math.isclose(metrics["equal_unit_fill_drain_baseline_proxy"], 2.5)
    assert math.isclose(
        metrics["equal_unit_fill_drain_delta_proxy"],
        metrics["equal_unit_fill_drain_baseline_proxy"] - metrics["equal_unit_fill_drain_chunkflow_proxy"],
    )


def test_runtime_resets_step_metrics_and_tracks_local_store_peaks():
    runtime = ChunkflowRuntime(chunk_size=512)
    runtime.begin_step()
    runtime.record_plan(_plan([1100], pp_size=4))

    runtime.store(layer_number=1, group_id=0)
    runtime.store(layer_number=2, group_id=0)
    runtime.store(layer_number=1, group_id=1)
    runtime.release_group(0)
    runtime.release_group(1)

    metrics = runtime.consume_step_metrics(pp_size=4)
    assert metrics["pp_last_max_live_stores"] == 3
    assert metrics["pp_last_max_live_groups"] == 2
    assert runtime.consume_step_metrics(pp_size=4) == {}

    runtime.begin_step()
    assert runtime.consume_step_metrics(pp_size=4) == {}
