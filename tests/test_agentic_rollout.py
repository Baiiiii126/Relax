# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import asyncio
import time
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from relax.agentic.pipeline import runtime as runtime_mod
from relax.agentic.pipeline.reward import RewardDomain
from relax.agentic.pipeline.runtime import (
    BackendGenerateResult,
    RuntimeGroup,
    SGLangBackendAdapter,
    _request_envelope_from_sample,
)
from relax.agentic.pipeline.transfer import TransferDomain
from relax.agentic.rollout import AgenticResidentPipeline, _AgenticStepHandle
from relax.agentic.session import service as service_mod
from relax.agentic.session.admission import (
    AdmissionAction,
    AdmissionFeatures,
    AdmissionReason,
    BudgetState,
    WorkerSnapshot,
    compute_reservation_tokens,
    decide_admission_prelude,
    interpret_budget_response,
)
from relax.agentic.session.service import (
    AgenticSessionShard,
    _decide_ir_release,
    _normalized_chat_request,
    _openai_token_logprobs_payload,
    _SessionRecord,
)
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages
from relax.utils.types import Sample


def _runtime_args(**overrides):
    base = {
        "agent_command": "python -c 'pass'",
        "agent_cwd": None,
        "agent_env": [],
        "agentic_prepare_pool_size": None,
        "hf_checkpoint": "/tmp/relax-test-model",
        "mm_processor_pool_size": 0,
        "rollout_batch_size": 2,
        "n_samples_per_prompt": 2,
        "over_sampling_batch_size": None,
        "rollout_max_context_len": 4096,
        "rollout_max_response_len": 128,
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "rollout_skip_special_tokens": False,
        "group_rm": False,
        "reward_max_concurrency": None,
        "partial_rollout": False,
        "fully_async": False,
        "max_staleness": 0,
        "colocate": True,
        "global_batch_size": 2,
        "num_iters_per_train_update": 1,
    }
    base.update(overrides)
    if base["over_sampling_batch_size"] is None:
        base["over_sampling_batch_size"] = base["rollout_batch_size"]
    return SimpleNamespace(**base)


async def test_reward_domain_delegates_sample_reward_to_executor(monkeypatch) -> None:
    calls: list[Sample] = []

    async def fake_async_rm(args, sample):
        del args
        calls.append(sample)
        return 1.0

    monkeypatch.setattr("relax.engine.rewards.async_rm", fake_async_rm)
    reward_domain = RewardDomain(
        args=_runtime_args(reward_max_concurrency=1),
        group_filter=None,
        max_submissions_per_step=None,
    )
    sample = Sample(index=0, group_index=0, session_id="sample-0", metadata={})

    rewarded_sample = await reward_domain._run_sample_reward(sample)

    assert calls == [sample]
    assert rewarded_sample.reward == 1.0


async def test_reward_domain_delegates_group_reward_to_executor(monkeypatch) -> None:
    calls: list[list[Sample]] = []

    async def fake_batched_async_rm(args, samples):
        del args
        calls.append(samples)
        return [float(sample.index) for sample in samples]

    monkeypatch.setattr("relax.engine.rewards.batched_async_rm", fake_batched_async_rm)
    reward_domain = RewardDomain(
        args=_runtime_args(group_rm=True, reward_max_concurrency=1),
        group_filter=None,
        max_submissions_per_step=None,
    )
    group = [Sample(index=index, group_index=0, session_id=f"sample-{index}", metadata={}) for index in range(3)]

    rewarded_group = await reward_domain._run_group_reward(group)

    assert calls == [group]
    assert [sample.reward for sample in rewarded_group] == [0.0, 1.0, 2.0]


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _forest_with_initial_obs(
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    train_token_delta: list[int],
    rollout_token_delta: list[int],
    rollout_id: int = 0,
    metadata: dict[str, Any] | None = None,
    group_index: int | None = None,
    index: int | None = None,
    label: str | None = None,
    train_metadata: dict[str, Any] | None = None,
):
    forest = SessionForest.create_empty(
        session_id=session_id,
        group_index=group_index,
        index=index,
        label=label,
        train_metadata=train_metadata,
        metadata=metadata,
    )
    initial_obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=rollout_id,
        abort_count=0,
        messages_delta=check_messages(messages),
        train_token_delta=list(train_token_delta),
        rollout_token_delta=list(rollout_token_delta),
    )
    return forest, initial_obs


def _make_chat_test_shard(
    *,
    session_sampling_params: dict | None = None,
):
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(
        partial_rollout=True,
        partial_rollout_max_aborted_count=2,
        fully_async=False,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        rollout_max_response_len=8,
        rollout_max_context_len=64,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=1,
    )
    shard.backend = SimpleNamespace(tokenizer=_FakeTokenizer())
    shard._session_records = {}
    shard._session_locks = {}
    shard._evaluating = 0
    shard._terminal_ir_gate_closed = False
    shard._sglang_request_semaphore = None
    shard._sglang_request_limiter = None
    shard._admission_client = None
    shard._admission_enabled = False
    shard._admission_pump_task = None
    shard._admission_stats = {}
    forest, initial_obs = _forest_with_initial_obs(
        session_id="sess-chat",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        train_token_delta=_chars("hello"),
        rollout_token_delta=_chars("hello"),
        rollout_id=0,
        metadata={"template_kwargs": {}},
    )
    record = SimpleNamespace(
        forest=forest,
        next_ir_sequence=0,
        rollout_id=0,
        scope_id="train",
        group_id=None,
        group_generation=None,
        session_seed={"prompt": "hello", "metadata": {"template_kwargs": {}}},
        session_sampling_params=session_sampling_params or {"max_new_tokens": 8},
        resp_state_hash_by_request_id={},
        irs_by_id={},
        ir_queue=deque(),
        active_ir_runner_tasks={},
        pending_chat_waiters={},
        gate_reason=None,
        protected_until_finalize=False,
        admission_deferred=False,
        admission_deferred_since=0.0,
        admission_marked=False,
        admission_aged_resume=False,
    )
    shard._session_records["sess-chat"] = record
    shard._session_locks["sess-chat"] = asyncio.Lock()

    async def _ensure_record(**kwargs):
        del kwargs
        return record, {}

    async def _append_observation_if_needed(**kwargs):
        del kwargs
        return initial_obs.state_hash, {}

    shard._ensure_record = _ensure_record
    shard._match_parent_state_hash = lambda **kwargs: (initial_obs.state_hash, [])
    shard._append_observation_if_needed = _append_observation_if_needed
    shard._budget_sampling_params = lambda **kwargs: dict(kwargs["sampling_params"])
    return shard_cls, shard, record, initial_obs


def _sample_group(name: str, *, group_index: int, rollout_id: int, count: int = 1) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=i,
            session_id=f"{name}-{i}",
            reward=1.0,
            metadata={"start_rollout_id": rollout_id},
        )
        for i in range(count)
    ]


def _pipeline_with_transfer(args):
    runtime = SimpleNamespace(
        args=args,
        rollout_id=0,
        runtime_groups_by_key={},
        interrupted_groups=0,
    )
    runtime.require_rollout_id = lambda: runtime.rollout_id
    runtime.resident_group_keys = lambda: set(runtime.runtime_groups_by_key)
    runtime.accounting_snapshot = lambda: {
        "resident_groups": len(runtime.runtime_groups_by_key),
        "interrupted_groups": runtime.interrupted_groups,
    }

    async def _refresh_interrupted_close_accounting() -> dict[str, int]:
        return {
            "interrupted_groups": runtime.interrupted_groups,
        }

    runtime.refresh_interrupted_close_accounting = _refresh_interrupted_close_accounting
    pipeline = AgenticResidentPipeline()
    pipeline.runtime_domain = runtime
    pipeline.prepare_domain = SimpleNamespace(accounting_snapshot=lambda: {"ready_groups": 0})
    pipeline.reward_domain = SimpleNamespace(
        accounting_snapshot=lambda: {"waiting_groups": 0, "completed_groups": 0, "ready_groups": 0},
        resident_group_keys=lambda: set(),
    )
    pipeline.transfer_domain = TransferDomain(args=args, data_system_client=None)
    return pipeline


def _set_runtime_resident_groups(pipeline, count: int, *, rollout_id: int = 0) -> None:
    pipeline.runtime_domain.runtime_groups_by_key = {
        (("resident", idx),): RuntimeGroup(
            group_key=(("resident", idx),),
            expected_count=1,
            admission_rollout_id=rollout_id,
        )
        for idx in range(count)
    }


def test_session_forest_build_sample_and_request_envelope() -> None:
    forest, initial_obs = _forest_with_initial_obs(
        session_id="sess-build",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        train_token_delta=_chars("hello"),
        rollout_token_delta=_chars("hello"),
        rollout_id=11,
        group_index=3,
        index=7,
        label="lab",
        train_metadata={"loss": "grpo"},
        metadata={"seed_stage": "bootstrap"},
    )
    response_kwargs = {
        "parent_state_hash": initial_obs.state_hash,
        "rollout_id": 11,
        "abort_count": 0,
        "messages_delta": [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}],
        "train_token_delta": _chars("ok"),
        "rollout_token_delta": _chars("ok"),
        "logprob_delta": [-0.1, -0.2],
        "status": "completed",
        "reward": {"score": 0.5},
        "export_metadata_patch": {"request_id": "req-build", "base_state_hash": initial_obs.state_hash},
    }
    leaf = forest.append_resp(**response_kwargs)
    duplicate_leaf = forest.append_resp(**response_kwargs)
    assert duplicate_leaf.state_hash == leaf.state_hash
    assert forest.export_leaf_hashes() == [leaf.state_hash]
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_FakeTokenizer())
    assert (sample.prompt, sample.response, sample.group_index, sample.index) == ("hello", "ok", 3, 7)
    assert sample.train_metadata == {"loss": "grpo"}
    assert sample.metadata["agentic_trace"]["turn_count"] == 1
    envelope = _request_envelope_from_sample(sample, rollout_id=9, sampling_params={"temperature": 0.2})
    assert (envelope.rollout_id, envelope.session_id, envelope.seed.train_metadata) == (
        9,
        "sess-build",
        {"loss": "grpo"},
    )
    with pytest.raises(ValueError, match="group_index"):
        _request_envelope_from_sample(Sample(index=3, prompt="bad"), rollout_id=9)


def test_session_shard_prepare_gate_activation_and_logprobs() -> None:
    shard_cls, shard, record, _initial_obs = _make_chat_test_shard()
    record.group_id = "group-prepare"
    record.group_generation = 3
    record.scope_id = "train"
    record.gate_reason = "prepare"
    backend_calls = {"count": 0}
    backend_return_logprobs = []

    async def _generate(**kwargs):
        backend_return_logprobs.append(kwargs["return_logprob"])
        backend_calls["count"] += 1
        return BackendGenerateResult(
            new_tokens=_chars("ok"), new_log_probs=[-0.1, -0.2], finish_type="stop", meta_info={}, elapsed=0.1
        )

    shard.backend.generate = _generate

    async def _run():
        chat_task = asyncio.create_task(
            shard_cls.chat(
                shard,
                session_id="sess-chat",
                messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                tools=[],
                chat_template_kwargs=None,
                temperature=None,
                top_p=None,
                max_completion_tokens=None,
                stop=None,
                seed=None,
                logprobs=False,
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
            if record.ir_queue:
                break
        assert backend_calls["count"] == 0
        assert await shard_cls.prepare_group_status(shard, scope_id="train") == [
            {"group_id": "group-prepare", "group_generation": 3, "total_sessions": 1, "ready_sessions": 1}
        ]
        activation = await shard_cls.activate_group_sessions(
            shard,
            scope_id="train",
            groups=[{"group_id": "group-prepare", "group_generation": 3}],
            rollout_id=7,
        )
        assert activation == {"activated_sessions": 1, "started_sessions": 1}
        return await chat_task

    payload = asyncio.run(_run())
    assert payload["message"]["content"] == "ok"
    assert payload["logprobs"] is None
    assert backend_return_logprobs == [True]
    assert backend_calls["count"] == 1
    assert record.rollout_id == 7


def test_chat_request_validation_context_limit_and_logprob_payload() -> None:
    assert (
        _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "logprobs": True})["logprobs"]
        is True
    )
    with pytest.raises(HTTPException, match="logprobs must be a boolean"):
        _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "logprobs": "true"})
    with pytest.raises(HTTPException, match="top_logprobs is not supported"):
        _normalized_chat_request({"messages": [{"role": "user", "content": "hello"}], "top_logprobs": 1})
    assert (
        _openai_token_logprobs_payload(tokenizer=_FakeTokenizer(), token_ids=_chars("ok"), token_logprobs=[-0.1])[
            "content"
        ][1]["logprob"]
        == -9999.0
    )

    shard_cls, shard, record, _initial_obs = _make_chat_test_shard(session_sampling_params={"max_new_tokens": 0})
    shard.backend.generate = lambda **kwargs: (_ for _ in ()).throw(AssertionError("backend should not run"))

    async def _run() -> dict[str, Any]:
        return await shard_cls.chat(
            shard,
            session_id="sess-chat",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tools=[],
            chat_template_kwargs=None,
            temperature=None,
            top_p=None,
            max_completion_tokens=None,
            stop=None,
            seed=None,
        )

    payload = asyncio.run(_run())
    assert payload["_http_status"] == 400
    assert record.pending_chat_waiters == record.irs_by_id == record.active_ir_runner_tasks == {}


@pytest.mark.parametrize(
    ("fully_async", "initial_abort_count", "expected_kind", "expected_protected", "expected_sibling_kind"),
    [
        (False, 0, RequestKind.RESUMED, False, RequestKind.FRESH),
        (False, 1, RequestKind.PROTECTED, True, RequestKind.PROTECTED),
        (True, 1, RequestKind.PROTECTED, True, RequestKind.PROTECTED),
    ],
)
def test_ir_gate_and_requeue_policy(
    fully_async: bool,
    initial_abort_count: int,
    expected_kind: RequestKind,
    expected_protected: bool,
    expected_sibling_kind: RequestKind,
) -> None:
    shard_cls, shard, record, _initial_obs = _make_chat_test_shard()
    shard.args.fully_async = fully_async
    shard._release_ir_locked = lambda record, ir_id: record.irs_by_id.pop(ir_id, None)
    shard._enqueue_ir_locked = lambda record, ir: record.ir_queue.append(ir.request_id)
    ir = SimpleNamespace(request_id="req-1", abort_count=initial_abort_count, kind=None, pending_status=None)
    sibling = SimpleNamespace(request_id="req-sibling", kind=RequestKind.FRESH)
    record.irs_by_id = {ir.request_id: ir, sibling.request_id: sibling}
    assert _decide_ir_release(record=record).allow is True
    record.gate_reason = "prepare"
    assert _decide_ir_release(record=record).blocked_reason == "prepare_gate"
    record.gate_reason = None

    shard_cls._requeue_aborted_ir_locked(shard, record=record, ir_id=ir.request_id, ir=ir)
    assert (ir.kind, record.gate_reason, record.protected_until_finalize, sibling.kind) == (
        expected_kind,
        "partial_resume",
        expected_protected,
        expected_sibling_kind,
    )


def test_admission_quota_prepare_isolation_and_resident_tail_carry() -> None:
    args = _runtime_args(
        partial_rollout=True, rollout_batch_size=32, over_sampling_batch_size=48, n_samples_per_prompt=1
    )
    pipeline = _pipeline_with_transfer(args)
    pipeline.transfer_domain.rebind_step(rollout_id=1)
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=32)
    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    assert snapshot["current_partition_quota"] == 32
    assert pipeline._current_window_admission_counts(resident_group_count=0, transfer_snapshot=snapshot)[1:] == (
        0,
        0,
        48,
    )

    pipeline.prepare_domain = SimpleNamespace(accounting_snapshot=lambda: {"ready_groups": 99})
    assert pipeline._current_window_admission_counts(resident_group_count=0, transfer_snapshot=snapshot)[1:] == (
        0,
        0,
        48,
    )

    pipeline.transfer_domain._committed_current_group_count = 32
    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    _set_runtime_resident_groups(pipeline, 16)
    step_handle = _AgenticStepHandle(rollout_id=1, required_group_count=32, terminal_step=False)
    assert pipeline._close_status(step_handle) is None
    assert pipeline._current_window_admission_counts(
        resident_group_count=pipeline.resident_group_count, transfer_snapshot=snapshot
    )[1:] == (16, 48, 0)

    pipeline.transfer_domain.rebind_step(rollout_id=2)
    pipeline.runtime_domain.rollout_id = 2
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=32)
    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    assert pipeline._current_window_admission_counts(
        resident_group_count=pipeline.resident_group_count, transfer_snapshot=snapshot
    )[1:] == (16, 16, 32)


@pytest.mark.parametrize(
    (
        "fully_async",
        "terminal_step",
        "previous_quota",
        "committed_previous",
        "committed_current",
        "interrupted_groups",
        "required_groups",
        "expected_finish",
        "status",
    ),
    [
        (True, False, 0, 0, 0, 0, 2, False, "committed_target"),
        (True, False, 0, 0, 0, 2, 2, True, None),
        (False, False, 0, 0, 0, 0, 2, False, "committed_target"),
        (True, True, 0, 0, 0, 2, 2, True, None),
        (True, False, 4, 4, 0, 8, 8, True, None),
    ],
)
def test_finish_eligibility_interrupted_policy(
    fully_async: bool,
    terminal_step: bool,
    previous_quota: int,
    committed_previous: int,
    committed_current: int,
    interrupted_groups: int,
    required_groups: int,
    expected_finish: bool,
    status: str | None,
) -> None:
    args = _runtime_args(fully_async=fully_async, rollout_batch_size=required_groups, n_samples_per_prompt=1)
    pipeline = _pipeline_with_transfer(args)
    pipeline.transfer_domain.rebind_step(rollout_id=3)
    pipeline.runtime_domain.rollout_id = 3
    pipeline.transfer_domain.configure_transfer_quota(
        previous_partition_quota=previous_quota,
        current_partition_quota=required_groups,
    )
    pipeline.transfer_domain._committed_previous_group_count = committed_previous
    pipeline.transfer_domain._committed_current_group_count = committed_current
    pipeline.runtime_domain.interrupted_groups = interrupted_groups
    step_handle = _AgenticStepHandle(rollout_id=3, required_group_count=required_groups, terminal_step=terminal_step)
    assert pipeline._close_status(step_handle) == status
    assert (status is None) is expected_finish


def test_transfer_fifo_routes_slots_by_arrival_ignoring_metadata(monkeypatch) -> None:
    recorded: list[tuple[int, list[str]]] = []
    recorded_is_last: list[tuple[int, bool]] = []

    async def _fake_transfer(args, batch_samples, batch_count, rollout_id, data_system_client, is_last=False):
        del args, batch_count, data_system_client
        recorded.append((int(rollout_id), [s.metadata["label"] for group in batch_samples for s in group]))
        recorded_is_last.append((int(rollout_id), bool(is_last)))

    monkeypatch.setattr("relax.agentic.pipeline.transfer._transfer_batch_to_data_system", _fake_transfer)
    args = _runtime_args(fully_async=True, rollout_batch_size=2, n_samples_per_prompt=1)
    transfer = TransferDomain(args=args, data_system_client=object())
    transfer.rebind_step(rollout_id=3)
    transfer.configure_transfer_quota(previous_partition_quota=2, current_partition_quota=2)
    labels = ["current-first", "old-second", "old-third", "current-fourth"]
    for idx, label in enumerate(labels):
        group = _sample_group(label, group_index=idx, rollout_id=3 if "current" in label else 2)
        for sample in group:
            sample.metadata.update(label=label, admission_rollout_id=3 if "current" in label else 2)
        transfer.enqueue_ready_groups([group])
    released_groups, released_count = asyncio.run(transfer.drain_ready_group_payloads())
    assert released_count == 4
    assert len(released_groups) == 4
    asyncio.run(transfer.wait_for_pending_transfers())
    assert recorded == [(2, ["current-first", "old-second"]), (3, ["old-third", "current-fourth"])]
    # Both partitions are fully filled this step (prev quota=2 backfilled, current
    # quota=2 met), so each partition's batch is marked is_last for end-of-stream.
    assert recorded_is_last == [(2, True), (3, True)]


def test_transfer_is_last_only_when_partition_target_met(monkeypatch) -> None:
    # Current partition NOT fully filled this step (quota=4, only 2 committed) -> its
    # batch is NOT marked is_last; the tail is backfilled next step (as the previous
    # partition), which is where is_last fires. Guards against premature end-of-stream.
    recorded_is_last: list[tuple[int, bool]] = []

    async def _fake_transfer(args, batch_samples, batch_count, rollout_id, data_system_client, is_last=False):
        del args, batch_samples, batch_count, data_system_client
        recorded_is_last.append((int(rollout_id), bool(is_last)))

    monkeypatch.setattr("relax.agentic.pipeline.transfer._transfer_batch_to_data_system", _fake_transfer)
    args = _runtime_args(fully_async=True, rollout_batch_size=4, over_sampling_batch_size=4, n_samples_per_prompt=1)
    transfer = TransferDomain(args=args, data_system_client=object())
    transfer.rebind_step(rollout_id=5)
    transfer.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=4)
    # Only 2 groups available this step -> current partition under-filled (deficit 2).
    for idx in range(2):
        transfer.enqueue_ready_groups([_sample_group(f"g{idx}", group_index=idx, rollout_id=5)])
    asyncio.run(transfer.drain_ready_group_payloads())
    asyncio.run(transfer.wait_for_pending_transfers())
    # No is_last: the current partition will be completed next step via backfill.
    assert recorded_is_last == [(5, False)]


def test_oversampling_surplus_retained_not_dropped(monkeypatch) -> None:
    # When over_sampling_batch_size > rollout_batch_size, completed groups beyond the
    # commit target (current_partition_quota) stay in ready_group_buffer (NOT dropped),
    # so the next step's current partition can re-commit them. We do NOT account them
    # separately — the next-step previous-partition debt is sized from the deficit, not
    # from the buffer (see test_previous_quota_is_current_partition_deficit).
    async def _fake_transfer(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr("relax.agentic.pipeline.transfer._transfer_batch_to_data_system", _fake_transfer)
    args = _runtime_args(fully_async=True, rollout_batch_size=2, over_sampling_batch_size=4, n_samples_per_prompt=1)
    transfer = TransferDomain(args=args, data_system_client=object())
    transfer.rebind_step(rollout_id=3)
    transfer.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=2)
    for idx in range(4):  # over-sample: 4 ready groups, commit target only 2
        transfer.enqueue_ready_groups([_sample_group(f"g{idx}", group_index=idx, rollout_id=3)])
    _released_groups, released_count = asyncio.run(transfer.drain_ready_group_payloads())

    assert released_count == 2  # only current_partition_quota committed
    assert len(transfer.ready_group_buffer) == 2  # surplus preserved, not dropped


def test_previous_quota_is_current_partition_deficit() -> None:
    # Core fix: previous_partition_quota (next-step backfill debt) equals how many groups
    # the previous step left short of its current-partition target (rollout_batch_size),
    # i.e. rollout_batch_size - committed_current. It is INDEPENDENT of any over-sampling
    # surplus still resident in the transfer ready buffer.
    args = _runtime_args(fully_async=True, rollout_batch_size=4, over_sampling_batch_size=6, n_samples_per_prompt=1)
    pipeline = _pipeline_with_transfer(args)

    # Case A: previous step met its target (committed_current == rollout_batch_size).
    # Even with surplus left in the buffer, the deficit (and thus next-step debt) is 0.
    pipeline.transfer_domain.rebind_step(rollout_id=0)
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=4)
    pipeline.transfer_domain._committed_current_group_count = 4
    for idx in range(2):  # 2 surplus completed groups parked in the buffer
        pipeline.transfer_domain.enqueue_ready_groups([_sample_group(f"s{idx}", group_index=idx, rollout_id=0)])
    end_snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    required = 4  # required_group_count == current_partition_quota
    deficit = max(required - end_snapshot["committed_current_groups"], 0)
    assert deficit == 0  # met target → no debt, regardless of the 2 buffered surplus groups

    # Case B: previous step fell short by 1 (an aborted group never came back).
    pipeline.transfer_domain.rebind_step(rollout_id=1)
    pipeline.transfer_domain.configure_transfer_quota(previous_partition_quota=0, current_partition_quota=4)
    pipeline.transfer_domain._committed_current_group_count = 3
    end_snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    deficit = max(required - end_snapshot["committed_current_groups"], 0)
    assert deficit == 1  # short by exactly 1 → next step backfills 1


def test_deficit_quota_keeps_admission_ledger_consistent() -> None:
    # With the deficit-sized previous quota, the admission ledger stays self-consistent
    # even when over-sampling surplus is resident: surplus is folded into the current
    # window (resident_current_window_groups), not the previous debt, and no RuntimeError
    # invariant fires in _current_window_admission_counts.
    args = _runtime_args(fully_async=True, rollout_batch_size=4, over_sampling_batch_size=6, n_samples_per_prompt=1)
    pipeline = _pipeline_with_transfer(args)
    pipeline.transfer_domain.rebind_step(rollout_id=1)
    pipeline.runtime_domain.rollout_id = 1

    # Previous step left a deficit of 1; pipeline holds 1 aborted group (runtime) plus
    # 2 over-sampling surplus groups (transfer ready buffer).
    pipeline._last_step_current_deficit = 1
    _set_runtime_resident_groups(pipeline, 1, rollout_id=0)
    for idx in range(2):
        pipeline.transfer_domain.enqueue_ready_groups([_sample_group(f"surplus{idx}", group_index=idx, rollout_id=1)])

    previous_partition_quota = pipeline._last_step_current_deficit
    pipeline.transfer_domain.configure_transfer_quota(
        previous_partition_quota=previous_partition_quota, current_partition_quota=4
    )
    resident_group_count = pipeline.resident_group_count
    assert resident_group_count == 3  # 1 abort + 2 surplus

    snapshot = dict(pipeline.transfer_domain.accounting_snapshot())
    remaining_previous_debt, resident_current_window_groups, _, current_window_slack = (
        pipeline._current_window_admission_counts(
            resident_group_count=resident_group_count, transfer_snapshot=snapshot
        )
    )
    assert remaining_previous_debt == 1  # only the genuine deficit is debt
    assert resident_current_window_groups == 2  # the 2 surplus folded into current window
    assert current_window_slack >= 0


# ── Admission + session lifecycle ──────────────────────────────────────────────────────


def _admission_features(**overrides) -> AdmissionFeatures:
    base = dict(
        enabled=True,
        session_id="sess",
        scope_allowed=True,
        is_protected=False,
        marked=False,
        prompt_tokens=100,
        expected_decode_tokens=8,
        reservation_tokens=108,
        dispatch_id="req:0",
        admission_decision_id="d0",
        serving_weight_version=None,
        aged=False,
    )
    base.update(overrides)
    return AdmissionFeatures(**base)


class _FakeBudgetClient:
    def __init__(self, *, grants=None, hint=None) -> None:
        self.reserve_reqs: list[dict] = []
        self.released: list[str] = []
        self._grants = list(grants or [])
        self._hint = (
            hint
            if hint is not None
            else {
                "degraded": False,
                "available": 10**9,
                "ceiling": 10**9,
                "reserved": 0,
                "epoch": 1,
            }
        )

    async def reserve(self, req):
        self.reserve_reqs.append(req)
        if self._grants:
            return self._grants.pop(0)
        return {
            "granted": True,
            "reason": "capacity_available",
            "lease_id": f"L{len(self.reserve_reqs)}",
            "owner_epoch": 1,
            "reservation_tokens": req["tokens"],
        }

    async def release(self, lease_id, actual_tokens=None):
        self.released.append(lease_id)

    async def capacity_hint(self):
        return dict(self._hint)


class _RaisingBudgetClient:
    async def reserve(self, req):
        raise RuntimeError("coordinator down")

    async def release(self, lease_id, actual_tokens=None):
        return None

    async def capacity_hint(self):
        raise RuntimeError("coordinator down")


def _make_admission_shard(*, client=None, enabled=True, scope="train", max_wait_s=30.0, forced_cap=8):
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(
        agentic_program_admission=enabled,
        agentic_admission_scope=scope,
        agentic_admission_expected_decode_cap=8,
        rollout_max_response_len=8,
        agentic_session_lifecycle=False,
    )
    shard._admission_client = client
    shard._admission_enabled = bool(enabled) and client is not None
    shard._admission_scope = scope
    shard._admission_expected_decode_cap = 8
    shard._rollout_max_response_len = 8
    shard._admission_reconcile_interval_s = 0.01
    shard._admission_max_wait_s = max_wait_s
    shard._admission_forced_resume_cap = forced_cap
    shard._admission_pump_task = None
    shard._admission_stats = {}
    shard._session_records = {}
    shard._session_locks = {}
    return shard_cls, shard


def test_generate_sends_session_id_only_when_lifecycle_enabled(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    async def _fake_post(url, payload, headers=None):
        captured["payload"] = dict(payload)
        captured["headers"] = headers
        raise _Stop()

    monkeypatch.setattr(runtime_mod, "post", _fake_post)
    adapter = object.__new__(SGLangBackendAdapter)
    adapter._resolved_router_ip = "10.0.0.1"
    adapter._resolved_router_port = 8000
    adapter._use_rollout_routing_replay = False
    adapter._router_policy = "cache_aware"
    adapter._slime_router_sticky = False
    adapter.tokenizer = _FakeTokenizer()
    adapter.compiler = SimpleNamespace(processor=None)

    adapter._session_lifecycle = True
    with pytest.raises(_Stop):
        asyncio.run(adapter.generate(input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 4}, session_id="sess-9"))
    assert captured["payload"]["session_id"] == "sess-9"
    assert captured["headers"] is None  # cache_aware policy -> no sticky routing header

    captured.clear()
    adapter._session_lifecycle = False
    with pytest.raises(_Stop):
        asyncio.run(adapter.generate(input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 4}, session_id="sess-9"))
    assert "session_id" not in captured["payload"]


def test_close_engine_sessions_fans_out_to_all_engines(monkeypatch) -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(
        agentic_session_lifecycle=True,
        agentic_session_close_timeout_ms=500,
        agentic_session_close_max_retries=2,
    )
    posts: list[tuple] = []

    async def _fake_urls(args):
        return ["http://e0:1", "http://e1:1"]

    async def _fake_post(url, payload, max_retries=6):
        posts.append((url, payload))

    monkeypatch.setattr(service_mod, "_sglang_worker_urls", _fake_urls)
    monkeypatch.setattr(service_mod, "post", _fake_post)
    asyncio.run(shard_cls._close_engine_sessions(shard, ["sess-1"]))
    assert sorted(posts) == [
        ("http://e0:1/close_session", {"session_id": "sess-1"}),
        ("http://e1:1/close_session", {"session_id": "sess-1"}),
    ]


def test_close_is_noop_when_disabled(monkeypatch) -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(
        agentic_session_lifecycle=False,
        agentic_session_close_timeout_ms=500,
        agentic_session_close_max_retries=2,
    )
    posts: list[tuple] = []

    async def _fake_urls(args):
        return ["http://e0:1"]

    async def _fake_post(url, payload, max_retries=6):
        posts.append((url, payload))

    monkeypatch.setattr(service_mod, "_sglang_worker_urls", _fake_urls)
    monkeypatch.setattr(service_mod, "post", _fake_post)
    asyncio.run(shard_cls._close_engine_sessions(shard, ["sess-1"]))
    assert posts == []


def test_close_is_fail_open_on_post_error(monkeypatch) -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(
        agentic_session_lifecycle=True,
        agentic_session_close_timeout_ms=500,
        agentic_session_close_max_retries=2,
    )

    async def _fake_urls(args):
        return ["http://e0:1"]

    async def _fake_post(url, payload, max_retries=6):
        raise RuntimeError("engine unreachable")

    monkeypatch.setattr(service_mod, "_sglang_worker_urls", _fake_urls)
    monkeypatch.setattr(service_mod, "post", _fake_post)
    # Must not raise: close is best-effort and never blocks the terminal path.
    asyncio.run(shard_cls._close_engine_sessions(shard, ["sess-1"]))


def test_finalize_closes_once_only_when_record_removed() -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace()
    close_calls: list[list[str]] = []

    async def _fake_close(ids):
        close_calls.append(list(ids))

    async def _fake_abort(ids):
        return None

    shard._close_engine_sessions = _fake_close
    shard._abort_backend_request_ids = _fake_abort

    lock = asyncio.Lock()
    shard._session_records = {}
    shard._session_locks = {"sess-x": lock}
    result = asyncio.run(
        shard_cls._finish_discarded_session(
            shard,
            session_id="sess-x",
            lock=lock,
            removed=_SessionRecord(),
            active_tasks=[],
            backend_request_ids=[],
            waiters=[],
            stats={"node_count": 0, "request_count": 0},
        )
    )
    assert result is True
    assert close_calls == [["sess-x"]]

    close_calls.clear()
    lock2 = asyncio.Lock()
    shard._session_locks = {"sess-y": lock2}
    result2 = asyncio.run(
        shard_cls._finish_discarded_session(
            shard,
            session_id="sess-y",
            lock=lock2,
            removed=None,
            active_tasks=[],
            backend_request_ids=[],
            waiters=[],
            stats=None,
        )
    )
    assert result2 is False
    assert close_calls == []  # nothing removed -> no close


def test_compute_reservation_tokens() -> None:
    assert (
        compute_reservation_tokens(
            prompt_tokens=100, sampling_max_new_tokens=50, expected_decode_cap=200, rollout_max_response_len=8192
        )
        == 150
    )
    assert (
        compute_reservation_tokens(
            prompt_tokens=100, sampling_max_new_tokens=None, expected_decode_cap=None, rollout_max_response_len=8192
        )
        == 100 + 8192
    )
    assert (
        compute_reservation_tokens(
            prompt_tokens=0, sampling_max_new_tokens=0, expected_decode_cap=0, rollout_max_response_len=0
        )
        == 0
    )


def test_decide_admission_prelude_branches() -> None:
    assert decide_admission_prelude(_admission_features(enabled=False)).reason_code is AdmissionReason.FEATURE_DISABLED
    assert decide_admission_prelude(_admission_features(session_id="")).reason_code is AdmissionReason.MISSING_IDENTITY
    assert (
        decide_admission_prelude(_admission_features(scope_allowed=False)).reason_code
        is AdmissionReason.CAPABILITY_UNAVAILABLE
    )
    protected = decide_admission_prelude(_admission_features(is_protected=True))
    assert protected.action is AdmissionAction.BYPASS and protected.reason_code is AdmissionReason.FAIRNESS_RESERVE
    marked = decide_admission_prelude(_admission_features(marked=True))
    assert marked.action is AdmissionAction.DEFER and marked.reason_code is AdmissionReason.PRESSURE_GUARD
    assert decide_admission_prelude(_admission_features()) is None


def test_budget_state_reserve_release_and_exhaust() -> None:
    budget = BudgetState(
        headroom=1.0, pressure_threshold=1.0, emergency_reserve_frac=0.0, lease_ttl_s=10.0, staleness_s=30.0
    )
    budget.reconcile([WorkerSnapshot("e0", 1000, 0.1)], now=100.0)
    assert budget.ceiling == 1000 and budget.epoch == 1
    grant = budget.reserve(tokens=600, dispatch_id="r1:0", admission_decision_id="d1", aged=False, now=100.0)
    assert grant.granted and grant.lease_id == "1:r1:0:d1"
    # idempotent: same dispatch does not double-count
    again = budget.reserve(tokens=600, dispatch_id="r1:0", admission_decision_id="d1", aged=False, now=100.0)
    assert again.lease_id == grant.lease_id and budget.reserved == 600
    exhausted = budget.reserve(tokens=600, dispatch_id="r2:0", admission_decision_id="d2", aged=False, now=100.0)
    assert not exhausted.granted and exhausted.reason is AdmissionReason.CAPACITY_EXHAUSTED
    budget.release(grant.lease_id)
    assert budget.reserved == 0
    budget.release(grant.lease_id)  # idempotent no-op
    assert budget.reserved == 0


def test_budget_state_pressure_and_emergency_reserve() -> None:
    pressured = BudgetState(
        headroom=1.0, pressure_threshold=0.90, emergency_reserve_frac=0.10, lease_ttl_s=10.0, staleness_s=30.0
    )
    pressured.reconcile([WorkerSnapshot("e0", 1000, 0.0, num_used_tokens=950)], now=1.0)
    # usage derived elsewhere; force max_usage via a high-usage snapshot
    pressured.reconcile([WorkerSnapshot("e0", 1000, 0.95)], now=1.0)
    blocked = pressured.reserve(tokens=10, dispatch_id="a:0", admission_decision_id="d", aged=False, now=1.0)
    assert not blocked.granted and blocked.reason is AdmissionReason.PRESSURE_GUARD
    aged_ok = pressured.reserve(tokens=10, dispatch_id="a:0", admission_decision_id="d", aged=True, now=1.0)
    assert aged_ok.granted  # aged bypasses the pressure guard

    emergency = BudgetState(
        headroom=1.0, pressure_threshold=1.0, emergency_reserve_frac=0.10, lease_ttl_s=10.0, staleness_s=30.0
    )
    emergency.reconcile([WorkerSnapshot("e0", 1000, 0.0)], now=1.0)
    assert not emergency.reserve(tokens=950, dispatch_id="b:0", admission_decision_id="d", aged=False, now=1.0).granted
    assert emergency.reserve(tokens=950, dispatch_id="b:0", admission_decision_id="d", aged=True, now=1.0).granted


def test_budget_state_ttl_reconcile_and_staleness() -> None:
    ttl = BudgetState(
        headroom=1.0, pressure_threshold=1.0, emergency_reserve_frac=0.0, lease_ttl_s=5.0, staleness_s=30.0
    )
    ttl.reconcile([WorkerSnapshot("e0", 1000, 0.0)], now=0.0)
    ttl.reserve(tokens=100, dispatch_id="x:0", admission_decision_id="d", aged=False, now=0.0)
    assert ttl.reserved == 100
    assert ttl.expire_ttl(now=10.0) == 1 and ttl.reserved == 0

    churn = BudgetState(
        headroom=1.0, pressure_threshold=1.0, emergency_reserve_frac=0.0, lease_ttl_s=100.0, staleness_s=30.0
    )
    churn.reconcile([WorkerSnapshot("e0", 1000, 0.0)], now=0.0)
    epoch0 = churn.epoch
    churn.reserve(tokens=100, dispatch_id="y:0", admission_decision_id="d", aged=False, now=0.0)
    churn.reconcile([WorkerSnapshot("e0", 1000, 0.0), WorkerSnapshot("e1", 1000, 0.0)], now=1.0)
    assert churn.epoch == epoch0 + 1 and churn.reserved == 0  # worker-set change drops stale-epoch leases

    stale = BudgetState(
        headroom=1.0, pressure_threshold=1.0, emergency_reserve_frac=0.0, lease_ttl_s=100.0, staleness_s=5.0
    )
    stale.reconcile([WorkerSnapshot("e0", 1000, 0.0)], now=0.0)
    degraded = stale.reserve(tokens=10, dispatch_id="z:0", admission_decision_id="d", aged=False, now=100.0)
    assert not degraded.granted and degraded.reason is AdmissionReason.DEGRADED


def test_parse_engine_kv_gauges_unsums_tp_replicated_max_total() -> None:
    """Regression: SGLang replicates max_total_num_tokens once per tp_rank; the generic
    summing parser inflated capacity by the TP degree and deflated the derived usage by the
    same factor, so a saturated engine (0.92) read as ~idle (0.23) and the pressure guard
    never fired. Aggregation must be MAX per gauge."""
    from relax.agentic.session.admission_coordinator import _parse_engine_kv_gauges

    # 4 tp_rank lines with an identical per-engine pool size + single-line scheduler gauges.
    text = "\n".join(
        [
            "# HELP sglang:max_total_num_tokens KV pool size",
            "# TYPE sglang:max_total_num_tokens gauge",
            'sglang:max_total_num_tokens{tp_rank="0"} 262144.0',
            'sglang:max_total_num_tokens{tp_rank="1"} 262144.0',
            'sglang:max_total_num_tokens{tp_rank="2"} 262144.0',
            'sglang:max_total_num_tokens{tp_rank="3"} 262144.0',
            'sglang:num_used_tokens{tp_rank="0"} 239901.0',
            'sglang:token_usage{tp_rank="0"} 0.915',
            'sglang:num_running_reqs{tp_rank="0"} 16.0',
        ]
    )
    gauges = _parse_engine_kv_gauges(text)
    # max, NOT the 4x sum (1048576) the generic parser produced.
    assert gauges["sglang:max_total_num_tokens"] == 262144.0
    assert gauges["sglang:num_used_tokens"] == 239901.0
    assert abs(gauges["sglang:token_usage"] - 0.915) < 1e-9
    # Derived usage from absolute counts must reflect the true ~0.92, not the 4x-deflated 0.23.
    assert abs(gauges["sglang:num_used_tokens"] / gauges["sglang:max_total_num_tokens"] - 0.915) < 1e-3


def test_budget_state_peak_usage_running_max() -> None:
    """The once-per-step metrics read (coordinator health) is sampled after the
    rollout has drained, so the instantaneous usage undershoots.

    capacity_hint must expose a running peak/mean over reconciles, drained only
    on the resetting read; the resume-pump read must not steal it.
    """
    st = BudgetState(headroom=1.0, pressure_threshold=2.0, emergency_reserve_frac=0.0, staleness_s=1e9)
    # A rollout window: usage rises to a peak, then drains to near-idle before the per-step read.
    st.reconcile([WorkerSnapshot("e0", 1000, 0.30)], now=0.0)
    st.reconcile([WorkerSnapshot("e0", 1000, 0.90)], now=1.0)  # peak
    st.reconcile([WorkerSnapshot("e0", 1000, 0.05)], now=2.0)  # drained to idle

    # Resume-pump style read: sees the true peak, keeps the instantaneous, does NOT drain.
    hint = st.capacity_hint(now=2.0)
    assert abs(hint["peak_usage"] - 0.90) < 1e-9
    assert hint["max_usage"] == 0.05
    assert abs(hint["window_mean_usage"] - (0.30 + 0.90 + 0.05) / 3) < 1e-9

    # Per-step read drains the window; this read still reports the peak.
    drained = st.capacity_hint(now=2.0, reset_peak=True)
    assert abs(drained["peak_usage"] - 0.90) < 1e-9

    # Next window starts fresh from post-drain reconciles only.
    st.reconcile([WorkerSnapshot("e0", 1000, 0.10)], now=3.0)
    after = st.capacity_hint(now=3.0)
    assert abs(after["peak_usage"] - 0.10) < 1e-9
    assert abs(after["window_mean_usage"] - 0.10) < 1e-9


def test_interpret_budget_response() -> None:
    features = _admission_features()
    admit = interpret_budget_response(
        {
            "granted": True,
            "reason": "capacity_available",
            "lease_id": "L",
            "owner_epoch": 3,
            "reservation_tokens": 108,
        },
        features,
    )
    assert admit.action is AdmissionAction.ADMIT and admit.lease_id == "L"
    degraded = interpret_budget_response(
        {"granted": False, "reason": "degraded", "lease_id": None, "owner_epoch": -1, "reservation_tokens": 108},
        features,
    )
    assert degraded.action is AdmissionAction.BYPASS
    for reason in ("capacity_exhausted", "pressure_guard"):
        deferred = interpret_budget_response(
            {"granted": False, "reason": reason, "lease_id": None, "owner_epoch": 2, "reservation_tokens": 108},
            features,
        )
        assert deferred.action is AdmissionAction.DEFER


def test_admit_ir_grant_defer_failopen_and_aged() -> None:
    granted = _make_admission_shard(client=_FakeBudgetClient())
    admit = asyncio.run(granted[0]._admit_ir(granted[1], features=_admission_features()))
    assert admit.action is AdmissionAction.ADMIT and admit.lease_id
    assert granted[1]._admission_client.reserve_reqs[0]["tokens"] == 108

    exhausted_grant = {
        "granted": False,
        "reason": "capacity_exhausted",
        "lease_id": None,
        "owner_epoch": 1,
        "reservation_tokens": 108,
    }
    deferring = _make_admission_shard(client=_FakeBudgetClient(grants=[dict(exhausted_grant)]))
    deferred = asyncio.run(deferring[0]._admit_ir(deferring[1], features=_admission_features()))
    assert deferred.action is AdmissionAction.DEFER and deferred.reason_code is AdmissionReason.CAPACITY_EXHAUSTED

    failing = _make_admission_shard(client=_RaisingBudgetClient())
    degraded = asyncio.run(failing[0]._admit_ir(failing[1], features=_admission_features()))
    assert degraded.action is AdmissionAction.BYPASS and degraded.reason_code is AdmissionReason.DEGRADED

    aged_client = _make_admission_shard(client=_FakeBudgetClient(grants=[dict(exhausted_grant)]))
    aged = asyncio.run(aged_client[0]._admit_ir(aged_client[1], features=_admission_features(aged=True)))
    assert aged.action is AdmissionAction.BYPASS and aged.reason_code is AdmissionReason.FAIRNESS_RESERVE

    disabled = _make_admission_shard(client=_FakeBudgetClient())
    bypass = asyncio.run(disabled[0]._admit_ir(disabled[1], features=_admission_features(enabled=False)))
    assert bypass.reason_code is AdmissionReason.FEATURE_DISABLED
    assert disabled[1]._admission_client.reserve_reqs == []  # never consulted the coordinator


def test_admission_defer_requeues_and_gates_release() -> None:
    shard_cls, shard = _make_admission_shard(client=_FakeBudgetClient(), enabled=False)  # disable pump in unit test
    record = _SessionRecord()
    ir = InflightRequest(request_id="r1", parent_state_hash="p", rollout_id=0, kind=RequestKind.FRESH, abort_count=0)
    record.irs_by_id["r1"] = ir
    record.active_ir_runner_tasks["r1"] = object()
    record.ir_queue = deque(["r1"])
    shard_cls._admission_defer_ir_locked(shard, record=record, ir_id="r1", ir=ir)
    assert record.admission_deferred is True
    assert record.admission_deferred_since > 0.0
    assert "r1" not in record.active_ir_runner_tasks
    assert list(record.ir_queue) == ["r1"]
    assert _decide_ir_release(record).allow is False
    record.protected_until_finalize = True
    assert _decide_ir_release(record).allow is True  # protected work is never held by the admission gate


def test_resume_pump_resumes_oldest_first() -> None:
    shard_cls, shard = _make_admission_shard(client=_FakeBudgetClient(), max_wait_s=1000.0)
    resumed: list[str] = []
    shard._maybe_start_next_ir_locked = lambda *, session_id, record: (resumed.append(session_id), True)[1]
    now = time.monotonic()
    r_old = _SessionRecord()
    r_old.admission_deferred = True
    r_old.admission_deferred_since = now - 5.0
    r_new = _SessionRecord()
    r_new.admission_deferred = True
    r_new.admission_deferred_since = now - 1.0
    shard._session_records = {"new": r_new, "old": r_old}
    shard._session_locks = {"new": asyncio.Lock(), "old": asyncio.Lock()}
    asyncio.run(shard_cls._resume_deferred_sessions_once(shard))
    assert resumed == ["old", "new"]  # oldest deferral first (aging)
    assert r_old.admission_deferred is False and r_new.admission_deferred is False


def test_resume_pump_forced_resume_respects_cap() -> None:
    starved = _FakeBudgetClient(
        hint={"degraded": False, "available": 0, "ceiling": 10**9, "reserved": 10**9, "epoch": 1}
    )
    shard_cls, shard = _make_admission_shard(client=starved, max_wait_s=0.0, forced_cap=2)
    resumed: list[str] = []
    shard._maybe_start_next_ir_locked = lambda *, session_id, record: (resumed.append(session_id), True)[1]
    now = time.monotonic()
    records = {}
    for i in range(3):
        rec = _SessionRecord()
        rec.admission_deferred = True
        rec.admission_deferred_since = now - 100.0
        records[f"s{i}"] = rec
    shard._session_records = records
    shard._session_locks = {key: asyncio.Lock() for key in records}
    asyncio.run(shard_cls._resume_deferred_sessions_once(shard))
    assert len(resumed) == 2  # forced-resume cap honored
    assert sum(1 for rec in records.values() if rec.admission_aged_resume) == 2
    assert sum(1 for rec in records.values() if rec.admission_deferred) == 1  # remainder stays deferred


def test_resume_pump_fails_open_when_capacity_signal_unavailable() -> None:
    shard_cls, shard = _make_admission_shard(client=_RaisingBudgetClient(), max_wait_s=1000.0)
    resumed: list[str] = []
    shard._maybe_start_next_ir_locked = lambda *, session_id, record: (resumed.append(session_id), True)[1]
    now = time.monotonic()
    rec = _SessionRecord()
    rec.admission_deferred = True
    rec.admission_deferred_since = now - 1.0
    shard._session_records = {"s": rec}
    shard._session_locks = {"s": asyncio.Lock()}
    asyncio.run(shard_cls._resume_deferred_sessions_once(shard))
    assert resumed == ["s"] and rec.admission_deferred is False  # degraded signal -> fail open (resume)
