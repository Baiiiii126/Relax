# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from time import perf_counter, thread_time
from typing import Any, Iterator

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_ACTIVE: ContextVar[Any] = ContextVar("training_profile", default=None)


class _StepProfile:
    def __init__(self) -> None:
        self.host: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"calls": 0, "inclusive_s": 0.0, "thread_cpu_s": 0.0}
        )
        self.restore = ExitStack()
        self.patched: set[tuple[int, str]] = set()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        with torch.profiler.record_function(f"train.segment.{name}"):
            start = perf_counter()
            cpu_start = thread_time()
            try:
                yield
            finally:
                row = self.host[name]
                row["calls"] += 1
                row["inclusive_s"] += perf_counter() - start
                row["thread_cpu_s"] += thread_time() - cpu_start

    def wrap_call(self, function: Any, name: str) -> Any:
        self.host.setdefault(name, {"calls": 0, "inclusive_s": 0.0, "thread_cpu_s": 0.0})

        @wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            with self.phase(name):
                return function(*args, **kwargs)

        return measured

    def wrap(self, target: Any, attribute: str, name: str) -> None:
        function = getattr(target, attribute, None)
        key = (id(target), attribute)
        if not callable(function) or key in self.patched:
            return
        self.patched.add(key)
        previous = vars(target).get(attribute)
        had_attribute = attribute in vars(target)
        setattr(target, attribute, self.wrap_call(function, name))
        if had_attribute:
            self.restore.callback(setattr, target, attribute, previous)
        else:
            self.restore.callback(delattr, target, attribute)

    def install(self, args: Any, models: Any, optimizer: Any, config: Any) -> None:
        from megatron.core.pipeline_parallel import p2p_communication, schedules

        backward_modules = [schedules]
        if getattr(args, "use_chunkflow", False):
            from relax.backends.megatron.chunkflow import gdn, pipeline_schedules, unit_batch

            backward_modules.append(pipeline_schedules)
            self.wrap(pipeline_schedules, "pool_terminal_units", "chunkflow.tail_pool.plan")
            self.wrap(pipeline_schedules, "slice_pooled_batch", "chunkflow.batch.pooled")
            self.wrap(pipeline_schedules, "slice_parent_batch", "chunkflow.batch.single")
            self.wrap(unit_batch, "slice_parent_batch", "chunkflow.batch.pool_part")
            original_invoke = gdn._invoke_kernel

            @wraps(original_invoke)
            def invoke(kernel: Any, name: str, *a: Any, **kw: Any) -> Any:
                with self.phase(f"chunkflow.gdn.kernel.{name}"):
                    return original_invoke(kernel, name, *a, **kw)

            gdn._invoke_kernel = invoke
            self.restore.callback(setattr, gdn, "_invoke_kernel", original_invoke)
        for module in backward_modules:
            self.wrap(module, "backward_step", "backward")
        for method in (
            "recv_forward",
            "send_forward",
            "recv_backward",
            "send_backward",
            "send_forward_recv_backward",
            "send_backward_recv_forward",
        ):
            self.wrap(p2p_communication.P2PCommunicator, method, f"pipeline.{method}")
        self.wrap(config, "grad_sync_func", "grad_sync")
        self.wrap(config, "finalize_model_grads_func", "grad_finalize")
        self.wrap(optimizer, "step", "optimizer")
        for chained in getattr(optimizer, "chained_optimizers", [optimizer]):
            self.wrap(chained, "prepare_grads", "optimizer.prepare_grads")
            hybrid = getattr(chained, "optimizer", None)
            if hybrid is None or not hasattr(hybrid, "cpu_optimizers"):
                continue
            self.wrap(hybrid, "_set_sub_optimizer_grads", "offload.grads_d2h")
            for cpu in hybrid.cpu_optimizers:
                self.wrap(cpu, "step", "offload.cpu_step")
                for key, hook in list(cpu._optimizer_step_post_hooks.items()):
                    if getattr(hook, "__name__", "") == "param_copy_back_gpu_hook":
                        cpu._optimizer_step_post_hooks[key] = self.wrap_call(hook, "offload.params_h2d")
                        self.restore.callback(cpu._optimizer_step_post_hooks.__setitem__, key, hook)
        for model in models:
            for module in model.modules():
                names = {cls.__name__ for cls in type(module).__mro__}
                if "GatedDeltaNet" in names:
                    self.wrap(module, "_forward_compute", "model.gdn_compute")
                elif "MoELayer" in names:
                    self.wrap(module, "forward", "model.moe_forward")
                    for attribute in ("router", "experts", "shared_experts"):
                        self.wrap(getattr(module, attribute, None), "forward", f"model.moe.{attribute}")
                    dispatcher = getattr(module, "token_dispatcher", None)
                    for method in (
                        "dispatch_preprocess",
                        "token_dispatch",
                        "dispatch_postprocess",
                        "combine_preprocess",
                        "token_combine",
                        "combine_postprocess",
                        "_maybe_dtoh_and_synchronize",
                    ):
                        self.wrap(dispatcher, method, f"model.moe.{method}")
                elif any(name.endswith("SelfAttention") for name in names):
                    self.wrap(module, "forward", "model.attention_forward")


@contextmanager
def train_phase(name: str) -> Iterator[None]:
    profile = _ACTIVE.get()
    if profile is None:
        yield
    else:
        with profile.phase(name):
            yield


def profile_forward_step(function: Any) -> Any:
    profile = _ACTIVE.get()
    if profile is None:
        return function

    @wraps(function)
    def forward(*args: Any, **kwargs: Any) -> Any:
        with profile.phase("forward"):
            output, loss_function = function(*args, **kwargs)
        return output, profile.wrap_call(loss_function, "loss")

    return forward


def _host_resources() -> dict[str, Any]:
    result: dict[str, Any] = {"process_cpu_s": sum(os.times()[:2]), "cgroup_visible_root": {}}
    for name in (
        "cpu.stat",
        "cpu.max",
        "cpu.pressure",
        "memory.current",
        "memory.max",
        "memory.events",
        "memory.pressure",
    ):
        try:
            result["cgroup_visible_root"][name] = (Path("/sys/fs/cgroup") / name).read_text().strip()
        except OSError:
            continue
    return result


@contextmanager
def _record_allocator_memory(args: Any, rollout_id: int, step_id: int) -> Iterator[None]:
    """Record allocator peaks without enabling the CUDA profiler or adding a
    GPU sync."""
    directory = os.environ.get("RELAX_TRAIN_MEMORY_DIR")
    if not directory or not torch.cuda.is_available():
        yield
        return
    from relax.utils.profile_utils import _get_rank_tag

    device = torch.cuda.current_device()
    before = {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "peak_allocated_since_previous_reset": torch.cuda.max_memory_allocated(device),
        "peak_reserved_since_previous_reset": torch.cuda.max_memory_reserved(device),
    }
    torch.cuda.reset_peak_memory_stats(device)
    yield
    rank = _get_rank_tag()
    row = {
        "rollout_id": rollout_id,
        "step_id": step_id,
        "rank": rank,
        "arm": "chunkflow" if getattr(args, "use_chunkflow", False) else "baseline",
        "moe_metadata_mode": os.environ.get("RELAX_MOE_METADATA_MODE", "off"),
        "before": before,
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "end_allocated": torch.cuda.memory_allocated(device),
        "end_reserved": torch.cuda.memory_reserved(device),
    }
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"memory_{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


@contextmanager
def profile_train_step(
    args: Any, rollout_id: int, step_id: int, models: Any, optimizer: Any, config: Any
) -> Iterator[None]:
    if not (
        getattr(args, "use_pytorch_profiler", False)
        and "train_overall" in getattr(args, "profile_target", [])
        and args.profile_step_start <= rollout_id <= args.profile_step_end
    ):
        with _record_allocator_memory(args, rollout_id, step_id):
            yield
        return
    from relax.utils.profile_utils import _get_rank_tag, _get_train_trace_dir

    profile = _StepProfile()
    cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    token = _ACTIVE.set(profile)
    try:
        with profile.restore:
            profile.install(args, models, optimizer, config)
            resources_before = _host_resources()
            with profile.phase("step"):
                yield
            resources_after = _host_resources()
    finally:
        _ACTIVE.reset(token)
    rank = _get_rank_tag()
    folder = Path(_get_train_trace_dir(args))
    folder.mkdir(parents=True, exist_ok=True)
    row = {
        "rollout_id": rollout_id,
        "step_id": step_id,
        "arm": "chunkflow" if getattr(args, "use_chunkflow", False) else "baseline",
        "tail_pool_enabled": bool(getattr(args, "use_chunkflow", False))
        and os.environ.get("RELAX_CHUNKFLOW_TAIL_POOL") == "1",
        "rank": rank,
        "host_inclusive": profile.host,
        "host_resources_before": resources_before,
        "host_resources_after": resources_after,
        "cpu_threads": {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
            "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        },
    }
    if cuda_device is not None:
        row["cuda_memory_bytes"] = {
            "peak_allocated": torch.cuda.max_memory_allocated(cuda_device),
            "peak_reserved": torch.cuda.max_memory_reserved(cuda_device),
            "end_allocated": torch.cuda.memory_allocated(cuda_device),
            "end_reserved": torch.cuda.memory_reserved(cuda_device),
        }
    # CPU intervals include waits and nested calls; GPU execution is recorded in the trace.
    with (folder / f"segments_{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    logger.info("Training segments rollout=%d step=%d %s: %s", rollout_id, step_id, rank, json.dumps(profile.host))
