# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_ACTIVE: ContextVar[Any] = ContextVar("training_profile", default=None)


class _StepProfile:
    def __init__(self) -> None:
        self.host: dict[str, dict[str, float | int]] = defaultdict(lambda: {"calls": 0, "inclusive_s": 0.0})
        self.restore = ExitStack()
        self.patched: set[tuple[int, str]] = set()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        with torch.profiler.record_function(f"train.segment.{name}"):
            start = perf_counter()
            try:
                yield
            finally:
                row = self.host[name]
                row["calls"] += 1
                row["inclusive_s"] += perf_counter() - start

    def wrap_call(self, function: Any, name: str) -> Any:
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
            from relax.backends.megatron.chunkflow import pipeline_schedules

            backward_modules.append(pipeline_schedules)
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


@contextmanager
def profile_train_step(
    args: Any, rollout_id: int, step_id: int, models: Any, optimizer: Any, config: Any
) -> Iterator[None]:
    if not (
        getattr(args, "use_pytorch_profiler", False)
        and "train_overall" in getattr(args, "profile_target", [])
        and args.profile_step_start <= rollout_id <= args.profile_step_end
    ):
        yield
        return
    from relax.utils.profile_utils import _get_rank_tag, _get_train_trace_dir

    profile = _StepProfile()
    token = _ACTIVE.set(profile)
    try:
        with profile.restore:
            profile.install(args, models, optimizer, config)
            with profile.phase("step"):
                yield
    finally:
        _ACTIVE.reset(token)
    rank = _get_rank_tag()
    folder = Path(_get_train_trace_dir(args))
    folder.mkdir(parents=True, exist_ok=True)
    row = {
        "rollout_id": rollout_id,
        "step_id": step_id,
        "arm": "chunkflow" if getattr(args, "use_chunkflow", False) else "baseline",
        "rank": rank,
        "host_inclusive": profile.host,
    }
    # CPU intervals include waits and nested calls; GPU execution is recorded in the trace.
    with (folder / f"segments_{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    logger.info("Training segments rollout=%d step=%d %s: %s", rollout_id, step_id, rank, json.dumps(profile.host))
