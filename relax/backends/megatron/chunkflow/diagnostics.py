# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import inspect
import json
import os
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_ACTIVE: ContextVar[Any] = ContextVar("gdn_diagnostic", default=None)


def _rank() -> int:
    return torch.distributed.get_rank(group=torch.distributed.group.WORLD)


def _identity(function: Any) -> dict:
    original = inspect.unwrap(function)
    try:
        source = inspect.getsourcefile(original)
    except TypeError:
        source = None
    code = getattr(function, "__code__", None)
    return {
        "callable": f"{getattr(original, '__module__', '')}.{getattr(original, '__qualname__', type(original).__name__)}",
        "source": source,
        "wrapper_source": getattr(code, "co_filename", None),
    }


def record_kernel(function: Any, arguments: dict, label: str) -> None:
    context = _ACTIVE.get()
    if context is None:
        return
    path, rollout_id, seen = context
    tensors = {k: list(v.shape) for k, v in arguments.items() if isinstance(v, torch.Tensor)}
    row = {
        "rollout": rollout_id,
        "layer": label,
        **_identity(function),
        "shapes": tensors,
        "initial_state": arguments.get("initial_state") is not None,
        "output_final_state": arguments.get("output_final_state", False),
    }
    key = json.dumps(row, sort_keys=True)
    if key not in seen:
        seen.add(key)
        with path.open("a") as stream:
            stream.write(key + "\n")
        if rollout_id == 0:
            logger.info("GDN diagnostic: %s", key)


@contextmanager
def trace_gdn_calls(models: Any, rollout_id: int) -> Iterator[None]:
    directory = os.environ.get("RELAX_CHUNKFLOW_DIAGNOSTIC_DIR")
    selected = os.environ.get("RELAX_CHUNKFLOW_GRAD_ROLLOUTS") or "0"
    if not directory or str(rollout_id) not in selected.split(","):
        yield
        return
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"backend_rank{_rank()}.jsonl"
    with ExitStack() as restore:
        if os.environ.get("RELAX_CHUNKFLOW_TAIL_TRACE") == "1":
            from relax.backends.megatron.chunkflow.tail_diagnostics import trace_tail

            restore.enter_context(trace_tail(models, folder, rollout_id, _rank()))
        elif os.environ.get("RELAX_CHUNKFLOW_CONV_TRACE") == "1":
            from relax.backends.megatron.chunkflow.conv_diagnostics import trace_convolutions

            restore.enter_context(trace_convolutions(models, folder, rollout_id, _rank()))
        elif os.environ.get("RELAX_CHUNKFLOW_LAYER_TRACE") == "1":
            from relax.backends.megatron.chunkflow.layer_diagnostics import trace_layers

            restore.enter_context(trace_layers(models, folder, rollout_id, _rank()))
        token = _ACTIVE.set((path, rollout_id, set()))
        restore.callback(_ACTIVE.reset, token)
        for index, model in enumerate(models):
            for name, module in model.named_modules():
                if not any(cls.__name__ == "GatedDeltaNet" for cls in type(module).__mro__):
                    continue
                function = module.gated_delta_rule

                def wrap(original: Any, label: str) -> Any:
                    signature = inspect.signature(original)

                    @wraps(original)
                    def measured(*args: Any, **kwargs: Any) -> Any:
                        bound = signature.bind(*args, **kwargs)
                        record_kernel(original, dict(bound.arguments), label + ":module_entry")
                        return original(*args, **kwargs)

                    return measured

                had_attribute = "gated_delta_rule" in vars(module)
                module.gated_delta_rule = wrap(function, f"{index}.{name}")
                if had_attribute:
                    restore.callback(setattr, module, "gated_delta_rule", function)
                else:
                    restore.callback(delattr, module, "gated_delta_rule")
        yield


def capture_gradients(models: Any, rollout_id: int, step_id: int) -> None:
    directory = os.environ.get("RELAX_CHUNKFLOW_DIAGNOSTIC_DIR")
    selected = os.environ.get("RELAX_CHUNKFLOW_GRAD_ROLLOUTS", "")
    if not directory or str(rollout_id) not in selected.split(","):
        return
    if os.environ.get("RELAX_CHUNKFLOW_CONV_TRACE") == "1":
        from relax.backends.megatron.chunkflow.conv_diagnostics import capture_final_gradients

        capture_final_gradients(models)
    rows = {}
    summaries = []
    for index, model in enumerate(models):
        groups = {}
        for name, module in model.named_modules():
            classes = {cls.__name__ for cls in type(module).__mro__}
            group = (
                "gdn"
                if "GatedDeltaNet" in classes
                else "moe"
                if "MoELayer" in classes
                else "attention"
                if any(cls.endswith("SelfAttention") for cls in classes)
                else None
            )
            if group:
                groups[name] = group
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = getattr(parameter, "main_grad", None)
            source = "main_grad"
            if grad is None:
                grad, source = parameter.grad, "grad"
            prefixes = [prefix for prefix in groups if not prefix or name.startswith(prefix + ".")]
            group = groups[max(prefixes, key=len)] if prefixes else "other"
            row = {
                "group": group,
                "shape": list(parameter.shape),
                "source": source,
                "missing": grad is None,
                "overwrite_main_grad": bool(getattr(parameter, "overwrite_main_grad", False)),
            }
            if grad is not None:
                flat = grad.detach().reshape(-1)
                count = min(64, flat.numel())
                positions = torch.arange(count, device=flat.device) * max(0, flat.numel() - 1) // max(1, count - 1)
                row.update(dtype=str(grad.dtype), sample_count=count)
                summaries.append(
                    torch.cat(
                        (
                            torch.linalg.vector_norm(flat, dtype=torch.float32).reshape(1),
                            torch.nn.functional.pad(flat[positions].float(), (0, 64 - count)),
                            torch.nn.functional.pad(
                                parameter.detach().reshape(-1)[positions].float(), (0, 64 - count)
                            ),
                        )
                    )
                )
            rows[f"{index}.{name}"] = row
    # Diagnostic-only transfer, batched to avoid one device synchronization per parameter.
    if summaries:
        values = iter(torch.stack(summaries).detach().cpu())
        for row in rows.values():
            if not row["missing"]:
                value = next(values)
                count = row["sample_count"]
                row.update(
                    norm=float(value[0]),
                    sample=value[1 : 1 + count].clone(),
                    weight_sample=value[65 : 65 + count].clone(),
                )
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(rows, folder / f"grad_r{rollout_id}_s{step_id}_rank{_rank()}.pt")
