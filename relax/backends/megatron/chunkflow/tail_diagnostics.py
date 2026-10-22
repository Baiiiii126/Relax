# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in first-backward probes at the model output and final layers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import torch

from relax.backends.megatron.chunkflow import layer_diagnostics as ld


_TRACE: ContextVar[Any] = ContextVar("chunkflow_tail_trace", default=None)


def _tap(
    value: torch.Tensor,
    layer: int,
    kind: str,
    layout: dict,
    rows: list[int],
    *,
    vocab: bool = False,
    separate_tp: bool = False,
    targets: torch.Tensor | None = None,
) -> None:
    trace = _TRACE.get()
    if trace is None or not torch.is_grad_enabled():
        return
    if value.shape[0] != len(rows):
        raise ValueError(f"Tail diagnostic row mismatch: {kind}: {value.shape[0]} != {len(rows)}")
    positions = [i for i, row in enumerate(rows) if row >= 0 and row in layout["selected"]]
    if not positions:
        return
    index = torch.tensor(positions, device=value.device, dtype=torch.long)
    features = target_indices = target_valid = None
    if vocab or separate_tp:
        from megatron.core import parallel_state as ps

        tp_rank = ps.get_tensor_model_parallel_rank()
        kind += f"_tp{tp_rank}"
    if vocab:
        if targets is not None:
            local = targets.index_select(0, index) - tp_rank * value.shape[-1]
            target_valid = ((local >= 0) & (local < value.shape[-1])).unsqueeze(1)
            target_indices = local.clamp(0, value.shape[-1] - 1).long().unsqueeze(1)
        count = min(64, value.shape[-1])
        features = torch.arange(count, device=value.device) * (value.shape[-1] - 1) // max(1, count - 1)
    canonical = [rows[i] for i in positions]

    def record(tensor: torch.Tensor, event: str) -> None:
        selected = tensor.detach().index_select(0, index).reshape(len(positions), -1)
        if features is not None:
            fixed = selected.index_select(1, features)
            if target_indices is not None:
                fixed = torch.cat((fixed, selected.gather(1, target_indices) * target_valid), dim=1)
            selected = fixed
        trace["events"].append(
            {
                "layer": layer,
                "kind": event,
                "parent": layout["parent"],
                "rows": canonical,
                "supervised": [row in layout.get("supervised", set()) for row in canonical],
                "values": selected.clone(),
            }
        )

    record(value, kind)
    if value.requires_grad:
        trace["handles"].append(value.register_hook(lambda grad: record(grad, kind + "_grad")))


def bind_loss_probe(function: Any, batch: dict) -> Any:
    trace = _TRACE.get()
    if trace is None:
        return function
    layout = ld._layout(batch)

    @wraps(function)
    def measured(*args: Any, **kwargs: Any) -> Any:
        active, coordinates = _TRACE.set(trace), ld._BATCH.set(layout)
        try:
            return function(*args, **kwargs)
        finally:
            ld._BATCH.reset(coordinates)
            _TRACE.reset(active)

    return measured


def tap_head(
    source: torch.Tensor, hidden: torch.Tensor, logits: torch.Tensor, log_probs: torch.Tensor, targets: torch.Tensor
) -> None:
    if _TRACE.get() is None:
        return
    layout = ld._BATCH.get()
    if layout is None or source.ndim != 3 or source.shape[0] != 1 or source.shape[1] != len(layout["rows"]):
        raise ValueError("Tail head probes require packed CP1 hidden states and canonical coordinates")
    # The response slicer must preserve a view of the original hidden rows.
    stride = source.stride(1)
    offset = hidden.storage_offset() - source.storage_offset()
    if (
        hidden.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        or hidden.stride(0) != stride
        or offset < 0
        or offset % stride
    ):
        raise ValueError("Response hidden states are no longer a row-aligned view; update diagnostic coordinates")
    start = offset // stride
    rows = layout["rows"][start : start + hidden.shape[0]]
    _tap(hidden, 40, "lm_head_input", layout, rows, separate_tp=True)
    _tap(logits, 40, "lm_head_logits", layout, rows, vocab=True, targets=targets)
    _tap(log_probs, 40, "log_probs", layout, rows)


def _boundary(value: torch.Tensor, layer: int, kind: str, *, separate_tp: bool = False) -> None:
    if _TRACE.get() is None or not torch.is_grad_enabled():
        return
    layout, _, _ = ld._selection(value)
    rows = layout["rows"]
    if value.shape[0] != len(rows):
        from megatron.core import parallel_state as ps

        start = ps.get_tensor_model_parallel_rank() * value.shape[0]
        rows = rows[start : start + value.shape[0]]
    _tap(value, layer, kind, layout, rows, separate_tp=separate_tp)


@contextmanager
def trace_tail(models: Any, folder: Path, rollout: int, rank: int) -> Iterator[None]:
    trace: dict = {"events": [], "handles": []}
    with ExitStack() as restore:
        active, coords, batch = _TRACE.set(trace), ld._TRACE.set([]), ld._BATCH.set(None)
        restore.callback(_TRACE.reset, active)
        restore.callback(ld._TRACE.reset, coords)
        restore.callback(ld._BATCH.reset, batch)

        def attach(module: Any, layer: int, kind: str, *, input_tp: bool = False, output_tp: bool = False) -> None:
            def pre(module: Any, args: tuple, kwargs: dict) -> None:
                value = args[0] if args else kwargs["hidden_states"]
                _boundary(value, layer, kind + "_input", separate_tp=input_tp)

            def post(module: Any, args: tuple, output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                _boundary(value, layer, kind + "_output", separate_tp=output_tp)

            restore.callback(module.register_forward_pre_hook(pre, with_kwargs=True).remove)
            restore.callback(module.register_forward_hook(post).remove)

        for model in models:
            for name, module in model.named_modules():
                if name.split(".")[-1] == "final_layernorm":
                    attach(module, 40, "final_norm")
                if any(cls.__name__ == "TransformerLayer" for cls in type(module).__mro__):
                    layer = module.layer_number
                    if layer in (39, 40):
                        attach(module, layer, "layer")
                    if layer == 40:
                        attach(module.self_attention, layer, "attention")
                        attach(module.mlp, layer, "moe")
                        attention = module.self_attention
                        for child_name in ("linear_proj", "linear_qkv", "q_layernorm", "k_layernorm"):
                            child = getattr(attention, child_name, None)
                            if child is not None:
                                attach(
                                    child,
                                    layer,
                                    child_name,
                                    input_tp=child_name != "linear_qkv",
                                    output_tp=child_name != "linear_proj",
                                )
                        core = getattr(attention, "core_attention", None)
                        if core is not None:

                            def core_pre(module: Any, args: tuple, kwargs: dict) -> None:
                                for index, name in enumerate(("query", "key", "value")):
                                    value = args[index] if len(args) > index else kwargs[name]
                                    _boundary(value, 40, "core_" + name, separate_tp=True)

                            def core_post(module: Any, args: tuple, output: Any) -> None:
                                _boundary(
                                    output[0] if isinstance(output, tuple) else output,
                                    40,
                                    "core_output",
                                    separate_tp=True,
                                )

                            restore.callback(core.register_forward_pre_hook(core_pre, with_kwargs=True).remove)
                            restore.callback(core.register_forward_hook(core_post).remove)
        try:
            yield
            folder.mkdir(parents=True, exist_ok=True)
            events = trace["events"]
            for event in events:
                event["values"] = event["values"].cpu()
            torch.save(events, folder / f"tail_r{rollout}_rank{rank}.pt")
        finally:
            for handle in trace["handles"]:
                handle.remove()
