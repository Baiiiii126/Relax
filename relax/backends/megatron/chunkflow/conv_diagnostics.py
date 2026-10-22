# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in probes at the actual GDN convolution kernel boundary."""

from __future__ import annotations

import inspect
import os
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import torch

from relax.backends.megatron.chunkflow import layer_diagnostics as ld


_TRACE: ContextVar[Any] = ContextVar("gdn_conv_trace", default=None)
_MODULE: ContextVar[Any] = ContextVar("gdn_conv_module", default=None)


def _channels(module: Any, width: int) -> list[int]:
    q, v = module.qk_dim_local_tp, module.v_dim_local_tp
    if q * 2 + v != width:
        raise ValueError("GDN convolution channel layout changed; expected local q/k/v sections")
    result, start = [], 0
    for size in (q, q, v):
        count = min(16, size)
        result.extend(start + i * (size - 1) // max(1, count - 1) for i in range(count))
        start += size
    return result


def call_convolution(kernel: Any, *args: Any, **kwargs: Any) -> Any:
    trace, context = _TRACE.get(), _MODULE.get()
    if trace is None or context is None or not torch.is_grad_enabled():
        return kernel(*args, **kwargs)
    module, packed = context
    bound = inspect.signature(kernel).bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    x, weight = arguments["x"], arguments["weight"]
    if x.ndim != 3 or x.shape[0] != 1 or weight.ndim != 2:
        raise ValueError("Convolution diagnostics require packed BTC inputs and [channels, width] weights")
    layout = ld._BATCH.get()
    if layout is None:
        raise RuntimeError("Convolution diagnostics require canonical batch coordinates")
    physical = layout["rows"]
    length, width = x.shape[1], weight.shape[1]
    if length < len(physical) and any(row >= 0 for row in physical[length:]):
        raise ValueError("Convolution cropping discarded real token coordinates")
    rows = physical[:length] + [-1] * max(0, length - len(physical))
    selected = set(layout["selected"])
    chunk_size = int(os.environ.get("RELAX_CHUNKFLOW_CONV_CHUNK_SIZE", "8192"))
    for start, end in layout["spans"]:
        for boundary in [*range(start, end, chunk_size), end]:
            selected.update(range(max(start, boundary - width), min(end, boundary + width + 1)))
    positions = [i for i, row in enumerate(rows) if row >= 0 and row in selected]
    channels = _channels(module, x.shape[-1])
    index = torch.tensor(positions, device=x.device, dtype=torch.long)
    channel_index = torch.tensor(channels, device=x.device, dtype=torch.long)
    ranges = []
    for row in rows:
        if row < 0:
            continue
        if ranges and ranges[-1][1] == row:
            ranges[-1][1] = row + 1
        else:
            ranges.append([row, row + 1])
    from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR

    unit = getattr(packed, UNIT_ATTR, None)
    record: dict = {
        "layer": int(module.layer_number),
        "parent": layout["parent"],
        "rows": [rows[i] for i in positions],
        "channels": channels,
        "physical_rows": length,
        "padding_rows": rows.count(-1),
        "ranges": ranges,
        "group": unit.chunks[0].group if unit is not None and unit.is_dependent else None,
        "group_index": unit.chunks[0].group_idx if unit is not None and unit.is_dependent else None,
        "values": {},
        "reference": None,
    }
    trace["calls"].append(record)
    values = record["values"]

    def sample(value: torch.Tensor) -> torch.Tensor:
        return value.detach()[0].index_select(0, index).index_select(1, channel_index).clone()

    def state_sample(value: torch.Tensor) -> torch.Tensor:
        return value.detach().index_select(1, channel_index).clone()

    def state_hook(field: str) -> Any:
        def capture(grad: torch.Tensor | None) -> None:
            if grad is not None:
                values[field] = state_sample(grad)

        return capture

    values["x"] = sample(x)
    initial = arguments.get("initial_state")
    if isinstance(initial, torch.Tensor):
        values["h0"] = state_sample(initial)
        if initial.requires_grad:
            trace["handles"].append(initial.register_hook(state_hook("dh0")))
    # Keep complete time axes for 48 channels in the previously worst GDN layer only.
    if module.layer_number == 15:
        cu = arguments.get("cu_seqlens_cpu")
        if cu is None:
            cu = arguments.get("cu_seqlens")
        cu = [0, length] if cu is None else [int(n) for n in cu.detach().cpu().tolist()]
        bias = arguments.get("bias")
        record["reference"] = {
            "x": x.detach().index_select(2, channel_index).clone(),
            "weight": weight.detach().index_select(0, channel_index).clone(),
            "bias": bias.detach().index_select(0, channel_index).clone() if bias is not None else None,
            "initial_state": state_sample(initial) if initial is not None else None,
            "cu": cu,
            "activation": arguments.get("activation"),
        }
    if x.requires_grad:
        trace["handles"].append(x.register_hook(lambda grad: values.__setitem__("dx", sample(grad))))
    if weight.requires_grad:
        trace["handles"].append(
            weight.register_hook(
                lambda grad: values.__setitem__("dw", grad.detach().index_select(0, channel_index).clone())
            )
        )
    output, final = kernel(*args, **kwargs)
    values["y"] = sample(output)
    if output.requires_grad:

        def upstream(grad: torch.Tensor) -> None:
            values["dy"] = sample(grad)
            if record["reference"] is not None:
                record["reference"]["dy"] = grad.detach().index_select(2, channel_index).clone()

        trace["handles"].append(output.register_hook(upstream))
    if isinstance(final, torch.Tensor):
        values["hT"] = state_sample(final)
        if final.requires_grad:
            trace["handles"].append(final.register_hook(state_hook("dhT")))
    return output, final


def capture_final_gradients(models: Any) -> None:
    trace = _TRACE.get()
    if trace is None:
        return
    for model in models:
        for module in model.modules():
            if not any(cls.__name__ == "GatedDeltaNet" for cls in type(module).__mro__):
                continue
            parameter = module.conv1d.weight
            grad = getattr(parameter, "main_grad", None)
            grad = parameter.grad if grad is None else grad
            trace["final"][int(module.layer_number)] = grad.detach().clone() if grad is not None else None


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    return value


@contextmanager
def trace_convolutions(models: Any, folder: Path, rollout: int, rank: int) -> Iterator[None]:
    from megatron.core.ssm import gated_delta_net

    trace = {"calls": [], "final": {}, "handles": []}
    with ExitStack() as restore:
        token, coords, batch = _TRACE.set(trace), ld._TRACE.set([]), ld._BATCH.set(None)
        restore.callback(_TRACE.reset, token)
        restore.callback(ld._TRACE.reset, coords)
        restore.callback(ld._BATCH.reset, batch)
        convolution = gated_delta_net.causal_conv1d
        if not hasattr(convolution, "_chunkflow_original"):

            @wraps(convolution)
            def measured(*args: Any, **kwargs: Any) -> Any:
                return call_convolution(convolution, *args, **kwargs)

            gated_delta_net.causal_conv1d = measured
            restore.callback(setattr, gated_delta_net, "causal_conv1d", convolution)
        for model in models:
            for module in model.modules():
                if not any(cls.__name__ == "GatedDeltaNet" for cls in type(module).__mro__):
                    continue
                original = module._forward_compute
                had_attribute = "_forward_compute" in vars(module)

                @wraps(original)
                def compute(*args: Any, original: Any = original, module: Any = module, **kwargs: Any) -> Any:
                    arguments = inspect.signature(original).bind(*args, **kwargs).arguments
                    packed = arguments.get("packed_seq_params")
                    layout = getattr(packed, "_chunkflow_conv_layout", None)
                    if layout is None:
                        layout = ld._BATCH.get()
                        if packed is not None:
                            packed._chunkflow_conv_layout = layout
                    coordinates = ld._BATCH.set(layout)
                    active = _TRACE.set(trace)
                    token = _MODULE.set((module, packed))
                    try:
                        return original(*args, **kwargs)
                    finally:
                        _MODULE.reset(token)
                        _TRACE.reset(active)
                        ld._BATCH.reset(coordinates)

                module._forward_compute = compute
                if had_attribute:
                    restore.callback(setattr, module, "_forward_compute", original)
                else:
                    restore.callback(delattr, module, "_forward_compute")
        try:
            yield
            trace["calls"] = [call for call in trace["calls"] if "dy" in call["values"]]
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(
                _cpu({k: v for k, v in trace.items() if k != "handles"}), folder / f"conv_r{rollout}_rank{rank}.pt"
            )
        finally:
            for handle in trace["handles"]:
                handle.remove()
