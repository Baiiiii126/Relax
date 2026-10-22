# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

import torch


_TRACE: ContextVar[Any] = ContextVar("chunkflow_layer_trace", default=None)
_BATCH: ContextVar[Any] = ContextVar("chunkflow_trace_batch", default=None)
_LAYOUT = "__chunkflow_trace_layout__"


def _layout(batch: dict) -> dict:
    if _LAYOUT not in batch:
        tokens = batch["tokens"].detach().reshape(-1)
        lengths = [int(n) for n in batch["total_lengths"]]
        cu = getattr(batch.get("packed_seq_params"), "cu_seqlens_q_cpu", None)
        if cu is None or len(cu) < len(lengths) + 1:
            cu = [0]
            for length in lengths:
                cu.append(cu[-1] + length)
        spans = [
            (int(start), min(int(start) + length, int(end), tokens.numel()))
            for start, end, length in zip(cu[: len(lengths)], cu[1 : len(lengths) + 1], lengths, strict=True)
        ]
        valid_rows = [row for start, end in spans for row in range(start, end)]
        physical = [-1] * tokens.numel()
        for row in valid_rows:
            physical[row] = row
        identity = hashlib.sha256(tokens[valid_rows].cpu().numpy().tobytes())
        identity.update(str((tuple(lengths), tuple(int(n) for n in cu))).encode())
        batch[_LAYOUT] = {
            "parent": identity.hexdigest(),
            "rows": physical,
            "spans": spans,
            "selected": {valid_rows[i * (len(valid_rows) - 1) // 31] for i in range(32)} if valid_rows else set(),
        }
    return batch[_LAYOUT]


def unit_layout(parent: dict, batch: dict, spans: list[tuple[int, int]]) -> None:
    if _TRACE.get() is None:
        return
    layout = _layout(parent)
    rows = [row for start, end in spans for row in layout["rows"][start:end]]
    rows.extend([-1] * (batch["tokens"].numel() - len(rows)))
    batch[_LAYOUT] = {**layout, "rows": rows}


def set_batch(batch: dict) -> None:
    if _TRACE.get() is not None:
        _BATCH.set(_layout(batch))


def _selection(tensor: torch.Tensor, *, all_tokens: bool = False) -> tuple[dict, list[int], list[int]]:
    from megatron.core import parallel_state as ps

    layout = _BATCH.get()
    if layout is None:
        raise RuntimeError("Layer diagnostics require batch token coordinates")
    rows = layout["rows"]
    length = tensor.shape[0]
    if length != len(rows):
        tp = ps.get_tensor_model_parallel_world_size()
        if length * tp != len(rows):
            raise RuntimeError(f"Unrecognized layer/SP layout: tensor rows={length}, token rows={len(rows)}")
        rank = ps.get_tensor_model_parallel_rank()
        rows = rows[rank * length : (rank + 1) * length]
    indices = [i for i, row in enumerate(rows) if row >= 0 and (all_tokens or row in layout["selected"])]
    return layout, indices, [rows[i] for i in indices]


def _tap(tensor: torch.Tensor, layer: int, kind: str, *, backward: bool = False, full: bool = False) -> None:
    trace = _TRACE.get()
    layout, positions, rows = _selection(tensor, all_tokens=layer == 1 or kind == "router_map")
    if not positions:
        return
    indices = torch.tensor(positions, device=tensor.device)
    parent = layout["parent"]

    def record(value: torch.Tensor, event: str) -> None:
        selected = value.detach().index_select(0, indices).reshape(len(rows), -1)
        if not full and layer != 1:
            features = torch.arange(min(64, selected.shape[1]), device=value.device)
            features = features * max(0, selected.shape[1] - 1) // max(1, features.numel() - 1)
            selected = selected.index_select(1, features)
        trace.append(
            {
                "layer": layer,
                "kind": event,
                "parent": parent,
                "rows": rows,
                "values": selected.clone(),
            }
        )

    record(tensor, kind)
    if backward and tensor.requires_grad:

        def hook(grad: torch.Tensor) -> None:
            record(grad, kind + "_grad")

        tensor.register_hook(hook)


@contextmanager
def trace_layers(models: Any, folder: Path, rollout: int, rank: int) -> Iterator[None]:
    with ExitStack() as restore:
        events: list[dict] = []
        token = _TRACE.set(events)
        batch_token = _BATCH.set(None)
        restore.callback(_TRACE.reset, token)
        restore.callback(_BATCH.reset, batch_token)
        for model in models:
            for module in model.modules():
                classes = {cls.__name__ for cls in type(module).__mro__}
                if "TransformerLayer" not in classes:
                    continue
                layer = module.layer_number

                def pre(module: Any, args: tuple, kwargs: dict, layer: int = layer) -> None:
                    value = args[0] if args else kwargs["hidden_states"]
                    _tap(value, layer, "input", backward=True)

                def post(module: Any, args: tuple, output: Any, layer: int = layer) -> None:
                    value = output[0] if isinstance(output, tuple) else output
                    _tap(value, layer, "output", backward=True)

                restore.callback(module.register_forward_pre_hook(pre, with_kwargs=True).remove)
                restore.callback(module.register_forward_hook(post).remove)
                if layer == 1:
                    for name in ("self_attention", "mlp"):
                        child = getattr(module, name, None)
                        if child is None:
                            continue

                        def component_pre(module: Any, args: tuple, kwargs: dict, name: str = name) -> None:
                            value = args[0] if args else kwargs["hidden_states"]
                            _tap(value, 1, name + "_input", backward=True)

                        def component_post(module: Any, args: tuple, output: Any, name: str = name) -> None:
                            value = output[0] if isinstance(output, tuple) else output
                            _tap(value, 1, name + "_output", backward=True)

                        restore.callback(child.register_forward_pre_hook(component_pre, with_kwargs=True).remove)
                        restore.callback(child.register_forward_hook(component_post).remove)
                for child in module.modules():
                    if not any(cls.__name__ == "TopKRouter" for cls in type(child).__mro__):
                        continue

                    def router_pre(module: Any, args: tuple, kwargs: dict, layer: int = layer) -> None:
                        value = args[0] if args else kwargs["input"]
                        _tap(value, layer, "router_input", full=True)

                    def router_post(module: Any, args: tuple, output: Any, layer: int = layer) -> None:
                        probs, routes = output
                        _tap(probs, layer, "router_probs", full=True)
                        _tap(routes, layer, "router_map", full=True)

                    restore.callback(child.register_forward_pre_hook(router_pre, with_kwargs=True).remove)
                    restore.callback(child.register_forward_hook(router_post).remove)
                    if layer == 1 and hasattr(child, "gating"):
                        gating = child.gating
                        had_gating = "gating" in vars(child)

                        def measured_gating(*args: Any, original: Any = gating, **kwargs: Any) -> torch.Tensor:
                            logits = original(*args, **kwargs)
                            _tap(logits, 1, "router_logits", full=True)
                            return logits

                        child.gating = measured_gating
                        if had_gating:
                            restore.callback(setattr, child, "gating", gating)
                        else:
                            restore.callback(delattr, child, "gating")
        yield
        folder.mkdir(parents=True, exist_ok=True)
        for event in events:
            event["values"] = event["values"].cpu()
        torch.save(events, folder / f"layers_r{rollout}_rank{rank}.pt")
