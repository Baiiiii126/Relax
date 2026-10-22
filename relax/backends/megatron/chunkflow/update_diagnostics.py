# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in first-update samples; never call optimizer preparation a second
time."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def _samples(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result, device_rows = {}, {}
    for key, value in values.items():
        flat = value.detach().reshape(-1)
        if not flat.numel():
            continue
        positions = torch.arange(64, device=flat.device) * (flat.numel() - 1) // 63
        sample = flat[positions].float().clone()
        device_rows.setdefault(sample.device, []).append((key, sample))
    for rows in device_rows.values():
        for (key, _), sample in zip(rows, torch.stack([v for _, v in rows]).cpu(), strict=True):
            result[key] = sample.clone()
    return result


def _owners(optimizer: Any) -> list[Any]:
    children = getattr(optimizer, "chained_optimizers", None)
    return [leaf for child in children for leaf in _owners(child)] if children is not None else [optimizer]


@contextmanager
def trace_update(models: Any, optimizer: Any, rollout: int, step: int) -> Iterator[None]:
    directory = os.environ.get("RELAX_CHUNKFLOW_DIAGNOSTIC_DIR")
    selected = os.environ.get("RELAX_CHUNKFLOW_GRAD_ROLLOUTS", "").split(",")
    if not directory or os.environ.get("RELAX_CHUNKFLOW_UPDATE_TRACE") != "1" or str(rollout) not in selected:
        yield
        return
    parameters = {
        f"{i}.{name}": p for i, model in enumerate(models) for name, p in model.named_parameters() if p.requires_grad
    }
    names = {id(p): name for name, p in parameters.items()}
    data = {"before_model": _samples(parameters), "optimizers": [], "issues": []}
    originals = []
    all_shards = {}
    prepare = None
    had_prepare = False
    try:
        for owner in _owners(optimizer):
            leaf = getattr(owner, "optimizer", owner)
            mapping = getattr(leaf, "param_to_inner_param", None)
            if mapping is None:
                data["issues"].append(f"Unsupported optimizer mapping: {type(leaf).__name__}")
                continue
            mapped, ranges = {}, {}
            group_map = getattr(owner, "model_param_group_index_map", None)
            if group_map is not None:
                for p, (group, index) in group_map.items():
                    if id(p) not in names:
                        continue
                    shard = leaf.param_groups[group]["params"][index]
                    master = mapping.get(shard)
                    if master is not None:
                        name = names[id(p)]
                        mapped[name] = (shard, master)
                        interval = owner._get_model_param_range_map(p)["param"]
                        ranges[name] = (interval.start, interval.end, tuple(p.shape))
            else:
                mapped = {names[id(p)]: (p, master) for p, master in mapping.items() if id(p) in names}
            if not mapped:
                data["issues"].append(f"No named parameters mapped for {type(leaf).__name__}")
                continue
            all_shards.update({n: p for n, (p, _) in mapped.items()})
            original = leaf.step
            had_step = "step" in vars(leaf)
            originals.append((leaf, original, had_step))

            def measured(
                *args: Any,
                original: Any = original,
                leaf: Any = leaf,
                mapped: dict = mapped,
                ranges: dict = ranges,
                **kwargs: Any,
            ) -> Any:
                record = {
                    "class": type(leaf).__name__,
                    "ranges": ranges,
                    "before_master": _samples({n: m for n, (_, m) in mapped.items()}),
                }
                grads = {
                    n: getattr(p, "decoupled_grad", None) if getattr(p, "decoupled_grad", None) is not None else p.grad
                    for n, (p, _) in mapped.items()
                }
                record["clipped_grad"] = _samples({n: g for n, g in grads.items() if g is not None})
                result = original(*args, **kwargs)
                record["after_master"] = _samples({n: m for n, (_, m) in mapped.items()})
                for field in ("exp_avg", "exp_avg_sq"):
                    states = {n: leaf.state.get(p, {}).get(field) for n, (p, _) in mapped.items()}
                    record[field] = _samples({n: v for n, v in states.items() if isinstance(v, torch.Tensor)})
                record["param_groups"] = [
                    {key: group.get(key) for key in ("lr", "betas", "eps", "weight_decay")}
                    for group in leaf.param_groups
                ]
                data["optimizers"].append(record)
                return result

            leaf.step = measured
        prepare = getattr(optimizer, "prepare_grads", None)
        had_prepare = "prepare_grads" in vars(optimizer)
        if prepare is not None:

            def measured_prepare(*args: Any, **kwargs: Any) -> Any:
                result = prepare(*args, **kwargs)
                grads = {n: getattr(p, "decoupled_grad", None) for n, p in all_shards.items()}
                grads = {n: g if g is not None else all_shards[n].grad for n, g in grads.items()}
                data["prepared_grad"] = _samples({n: g for n, g in grads.items() if g is not None})
                return result

            optimizer.prepare_grads = measured_prepare
        yield
        # CPU offload may enqueue GPU copies; this synchronization is diagnostic-only.
        if any(p.is_cuda for p in parameters.values()):
            torch.cuda.synchronize()
        data["after_model"] = _samples(parameters)
        mapped_names = {n for record in data["optimizers"] for n in record["before_master"]}
        data["issues"].extend(f"No master snapshot: {name}" for name in parameters.keys() - mapped_names)
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank(group=torch.distributed.group.WORLD)
        torch.save(data, folder / f"update_r{rollout}_s{step}_rank{rank}.pt")
    finally:
        if prepare is not None:
            if had_prepare:
                optimizer.prepare_grads = prepare
            else:
                del optimizer.prepare_grads
        for leaf, original, had_step in originals:
            if had_step:
                leaf.step = original
            else:
                del leaf.step
