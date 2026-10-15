# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Align ChunkFlow's theoretical step benefit against what a run actually got.

Four questions, one command, two log files, no GPU and no ClearML:

  1. theory  -- bubble ratio the schedule should save
  2. actual  -- actor-train ratio the step actually got
  3. align   -- actual / theory
  4. gap     -- excess seconds beyond the bubble-only prediction, with cost
               proxies to guide follow-up ablations

Nothing is recomputed here. The chunkflow arm already logs the whole theory
side per step -- ``train/chunkflow/num_microbatches``, ``num_units``, the two
``equal_unit_fill_drain_*_proxy`` values, and the cost multipliers -- inside its
``step <id>: {...}`` stdout line. Every arm logs ``perf/step_time`` in its
``perf <id>: {...}`` line. The default uses ``perf/actor_train_time`` to isolate
the scheduler; ``--metric step_time`` reports the end-to-end ratio instead. So
an A/B pair of ordinary runs is enough; the standalone theory calculator is
only needed to *predict* before running.

    python scripts/training/sft/tools/chunkflow_align.py \
        --baseline baseline.log --chunkflow chunkflow.log

Both arms must use the same data, seed and parallel layout, and ``DP=1`` (see
docs/draft/chunkflow-scheduling-benefit.md), otherwise the per-step pairing and
the ``M`` the telemetry reports are not the ones that executed.
"""

from __future__ import annotations

import argparse
import ast
import random
import re
from pathlib import Path
from typing import Any, Iterator


# Suffix-matched so the tool works for actor (``train/chunkflow/...``) and for
# any role-prefixed variant (``train/critic-chunkflow/...``) alike.
CHUNKFLOW_KEYS = {
    "M": "chunkflow/num_microbatches",
    "U": "chunkflow/num_units",
    "bubble_base": "chunkflow/equal_unit_fill_drain_baseline_proxy",
    "bubble_cf": "chunkflow/equal_unit_fill_drain_chunkflow_proxy",
    "frag": "chunkflow/fragmentation_amplification",
    "tok_recomp": "chunkflow/token_forward_recompute_multiplier",
    "attn_recomp": "chunkflow/attention_forward_recompute_multiplier",
    "launch": "chunkflow/launch_amplification",
    "logical_attn": "chunkflow/logical_attention_pairs",
    "kernel_rect": "chunkflow/kernel_rect_pairs",
    "p2p_messages": "chunkflow/p2p_messages",
    "p2p_rows": "chunkflow/p2p_row_transfers",
    "warmup_max": "chunkflow/warmup_max",
    "min_steady_fraction": "chunkflow/min_steady_unit_fraction",
}


def _pick(row: dict[str, Any], suffix: str) -> float | None:
    for key, value in row.items():
        if key.endswith(suffix):
            return float(value)
    return None


def _iter_dicts(text: str, label: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(index, log_dict)`` for every ``<label> <id>: {...}`` line.

    The dict is a Python repr (single quotes), so ``ast.literal_eval`` reads it
    back without needing the run's dependencies.
    """
    pattern = re.compile(rf"(?<![\w-]){label} (\d+): (\{{.*\}})")
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            yield int(match.group(1)), ast.literal_eval(match.group(2))
        except (ValueError, SyntaxError):
            continue


def _perf_rows(path: Path, metric: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for index, row in _iter_dicts(path.read_text(errors="replace"), "perf"):
        if _pick(row, f"perf/{metric}") is not None:
            rows[index] = row
    return rows


def _theory(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for index, row in _iter_dicts(path.read_text(errors="replace"), "step"):
        values = {name: _pick(row, suffix) for name, suffix in CHUNKFLOW_KEYS.items()}
        if values["bubble_base"] is not None and values["bubble_cf"]:
            out[index] = {k: v for k, v in values.items() if v is not None}
    return out


def _pair(baseline: Path, chunkflow: Path, metric: str) -> list[dict[str, float]]:
    perf_base = _perf_rows(baseline, metric)
    perf_cf = _perf_rows(chunkflow, metric)
    theory = _theory(chunkflow)

    rows = []
    for index in sorted(set(perf_base) & set(perf_cf) & set(theory)):
        t = theory[index]
        t_base = _pick(perf_base[index], f"perf/{metric}")
        t_cf = _pick(perf_cf[index], f"perf/{metric}")
        if not t_base or not t_cf or not t["bubble_cf"]:
            continue
        tokens_base = _pick(perf_base[index], "perf/actor_train_tokens")
        tokens_cf = _pick(perf_cf[index], "perf/actor_train_tokens")
        if tokens_base is None or tokens_cf is None:
            raise ValueError(f"step {index}: perf/actor_train_tokens missing; cannot verify paired inputs")
        if tokens_base != tokens_cf:
            raise ValueError(
                f"step {index}: paired arms processed different token counts "
                f"({tokens_base:g} != {tokens_cf:g}); inputs are not comparable"
            )
        theory_x = t["bubble_base"] / t["bubble_cf"]
        actual_x = t_base / t_cf
        ideal_cf_time = t_base / theory_x
        logical_attn = t.get("logical_attn", 0.0)
        rows.append(
            {
                "step": index,
                "M": t.get("M", 0.0),
                "U": t.get("U", 0.0),
                "theory_x": theory_x,
                "actual_x": actual_x,
                "align": actual_x / theory_x,
                "gap": theory_x / actual_x,
                "base_time": t_base,
                "cf_time": t_cf,
                "ideal_cf_time": ideal_cf_time,
                "excess_time": t_cf - ideal_cf_time,
                "tokens": tokens_base,
                "frag": t.get("frag", 1.0),
                "tok_recomp": t.get("tok_recomp", 1.0),
                "attn_recomp": t.get("attn_recomp", 1.0),
                "rect_attn": t.get("kernel_rect", 0.0) / logical_attn if logical_attn else 0.0,
                "launch": t.get("launch", 1.0),
                "p2p_messages": t.get("p2p_messages", 0.0),
                "p2p_rows": t.get("p2p_rows", 0.0),
                "warmup_max": t.get("warmup_max", 0.0),
                "min_steady_fraction": t.get("min_steady_fraction", 0.0),
            }
        )
    return rows


_HEAD = (
    "step",
    "M",
    "U",
    "theory_x",
    "actual_x",
    "align",
    "excess_time",
    "frag",
    "tok_recomp",
    "attn_recomp",
    "rect_attn",
    "launch",
)
_FMT = (
    "{:>4}",
    "{:>3}",
    "{:>4}",
    "{:>8.3f}",
    "{:>8.3f}",
    "{:>6.3f}",
    "{:>11.3f}",
    "{:>6.3f}",
    "{:>10.3f}",
    "{:>11.3f}",
    "{:>9.3f}",
    "{:>7.1f}",
)


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def _aggregate_speedup(rows: list[dict[str, float]]) -> float:
    """Ratio of total paired wall times, the primary A/B estimator."""
    return sum(row["base_time"] for row in rows) / sum(row["cf_time"] for row in rows)


def _aggregate_theory(rows: list[dict[str, float]]) -> float:
    """Compare total baseline time with the total bubble-only prediction."""
    return sum(row["base_time"] for row in rows) / sum(row["ideal_cf_time"] for row in rows)


def _bootstrap_ci(rows: list[dict[str, float]], samples: int, seed: int) -> tuple[float, float] | None:
    """Paired bootstrap confidence interval for the aggregate speedup."""
    if samples <= 0 or len(rows) < 2:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        estimates.append(_aggregate_speedup(draw))
    estimates.sort()
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline arm log (CHUNK_SIZE=0)")
    parser.add_argument("--chunkflow", type=Path, required=True, help="chunkflow arm log (CHUNK_SIZE>0)")
    parser.add_argument(
        "--metric",
        choices=["actor_train_time", "step_time"],
        default="actor_train_time",
        help="A/B timing metric; actor_train_time isolates the scheduler (default)",
    )
    parser.add_argument("--drop-first", type=int, default=0, help="steps to drop as warmup before averaging")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000, help="paired bootstrap resamples for 95%% CI")
    parser.add_argument("--seed", type=int, default=1234, help="bootstrap seed")
    args = parser.parse_args()
    if args.drop_first < 0:
        parser.error("--drop-first must be nonnegative")

    rows = _pair(args.baseline, args.chunkflow, args.metric)
    if args.drop_first:
        rows = rows[args.drop_first :]
    if not rows:
        raise SystemExit(
            "no paired steps: check that both logs come from the same data/seed/parallel layout "
            "and that the chunkflow arm actually ran with --use-chunkflow"
        )

    header = " ".join(f"{name:>{width}}" for name, width in zip(_HEAD, (4, 3, 4, 8, 8, 6, 11, 6, 10, 11, 9, 7)))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(fmt.format(row[key]) for fmt, key in zip(_FMT, _HEAD)))
    print("-" * len(header))
    mean_cells = [f"{'mean':>4}"] + [fmt.format(_mean(rows, key)) for fmt, key in zip(_FMT[1:], _HEAD[1:])]
    print(" ".join(mean_cells))

    theory_x = _aggregate_theory(rows)
    actual_x = _aggregate_speedup(rows)
    ci = _bootstrap_ci(rows, args.bootstrap_samples, args.seed)
    excess_time = sum(row["excess_time"] for row in rows)

    print()
    ci_text = "" if ci is None else f"   bootstrap_95%_ci=[{ci[0]:.3f}, {ci[1]:.3f}]"
    print(f"metric = perf/{args.metric}   paired_steps = {len(rows)}")
    print(f"aggregate_theory_x = {theory_x:.3f}   aggregate_actual_x = {actual_x:.3f}{ci_text}")
    print(f"alignment = {actual_x / theory_x:.3f}   gap = {theory_x / actual_x:.3f}")
    print(f"excess_vs_bubble_prediction = {excess_time:.3f}s total, {excess_time / len(rows):.3f}s/step")
    print(
        f"cost proxies (reported separately, not multiplied): frag {_mean(rows, 'frag'):.3f} | "
        f"tok_recomp {_mean(rows, 'tok_recomp'):.3f} | attn_recomp {_mean(rows, 'attn_recomp'):.3f} | "
        f"rect_attn {_mean(rows, 'rect_attn'):.3f} | launch {_mean(rows, 'launch'):.1f} | "
        f"p2p_messages {_mean(rows, 'p2p_messages'):.0f} | "
        f"warmup_max {_mean(rows, 'warmup_max'):.1f} | "
        f"min_steady_fraction {_mean(rows, 'min_steady_fraction'):.3f}"
    )
    print("These proxies suggest follow-up chunk-size ablations; one A/B pair cannot identify a causal cost.")


if __name__ == "__main__":
    main()
