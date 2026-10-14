# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Align ChunkFlow's theoretical step benefit against what a run actually got.

Four questions, one command, two log files, no GPU and no ClearML:

  1. theory  -- bubble ratio the schedule should save
  2. actual  -- wall-clock ratio the step actually got
  3. align   -- actual / theory
  4. gap     -- which logged cost multiplier accounts for the shortfall

Nothing is recomputed here. The chunkflow arm already logs the whole theory
side per step -- ``train/chunkflow/num_microbatches``, ``num_units``, the two
``equal_unit_fill_drain_*_proxy`` values, and the cost multipliers -- inside its
``step <id>: {...}`` stdout line. Every arm logs ``perf/step_time`` in its
``perf <id>: {...}`` line. So an A/B pair of ordinary runs is enough; the
standalone theory calculator is only needed to *predict* before running.

    python scripts/training/sft/tools/chunkflow_align.py \
        --baseline baseline.log --chunkflow chunkflow.log

Both arms must use the same data, seed and parallel layout, and ``DP=1`` (see
docs/draft/chunkflow-scheduling-benefit.md), otherwise the per-step pairing and
the ``M`` the telemetry reports are not the ones that executed.
"""

from __future__ import annotations

import argparse
import ast
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
}
# Not a multiplier (units per micro-batch), listed for context only.
NON_MULTIPLIER = {"launch"}
MULTIPLIERS = ("frag", "tok_recomp", "attn_recomp")


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


def _step_times(path: Path) -> dict[int, float]:
    times: dict[int, float] = {}
    for index, row in _iter_dicts(path.read_text(errors="replace"), "perf"):
        value = _pick(row, "perf/step_time")
        if value is not None:
            times[index] = value
    return times


def _theory(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for index, row in _iter_dicts(path.read_text(errors="replace"), "step"):
        values = {name: _pick(row, suffix) for name, suffix in CHUNKFLOW_KEYS.items()}
        if values["bubble_base"] is not None and values["bubble_cf"]:
            out[index] = {k: v for k, v in values.items() if v is not None}
    return out


def _pair(baseline: Path, chunkflow: Path) -> list[dict[str, float]]:
    t_base = _step_times(baseline)
    t_cf = _step_times(chunkflow)
    theory = _theory(chunkflow)

    rows = []
    for index in sorted(set(t_base) & set(t_cf) & set(theory)):
        t = theory[index]
        if not t_base[index] or not t_cf[index] or not t["bubble_cf"]:
            continue
        theory_x = t["bubble_base"] / t["bubble_cf"]
        actual_x = t_base[index] / t_cf[index]
        rows.append(
            {
                "step": index,
                "M": t.get("M", 0.0),
                "U": t.get("U", 0.0),
                "theory_x": theory_x,
                "actual_x": actual_x,
                "align": actual_x / theory_x,
                "gap": theory_x / actual_x,
                **{name: t.get(name, 1.0) for name in MULTIPLIERS},
                "launch": t.get("launch", 1.0),
            }
        )
    return rows


_HEAD = ("step", "M", "U", "theory_x", "actual_x", "align", "gap", "frag", "tok_recomp", "attn_recomp", "launch")
_FMT = ("{:>4}", "{:>3}", "{:>4}", "{:>8.3f}", "{:>8.3f}", "{:>6.3f}", "{:>6.3f}",
        "{:>6.3f}", "{:>10.3f}", "{:>11.3f}", "{:>7.1f}")


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline arm log (CHUNK_SIZE=0)")
    parser.add_argument("--chunkflow", type=Path, required=True, help="chunkflow arm log (CHUNK_SIZE>0)")
    parser.add_argument("--drop-first", type=int, default=0, help="steps to drop as warmup before averaging")
    args = parser.parse_args()

    rows = _pair(args.baseline, args.chunkflow)
    if args.drop_first:
        rows = rows[args.drop_first:]
    if not rows:
        raise SystemExit(
            "no paired steps: check that both logs come from the same data/seed/parallel layout "
            "and that the chunkflow arm actually ran with --use-chunkflow"
        )

    header = " ".join(f"{name:>{width}}" for name, width in zip(_HEAD, (4, 3, 4, 8, 8, 6, 6, 6, 10, 11, 7)))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(fmt.format(row[key]) for fmt, key in zip(_FMT, _HEAD)))
    print("-" * len(header))
    mean_cells = [f"{'mean':>4}"] + [fmt.format(_mean(rows, key)) for fmt, key in zip(_FMT[1:], _HEAD[1:])]
    print(" ".join(mean_cells))

    theory_x = _mean(rows, "theory_x")
    actual_x = _mean(rows, "actual_x")
    gap = theory_x / actual_x
    product = 1.0
    for name in MULTIPLIERS:
        product *= _mean(rows, name)

    print()
    print(f"theory_x = {theory_x:.3f}   actual_x = {actual_x:.3f}   align = {actual_x / theory_x:.3f}")
    print(f"gap = {gap:.3f}  (the schedule promised {theory_x:.2f}x, the step delivered {actual_x:.2f}x)")
    print(f"logged multipliers: frag {_mean(rows, 'frag'):.3f} | tok_recomp {_mean(rows, 'tok_recomp'):.3f} | "
          f"attn_recomp {_mean(rows, 'attn_recomp'):.3f}  ->  product {product:.3f}")
    print(f"residual after multipliers = {gap / product:.3f}"
          "  (≈1 means the logged costs explain the gap; >1 points at unmodeled cost, "
          "primarily launch_amplification and p2p)")


if __name__ == "__main__":
    main()
