# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Align ChunkFlow's theoretical step benefit against what a run actually got.

Four questions, one command, two log files, no GPU and no ClearML:

  1. theory  -- declared timeline model or legacy bubble proxy
  2. actual  -- actor-train ratio the step actually got
  3. align   -- actual / theory
  4. gap     -- excess seconds beyond the bubble-only prediction, with cost
               proxies to guide follow-up ablations

Nothing is recomputed here. The chunkflow arm already logs the whole theory
side per step -- ``train/chunkflow/num_microbatches``, ``num_units``, the two
``equal_unit_fill_drain_*_proxy`` values, and the cost multipliers -- inside its
``step <id>: {...}`` stdout line. Every arm logs ``perf/step_time`` in its
``perf <id>: {...}`` line. The default uses ``perf/actor_train_time`` to isolate
the training path, including optimizer work; ``--metric step_time`` also
includes producer wait. So
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
import csv
import json
import math
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
    "schedule_x": "chunkflow/schedule_model_speedup",
    "analysis_time_s": "chunkflow/schedule_model_analysis_time_s",
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
    pattern = re.compile(rf"(?<![\w-]){label}\s+(\d+):\s*(\{{.*\}})")
    for line in text.splitlines():
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        match = pattern.search(line)
        if not match:
            continue
        try:
            yield int(match.group(1)), _metric_dict(match.group(2))
        except (ValueError, SyntaxError):
            continue


def _metric_dict(text: str) -> dict[str, Any]:
    """Read numeric log values without executing calls in a Python repr."""

    class Scalars(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in ("nan", "inf"):
                return ast.Constant(float(node.id))
            return node

        def visit_Call(self, node: ast.Call) -> ast.AST:
            name = ast.unparse(node.func)
            allowed = {
                "np.float64",
                "np.float32",
                "numpy.float64",
                "numpy.float32",
                "np.int64",
                "np.int32",
                "numpy.int64",
                "numpy.int32",
                "tensor",
                "torch.tensor",
            }
            if name not in allowed or len(node.args) != 1:
                raise ValueError("unsupported scalar representation")
            return self.visit(node.args[0])

    tree = ast.parse(text, mode="eval").body
    if not isinstance(tree, ast.Dict):
        raise ValueError("expected a log dictionary")
    result = {}
    for key, value in zip(tree.keys, tree.values):
        try:
            result[ast.literal_eval(key)] = ast.literal_eval(Scalars().visit(value))
        except (ValueError, TypeError):
            continue
    return result


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


def _pair(baseline: Path, chunkflow: Path, metric: str, theory_source: str = "proxy") -> list[dict[str, Any]]:
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
        if theory_source == "schedule" and "schedule_x" not in t:
            raise ValueError(
                f"step {index}: schedule prediction missing; use RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=1 "
                "for new runs, or --theory-source proxy for the existing bubble proxy"
            )
        theory_x = t["schedule_x"] if theory_source == "schedule" else t["bubble_base"] / t["bubble_cf"]
        if not all(math.isfinite(v) and v > 0 for v in (theory_x, t_base, t_cf)):
            raise ValueError(f"step {index}: nonfinite or nonpositive timing/prediction")
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
                "theory_gain_pct": 100 * (theory_x - 1),
                "actual_gain_pct": 100 * (actual_x - 1),
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
                "analysis_time_s": t.get("analysis_time_s", 0.0),
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


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    if xx <= 1e-24 or yy <= 1e-24:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        end = i + 1
        while end < len(order) and values[order[end]] == values[order[i]]:
            end += 1
        for index in order[i:end]:
            ranks[index] = (i + end - 1) / 2
        i = end
    return ranks


def _attribution(rows: list[dict[str, Any]], tolerance_pct: float = 0.0) -> dict[str, Any]:
    xs = [row["theory_gain_pct"] for row in rows]
    ys = [row["actual_gain_pct"] for row in rows]
    table = {
        "actual_positive": {"predicted_positive": 0, "predicted_negative": 0},
        "actual_negative": {"predicted_positive": 0, "predicted_negative": 0},
    }
    neutral = 0
    for x, y in zip(xs, ys):
        if abs(x) <= tolerance_pct or abs(y) <= tolerance_pct:
            neutral += 1
            continue
        table["actual_positive" if y > 0 else "actual_negative"][
            "predicted_positive" if x > 0 else "predicted_negative"
        ] += 1
    directional = len(rows) - neutral
    agree = table["actual_positive"]["predicted_positive"] + table["actual_negative"]["predicted_negative"]
    return {
        "paired_steps": len(rows),
        "tolerance_pct": tolerance_pct,
        "counts": table,
        "neutral_steps": neutral,
        "direction_agreement": agree / directional if directional else None,
        "pearson": _pearson(xs, ys),
        "spearman": _pearson(_ranks(xs), _ranks(ys)),
        "mean_absolute_prediction_error_pp": sum(abs(x - y) for x, y in zip(xs, ys)) / len(rows),
    }


def _print_attribution(name: str, result: dict[str, Any]) -> None:
    def value(key: str) -> str:
        if result[key] is None:
            return "undefined (too few or constant predictions/measurements)"
        return f"{result[key]:.4f}"

    print(f"\nattribution: {name}, steps={result['paired_steps']}, neutral={result['neutral_steps']}")
    print("| 计数 | scheduling 预测正收益 | scheduling 预测负收益 |")
    print("| --- | ---: | ---: |")
    for actual, label in (("actual_positive", "实际正收益"), ("actual_negative", "实际负收益")):
        counts = result["counts"][actual]
        print(f"| {label} | {counts['predicted_positive']} | {counts['predicted_negative']} |")
    print(f"direction_agreement={value('direction_agreement')}")
    print(f"Pearson={value('pearson')} | Spearman={value('spearman')}")
    print(f"mean_absolute_prediction_error={result['mean_absolute_prediction_error_pp']:.3f} percentage points")


def _markdown_report(results: dict[str, Any]) -> str:
    source = "气泡公式代理（非调度时间轴）" if results["theory_source"] == "proxy" else "声明成本假设的调度时间轴模型"
    parts = [f"预测来源：{source}。每组去掉前 {results['drop_first_per_case']} 个配对 step。", ""]
    sections = [("全部 A/B 对比汇总", results["pooled"])]
    for case in results["cases"]:
        name = Path(case["chunkflow"]).name.replace("|", r"\|")
        sections.append((f"case {case['case']}：{name}", results["per_case"][str(case["case"])]))
    for name, result in sections:
        parts.extend(
            [f"### {name}", "", "| 计数 | scheduling 预测正收益 | scheduling 预测负收益 |", "| --- | ---: | ---: |"]
        )
        for actual, label in (("actual_positive", "实际正收益"), ("actual_negative", "实际负收益")):
            counts = result["counts"][actual]
            parts.append(f"| {label} | {counts['predicted_positive']} | {counts['predicted_negative']} |")
        parts.extend(
            [
                "",
                f"配对数：{result['paired_steps']}；中性、不进入四格表：{result['neutral_steps']}。",
                f"Pearson：{result['pearson']}；Spearman：{result['spearman']}。",
                "",
            ]
        )
    parts.append("正收益表示 speedup > 1，负收益表示 speedup < 1；阈值范围内的点计为中性。")
    parts.append("不同 chunk 配置复用 baseline，汇总计数表示 A/B 比较次数，不代表独立样本数。")
    return "\n".join(parts) + "\n"


def _tail_sweep_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Select six comparisons from the default 8-GPU tail sweep, showing all
    paths."""
    pattern = re.compile(
        r"^cfbenefit-gbs(?P<gbs>\d+)-cap32768-chunk(?P<chunk>\d+)-s(?P<steps>\d+)-gpu8-"
        r"(?P<stamp>\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})\.log$"
    )
    groups: dict[tuple[int, int, int], list[tuple[str, Path]]] = {}
    for path in directory.glob("cfbenefit-*.log"):
        match = pattern.fullmatch(path.name)
        if match:
            key = tuple(int(match[k]) for k in ("gbs", "steps", "chunk"))
            groups.setdefault(key, []).append((match["stamp"], path))
    pairs = []
    for gbs, steps, chunks in ((1, 35, (512, 1024, 2048, 4096)), (4, 20, (1024,)), (16, 10, (1024,))):
        baselines = groups.get((gbs, steps, 0), [])
        if not baselines:
            raise ValueError(f"missing default tail baseline: GBS={gbs}, steps={steps}, GPU=8, directory={directory}")
        baseline_stamp, baseline = max(baselines)
        for chunk in chunks:
            candidates = [
                (stamp, path) for stamp, path in groups.get((gbs, steps, chunk), []) if stamp >= baseline_stamp
            ]
            if not candidates:
                raise ValueError(
                    f"missing CF chunk={chunk} after selected baseline {baseline}; "
                    "use explicit --baseline/--chunkflow for other matrices or earlier runs"
                )
            pairs.append((baseline, max(candidates)[1]))
    return pairs


def _scatter_svg(rows: list[dict[str, Any]]) -> str:
    xs = [row["theory_gain_pct"] for row in rows]
    ys = [row["actual_gain_pct"] for row in rows]
    low, high = min(0.0, *xs, *ys), max(0.0, *xs, *ys)
    margin = max((high - low) * 0.08, 1.0)
    low, high = low - margin, high + margin

    def px(x: float) -> float:
        return 80 + 600 * (x - low) / (high - low)

    def py(y: float) -> float:
        return 660 - 580 * (y - low) / (high - low)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="740">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="14">',
        '<text x="80" y="30">Predicted vs measured speedup (%)</text>',
    ]
    for i in range(6):
        v = low + i * (high - low) / 5
        parts.extend(
            [
                f'<text x="{px(v):.1f}" y="685" text-anchor="middle">{v:.1f}</text>',
                f'<text x="70" y="{py(v):.1f}" text-anchor="end">{v:.1f}</text>',
            ]
        )
    parts.extend(
        [
            f'<path d="M80 {py(0):.1f} H680 M{px(0):.1f} 80 V660" stroke="#999" fill="none"/>',
            '<path d="M80 660 L680 80" stroke="#bbb" stroke-dasharray="5 5" fill="none"/>',
            '<text x="260" y="725">Predicted gain (%)</text>',
            '<text transform="translate(20 430) rotate(-90)">Measured gain (%)</text>',
        ]
    )
    colors = ["#2563eb", "#d97706", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]
    for row in rows:
        color = colors[int(row.get("case", 0)) % len(colors)]
        parts.append(
            f'<circle cx="{px(row["theory_gain_pct"]):.2f}" cy="{py(row["actual_gain_pct"]):.2f}" '
            f'r="4" fill="{color}" opacity="0.7"><title>case {row.get("case", 0)} '
            f"step {row['step']}</title></circle>"
        )
    return "\n".join(parts + ["</g></svg>"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, action="append", help="baseline log; repeat per A/B pair")
    parser.add_argument("--chunkflow", type=Path, action="append", help="CF log; repeat in matching order")
    parser.add_argument("--sweep-log-dir", type=Path, help="Select all six pairs from the default 8-GPU tail sweep")
    parser.add_argument("--theory-source", choices=["proxy", "schedule"], default="proxy")
    parser.add_argument("--tolerance-pct", type=float, default=0, help="Neutral band in speedup percentage points")
    parser.add_argument(
        "--output-prefix", type=Path, help="Save per-step CSV, statistics JSON, Markdown table, scatter SVG"
    )
    parser.add_argument(
        "--metric",
        choices=["actor_train_time", "step_time"],
        default="actor_train_time",
        help="A/B timing metric; actor_train_time measures training including the optimizer (default)",
    )
    parser.add_argument("--drop-first", type=int, default=0, help="steps to drop as warmup before averaging")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000, help="paired bootstrap resamples for 95%% CI")
    parser.add_argument("--seed", type=int, default=1234, help="bootstrap seed")
    args = parser.parse_args()
    if args.drop_first < 0:
        parser.error("--drop-first must be nonnegative")
    if args.sweep_log_dir:
        if args.baseline or args.chunkflow:
            parser.error("Use --sweep-log-dir or explicit log pairs, not both")
        try:
            selected = _tail_sweep_pairs(args.sweep_log_dir)
        except ValueError as exc:
            parser.error(str(exc))
        args.baseline, args.chunkflow = map(list, zip(*selected))
    if not args.baseline or not args.chunkflow:
        parser.error("Provide --sweep-log-dir or both --baseline and --chunkflow")
    if len(args.baseline) != len(args.chunkflow) or args.tolerance_pct < 0:
        parser.error("Provide equal numbers of baseline/CF logs and nonnegative --tolerance-pct")
    rows, cases = [], []
    for case, (base, cf) in enumerate(zip(args.baseline, args.chunkflow)):
        print(f"pair {case}: baseline={base} | chunkflow={cf}")
        paired = _pair(base, cf, args.metric, args.theory_source)
        kept = paired[args.drop_first :]
        cases.append(
            {
                "case": case,
                "baseline": str(base),
                "chunkflow": str(cf),
                "paired_before_warmup": len(paired),
                "kept_steps": len(kept),
            }
        )
        if not kept:
            parser.error(
                f"pair {case}: baseline timings={len(_perf_rows(base, args.metric))}, "
                f"CF timings={len(_perf_rows(cf, args.metric))}, theory={len(_theory(cf))}, "
                f"paired={len(paired)}, drop_first={args.drop_first}. Check raw metric lines and job completion."
            )
        for row in kept:
            row["case"] = case
        rows.extend(kept)
    if not rows:
        raise SystemExit(
            "no paired steps: check that both logs come from the same data/seed/parallel layout "
            "and that the chunkflow arm actually ran with --use-chunkflow"
        )

    header = " ".join(f"{name:>{width}}" for name, width in zip(_HEAD, (4, 3, 4, 8, 8, 6, 11, 6, 10, 11, 9, 7)))
    if len(cases) > 1:
        header = "case " + header
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(fmt.format(row[key]) for fmt, key in zip(_FMT, _HEAD))
        print((f"{row['case']:>4} " if len(cases) > 1 else "") + cells)
    print("-" * len(header))
    mean_cells = [f"{'mean':>4}"] + [fmt.format(_mean(rows, key)) for fmt, key in zip(_FMT[1:], _HEAD[1:])]
    print(" ".join(mean_cells))

    theory_x = _aggregate_theory(rows)
    actual_x = _aggregate_speedup(rows)
    ci = _bootstrap_ci(rows, args.bootstrap_samples, args.seed) if len(cases) == 1 else None
    excess_time = sum(row["excess_time"] for row in rows)

    print()
    ci_text = "" if ci is None else f"   bootstrap_95%_ci=[{ci[0]:.3f}, {ci[1]:.3f}]"
    print(f"metric = perf/{args.metric}   paired_steps = {len(rows)}")
    print(f"aggregate_theory_x = {theory_x:.3f}   aggregate_actual_x = {actual_x:.3f}{ci_text}")
    print(f"alignment = {actual_x / theory_x:.3f}   gap = {theory_x / actual_x:.3f}")
    prediction_name = "schedule" if args.theory_source == "schedule" else "bubble"
    print(f"excess_vs_{prediction_name}_prediction = {excess_time:.3f}s total, {excess_time / len(rows):.3f}s/step")
    print(f"logged_last_stage_schedule_simulation_cpu_time = {sum(r['analysis_time_s'] for r in rows):.6f}s")
    print(
        f"cost proxies (reported separately, not multiplied): frag {_mean(rows, 'frag'):.3f} | "
        f"tok_recomp {_mean(rows, 'tok_recomp'):.3f} | attn_recomp {_mean(rows, 'attn_recomp'):.3f} | "
        f"rect_attn {_mean(rows, 'rect_attn'):.3f} | launch {_mean(rows, 'launch'):.1f} | "
        f"p2p_messages {_mean(rows, 'p2p_messages'):.0f} | "
        f"warmup_max {_mean(rows, 'warmup_max'):.1f} | "
        f"min_steady_fraction {_mean(rows, 'min_steady_fraction'):.3f}"
    )
    print("These proxies suggest follow-up chunk-size ablations; one A/B pair cannot identify a causal cost.")
    print(f"Theory source: {args.theory_source}; gain_pct = 100 * (speedup - 1)")
    results = {
        "theory_source": args.theory_source,
        "timing_metric": args.metric,
        "gain_definition": "100 * (speedup - 1)",
        "drop_first_per_case": args.drop_first,
        "schedule_model_assumptions": "F=rows/128, B=2F, Fr=F, balanced stages, zero network latency",
        "cases": cases,
        "pooled": _attribution(rows, args.tolerance_pct),
        "per_case": {},
    }
    _print_attribution("pooled (cross-configuration correlation may be confounded)", results["pooled"])
    for case in cases:
        subset = [r for r in rows if r["case"] == case["case"]]
        result = _attribution(subset, args.tolerance_pct)
        result["aggregate_actual_x"] = _aggregate_speedup(subset)
        result["aggregate_theory_x"] = _aggregate_theory(subset)
        result["actual_speedup_ci95"] = _bootstrap_ci(subset, args.bootstrap_samples, args.seed)
        results["per_case"][str(case["case"])] = result
        _print_attribution(f"case {case['case']}: {case['chunkflow']}", result)
    print("Correlation measures association, not causation or paper conformance.")
    print("Constant predictions give no correlation. Reused baselines make cross-case observations dependent.")
    if args.output_prefix:
        prefix = args.output_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with Path(f"{prefix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        Path(f"{prefix}.json").write_text(json.dumps(results, indent=2, allow_nan=False))
        Path(f"{prefix}.svg").write_text(_scatter_svg(rows))
        Path(f"{prefix}.md").write_text(_markdown_report(results))


if __name__ == "__main__":
    main()
