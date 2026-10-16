# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Diagnose an existing attribution CSV on CPU; never launches training.

Reports ranking, calibration, within-configuration groups and moving-block
bootstrap intervals. These are observational diagnostics, not causal estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import chunkflow_align as align


def read_rows(path: Path) -> list[dict[str, float]]:
    required = {"step", "theory_x", "actual_x", "base_time", "cf_time", "tokens", "M", "U"}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("empty attribution CSV")
    seen = set()
    for row in rows:
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("nonfinite CSV values")
        row.setdefault("case", 0)
        identity = row["case"], row["step"]
        if identity in seen:
            raise ValueError(f"duplicate case/step: {identity}")
        seen.add(identity)
        if any(row[key] <= 0 for key in ("base_time", "cf_time", "theory_x", "actual_x", "tokens")):
            raise ValueError(f"nonpositive timing, prediction or tokens: {identity}")
        if not math.isclose(row["actual_x"], row["base_time"] / row["cf_time"], rel_tol=1e-6):
            raise ValueError(f"actual_x inconsistent with timings: {identity}")
        row["theory_gain_pct"] = 100 * (row["theory_x"] - 1)
        row["actual_gain_pct"] = 100 * (row["actual_x"] - 1)
        row["ideal_cf_time"] = row["base_time"] / row["theory_x"]
        row["excess_fraction"] = (row["cf_time"] - row["ideal_cf_time"]) / row["base_time"]
    return sorted(rows, key=lambda row: (row["case"], row["step"]))


def auc(scores: list[float], labels: list[float]) -> float | None:
    """Mann-Whitney AUROC, preserving tied prediction scores."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = align._ranks(scores)
    rank_sum = sum(rank + 1 for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def summarize(rows: list[dict[str, float]], tolerance: float) -> dict[str, Any]:
    result = align._attribution(rows, tolerance)
    xs = [row["theory_gain_pct"] for row in rows]
    ys = [row["actual_gain_pct"] for row in rows]
    directional = [row for row in rows if abs(row["actual_gain_pct"]) > tolerance]
    labels = [float(row["actual_gain_pct"] > 0) for row in directional]
    scores = [row["theory_gain_pct"] for row in directional]
    prediction_positive = [row for row in rows if row["theory_gain_pct"] > tolerance]
    result.update(
        baseline_total_s=sum(row["base_time"] for row in rows),
        chunkflow_total_s=sum(row["cf_time"] for row in rows),
        aggregate_actual_x=align._aggregate_speedup(rows),
        aggregate_theory_x=align._aggregate_theory(rows),
        predicted_gain_min_pct=min(xs),
        predicted_gain_max_pct=max(xs),
        mean_prediction_bias_pp=sum(x - y for x, y in zip(xs, ys)) / len(rows),
        actual_positive_fraction=sum(y > tolerance for y in ys) / len(rows),
        predicted_positive_precision=(
            sum(row["actual_gain_pct"] > tolerance for row in prediction_positive) / len(prediction_positive)
            if prediction_positive
            else None
        ),
        # A constant positive prediction has undefined sign correlation.
        sign_correlation=align._pearson(
            [float(row["theory_gain_pct"] > 0) for row in directional if abs(row["theory_gain_pct"]) > tolerance],
            [float(row["actual_gain_pct"] > 0) for row in directional if abs(row["theory_gain_pct"]) > tolerance],
        ),
        predicted_gain_vs_actual_win_correlation=align._pearson(scores, labels),
        actual_win_auroc=auc(scores, labels),
        logged_simulator_cpu_fraction=sum(row.get("analysis_time_s", 0) for row in rows)
        / sum(row["cf_time"] for row in rows),
    )
    return result


def token_adjusted_correlation(rows: list[dict[str, float]]) -> float | None:
    """Descriptive partial Pearson after linear removal of step-token count."""
    tokens = [row["tokens"] for row in rows]
    mean_tokens = sum(tokens) / len(tokens)
    variance = sum((value - mean_tokens) ** 2 for value in tokens)
    if len(rows) < 4 or variance <= 1e-24:
        return None

    def residual(key: str) -> list[float]:
        values = [row[key] for row in rows]
        mean = sum(values) / len(values)
        slope = sum((t - mean_tokens) * (v - mean) for t, v in zip(tokens, values)) / variance
        return [v - mean - slope * (t - mean_tokens) for t, v in zip(tokens, values)]

    return align._pearson(residual("theory_gain_pct"), residual("actual_gain_pct"))


def block_intervals(rows: list[dict[str, float]], samples: int, block: int, seed: int) -> dict[str, Any]:
    """Resample contiguous blocks within one case; never mix configurations."""
    n = len(rows)
    if n < 2 * block or any(b["step"] != a["step"] + 1 for a, b in zip(rows, rows[1:])):
        return {"available": False, "reason": "need contiguous steps and at least two blocks"}
    rng = random.Random(seed)
    values: dict[str, list[float]] = {key: [] for key in ("actual_x", "pearson", "spearman")}
    for _ in range(samples):
        draw = []
        while len(draw) < n:
            start = rng.randrange(n - block + 1)
            draw.extend(rows[start : start + block])
        draw = draw[:n]
        xs = [row["theory_gain_pct"] for row in draw]
        ys = [row["actual_gain_pct"] for row in draw]
        estimates = {
            "actual_x": align._aggregate_speedup(draw),
            "pearson": align._pearson(xs, ys),
            "spearman": align._pearson(align._ranks(xs), align._ranks(ys)),
        }
        for key, estimate in estimates.items():
            if estimate is not None:
                values[key].append(estimate)
    result: dict[str, Any] = {"available": True, "block_steps": block, "samples": samples}
    for key, estimates in values.items():
        estimates.sort()
        # Do not report an interval from only a handful of nonconstant draws.
        result[key] = (
            [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]]
            if len(estimates) >= 0.9 * samples
            else None
        )
    return result


def grouped(rows: list[dict[str, float]], key: str, tolerance: float) -> list[dict[str, Any]]:
    values = sorted(row[key] for row in rows)
    # Equal predictions stay together, even if that produces fewer than four bins.
    cuts = sorted({values[min(len(values) - 1, len(values) * i // 4)] for i in (1, 2, 3)})
    groups: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        index = sum(row[key] > cut for cut in cuts)
        groups.setdefault(index, []).append(row)
    return [
        {"min": min(row[key] for row in group), "max": max(row[key] for row in group), **summarize(group, tolerance)}
        for _, group in sorted(groups.items())
    ]


def diagnose(rows: list[dict[str, float]], tolerance: float, samples: int, block: int, seed: int) -> dict[str, Any]:
    cases = {}
    for case in sorted({row["case"] for row in rows}):
        subset = [row for row in rows if row["case"] == case]
        summaries = {
            str(m): {
                **summarize([row for row in subset if row["M"] == m], tolerance),
                "linear_token_adjusted_pearson": token_adjusted_correlation([row for row in subset if row["M"] == m]),
            }
            for m in sorted({row["M"] for row in subset})
        }
        cost_associations = {}
        for key in ("launch", "rect_attn", "frag", "tok_recomp", "p2p_messages", "warmup_max", "analysis_time_s"):
            if all(key in row for row in subset):
                cost_associations[key] = align._pearson(
                    [row[key] for row in subset], [row["excess_fraction"] for row in subset]
                )
        worst = sorted(subset, key=lambda row: row["actual_x"])[:5]
        best = sorted(subset, key=lambda row: row["actual_x"], reverse=True)[:5]
        cases[str(int(case))] = {
            "summary": summarize(subset, tolerance),
            "block_ci95": block_intervals(subset, samples, block, seed),
            "by_predicted_gain": grouped(subset, "theory_gain_pct", tolerance),
            "by_step_tokens": grouped(subset, "tokens", tolerance),
            "by_M": summaries,
            "chronological_halves": [
                summarize(group, tolerance)
                for group in (subset[: len(subset) // 2], subset[len(subset) // 2 :])
                if group
            ],
            "proxy_vs_excess_fraction_pearson": cost_associations,
            "representative_steps": {"best": best, "worst": worst},
        }
    return {"cases": cases, "tolerance_pct": tolerance}


def markdown(report: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "未定义" if value is None else f"{value:.4f}"

    parts = ["# ChunkFlow 收益诊断", "", "仅分析观测关联，不代表因果归因或论文复现结论。", ""]
    for case, result in report["cases"].items():
        summary = result["summary"]
        parts.extend([f"## case {case}", "", "| 指标 | 数值 |", "| --- | ---: |"])
        for key, label in (
            ("paired_steps", "配对 step 数"),
            ("aggregate_theory_x", "预测总体加速比"),
            ("aggregate_actual_x", "实测总体加速比"),
            ("pearson", "Pearson"),
            ("spearman", "Spearman"),
            ("sign_correlation", "预测正负与实际正负的相关性"),
            ("predicted_gain_vs_actual_win_correlation", "预测幅度与实际是否提速的相关性"),
            ("actual_win_auroc", "预测幅度区分实际正负的 AUROC"),
            ("predicted_positive_precision", "预测正收益中的实际正收益比例"),
            ("mean_prediction_bias_pp", "平均高估（百分点）"),
            ("logged_simulator_cpu_fraction", "已记录模拟器 CPU 时间 / ChunkFlow 训练时间"),
        ):
            parts.append(f"| {label} | {fmt(summary[key])} |")
        parts.extend(["", "连续 block bootstrap 的 95% 区间：", "", f"`{json.dumps(result['block_ci95'])}`", ""])
        for key, title in (("by_predicted_gain", "按预测收益分档（%）"), ("by_step_tokens", "按整步 token 数分档")):
            parts.extend(
                [
                    f"### {title}",
                    "",
                    "| 范围 | step 数 | 实际提速率 | 总体实际加速比 | Pearson | Spearman |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for group in result[key]:
                parts.append(
                    f"| {group['min']:.2f}–{group['max']:.2f} | {group['paired_steps']} | "
                    f"{group['actual_positive_fraction']:.1%} | {group['aggregate_actual_x']:.4f} | "
                    f"{fmt(group['pearson'])} | {fmt(group['spearman'])} |"
                )
            parts.append("")
        parts.extend(
            [
                "### 按 M 分组",
                "",
                "| M | step 数 | 实际提速率 | 实际加速比 | Pearson | Spearman | 控制 token 数后的 Pearson |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for m, group in result["by_M"].items():
            parts.append(
                f"| {m} | {group['paired_steps']} | {group['actual_positive_fraction']:.1%} | "
                f"{group['aggregate_actual_x']:.4f} | {fmt(group['pearson'])} | {fmt(group['spearman'])} | "
                f"{fmt(group['linear_token_adjusted_pearson'])} |"
            )
        parts.extend(["", "代表性 step 与成本代理相关性见同名 JSON。", ""])
    parts.extend(
        [
            "## 解读限制",
            "",
            "- 预测全为正时，正负方向的相关性未定义；连续收益仍可能正相关。",
            "- AUROC=0.5 表示无排序区分能力；仅存在一类实际结果时未定义。",
            "- 分档只用于描述，不能在同一批数据选阈值后宣称泛化有效；阈值需要独立运行验证。",
            "- block 区间考虑局部 step 依赖，仍不能消除两组顺序执行造成的机器状态差异；应改变 block 长度检查敏感性。",
            "- 整步 token 数不能识别某一步是否包含超长样本，相关性也可能受 M、长度等共同因素影响。",
            "- 控制 token 数只去除线性关系，不能排除非线性长度效应或其他混杂；前后半段摘要见 JSON，未使用数据调阈值。",
            "- 成本代理与预测残差的相关性存在共享变量，不能作为因果证据；已有 CSV 的默认值也不证明指标被实际采集。",
            "- 模拟器时间只覆盖已记录部分，不应直接从 GPU 关键路径耗时中相减。",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Existing attribution.csv; warmup already removed")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--tolerance-pct", type=float, default=1, help="Neutral band in speedup percentage points")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--block-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if (
        not math.isfinite(args.tolerance_pct)
        or args.tolerance_pct < 0
        or args.bootstrap_samples < 10
        or args.block_steps < 1
    ):
        parser.error("tolerance must be finite and nonnegative; bootstrap samples >= 10; block steps >= 1")
    try:
        report = diagnose(read_rows(args.csv), args.tolerance_pct, args.bootstrap_samples, args.block_steps, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    report["input_csv"] = str(args.csv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_prefix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    text = markdown(report)
    Path(f"{args.output_prefix}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
