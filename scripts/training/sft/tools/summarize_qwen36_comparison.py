# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Summarize paired numerical differences and timing without third-party
dependencies."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from pathlib import Path


METRICS = ("train/loss", "train/grad_norm", "perf/actor_train_time", "perf/step_time", "perf/actor_train_tokens")


def read_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
            match = re.search(r"(?:perf|step) (\d+): (\{.*\})", line)
            if match:
                try:
                    data = ast.literal_eval(match[2])
                except (ValueError, SyntaxError):
                    continue
                if isinstance(data, dict):
                    rows.setdefault(int(match[1]), {}).update(data)
    return rows


def number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def summarize(folder: Path, expected_steps: int | None = None) -> bool:
    base, chunk = (read_rows(folder / f"{arm}.log") for arm in ("baseline", "chunkflow"))
    steps = sorted(base.keys() & chunk.keys())
    all_steps = sorted(base.keys() | chunk.keys() | (set(range(expected_steps)) if expected_steps else set()))
    fields = ["step", *[f"{arm}:{metric}" for metric in METRICS for arm in ("baseline", "chunkflow")]]
    with (folder / "paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for step in all_steps:
            row = {"step": step}
            row.update(
                {
                    f"{arm}:{key}": data.get(step, {}).get(key)
                    for key in METRICS
                    for arm, data in (("baseline", base), ("chunkflow", chunk))
                }
            )
            writer.writerow(row)
    issues = []
    coverage = []
    required = set(range(expected_steps)) if expected_steps is not None else set(all_steps)
    for arm, data in (("baseline", base), ("chunkflow", chunk)):
        for metric in ("train/loss", "perf/actor_train_time", "perf/step_time", "perf/actor_train_tokens"):
            valid = {step for step, row in data.items() if number(row.get(metric))}
            missing = sorted(required - valid)
            coverage.append(f"| {arm} | {metric} | {len(valid & required)}/{len(required)} | {missing or 'none'} |")
            if missing:
                issues.append(f"{arm}: missing/nonfinite {metric} at {missing}")
        if required != set(data):
            issues.append(f"{arm}: step IDs differ from expected range; extra={sorted(set(data) - required)}")
    if not required:
        issues.append("No training steps found")
    lines = [
        "# Loss and performance diagnosis",
        "",
        f"Matched step IDs: {len(steps)}.",
        f"Expected steps per arm: {expected_steps if expected_steps is not None else 'not specified'}.",
        "Matching step IDs and token counts does not prove identical sample contents.",
        "",
        "## Completeness",
        "",
        "| Arm | Metric | Valid / expected | Missing or nonfinite steps |",
        "| --- | --- | ---: | --- |",
        *coverage,
        "",
        "## ClearML",
        "",
    ]
    for arm in ("baseline", "chunkflow"):
        urls = sorted(
            set(re.findall(r"url:\s*(https?://[^\s\x1b]+)", (folder / f"{arm}.log").read_text(errors="replace")))
        )
        lines.extend(f"- {arm}: {url}" for url in urls)
        if not urls:
            lines.append(f"- {arm}: no task URL found in log; all parsed points are in paired.csv.")
    lines += [
        "",
        "## Numerical differences",
        "",
        "| Step | Baseline loss | ChunkFlow loss | Absolute loss difference | Relative grad-norm difference |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in all_steps:
        b, c = base.get(step, {}), chunk.get(step, {})
        lb, lc = b.get("train/loss"), c.get("train/loss")
        gb, gc = b.get("train/grad_norm"), c.get("train/grad_norm")
        delta = f"{abs(lc - lb):.6g}" if number(lb) and number(lc) else "missing"
        grad = f"{abs(gc - gb) / abs(gb):.2%}" if number(gb) and number(gc) and gb else "undefined"
        lines.append(f"| {step} | {lb} | {lc} | {delta} | {grad} |")
    lines += [
        "",
        "Every step is shown, including warmup. Missing values are retained in paired.csv. Grad norms alone do not prove gradient alignment.",
        "",
        "## Timing sensitivity",
        "",
        "Speedup = sum(baseline time) / sum(ChunkFlow time). Report every window; do not select the best one.",
        "Throughput gain = speedup - 1; time saved = 1 - 1/speedup. step_time covers the instrumented training step, not job startup/model loading.",
        "",
        "| Start step | Metric | Pairs | Baseline mean (s) | ChunkFlow mean (s) | Speedup | Throughput gain | Time saved |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for start in (0, 10, 30, 50):
        for metric in ("perf/actor_train_time", "perf/step_time"):
            valid = [
                s
                for s in steps
                if s in required
                and s >= start
                and all(number(d[s].get(metric)) and d[s][metric] > 0 for d in (base, chunk))
            ]
            if not valid:
                continue
            window = {s for s in required if s >= start}
            if set(valid) != window:
                lines.append(
                    f"\nWindow {start}, {metric}: refused speedup; missing/invalid timing at {sorted(window - set(valid))}.\n"
                )
                continue
            mismatches = [
                s
                for s in valid
                if not number(base[s].get("perf/actor_train_tokens"))
                or base[s]["perf/actor_train_tokens"] <= 0
                or base[s].get("perf/actor_train_tokens") != chunk[s].get("perf/actor_train_tokens")
            ]
            if mismatches:
                lines.append(
                    f"\nWindow {start}, {metric}: refused speedup; missing/mismatched tokens at {mismatches}.\n"
                )
                continue
            bt, ct = (sum(d[s][metric] for s in valid) for d in (base, chunk))
            lines.append(
                f"| {start} | {metric} | {len(valid)} | {bt / len(valid):.4f} | {ct / len(valid):.4f} | {bt / ct:.4f}x | {bt / ct - 1:+.2%} | {1 - ct / bt:+.2%} |"
            )
    lines += [
        "",
        "Profiler runs are diagnostic only. LR=0 timing is not evidence of normal-training speedup.",
        "Missing steps are not filled. Check comparison.md for completion and experiment settings.",
        "Sequential runs can be affected by machine state and run order; one pair is a screening comparison.",
        "",
        "## Data issues",
        *([f"- {issue}" for issue in issues] if issues else ["None."]),
    ]
    (folder / "diagnosis.md").write_text("\n".join(lines) + "\n")
    return not issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()
    if args.expected_steps is not None and args.expected_steps <= 0:
        parser.error("--expected-steps must be positive")
    if not summarize(args.folder, args.expected_steps):
        raise SystemExit("Incomplete metrics; inspect diagnosis.md and paired.csv")
