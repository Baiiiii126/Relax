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
POOL_METRICS = tuple(
    f"train/chunkflow/{key}"
    for key in (
        "num_microbatches",
        "num_units",
        "unit_padded_rows",
        "tail_pool_enabled",
        "unpooled_units",
        "tail_pool_saved_units",
        "tail_pool_pooled_units",
        "max_warmup_units",
        "peak_live_activation_rows",
    )
)


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


def summarize(
    folder: Path,
    expected_steps: int | None = None,
    *,
    baseline_arm: str = "baseline",
    chunkflow_arm: str = "chunkflow",
    output_prefix: str = "",
) -> bool:
    base, chunk = (read_rows(folder / f"{arm}.log") for arm in (baseline_arm, chunkflow_arm))
    steps = sorted(base.keys() & chunk.keys())
    all_steps = sorted(base.keys() | chunk.keys() | (set(range(expected_steps)) if expected_steps else set()))
    fields = [
        "step",
        *[f"{arm}:{metric}" for metric in (*METRICS, *POOL_METRICS) for arm in ("baseline", "chunkflow")],
    ]
    with (folder / f"{output_prefix}paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for step in all_steps:
            row = {"step": step}
            row.update(
                {
                    f"{arm}:{key}": data.get(step, {}).get(key)
                    for key in (*METRICS, *POOL_METRICS)
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
        f"Reference (baseline columns): {baseline_arm}; candidate (ChunkFlow columns): {chunkflow_arm}.",
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
    for arm in (baseline_arm, chunkflow_arm):
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
    if chunkflow_arm == "chunkflow_pool":
        invalid = [s for s in required if chunk.get(s, {}).get("train/chunkflow/tail_pool_enabled") != 1]
        if invalid:
            issues.append(f"Missing tail-pool execution evidence at steps {invalid}")
    lines += [
        "",
        "Profiler runs are diagnostic only. LR=0 timing is not evidence of normal-training speedup.",
        "Missing steps are not filled. Check comparison.md for completion and experiment settings.",
        "Sequential runs can be affected by machine state and run order; one pair is a screening comparison.",
        "",
        "## Data issues",
        *([f"- {issue}" for issue in issues] if issues else ["None."]),
    ]
    if chunkflow_arm == "chunkflow_pool":
        lines += [
            "",
            "## Tail pool execution",
            "",
            "Activation rows are a scheduling estimate, not measured GPU memory.",
            "",
            "| Step | M | Unpooled units | Executed units | Saved units | Pooled units | Max warmup | Peak live rows |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for s in all_steps:
            keys = (
                "num_microbatches",
                "unpooled_units",
                "num_units",
                "tail_pool_saved_units",
                "tail_pool_pooled_units",
                "max_warmup_units",
                "peak_live_activation_rows",
            )
            values = [str(chunk.get(s, {}).get(f"train/chunkflow/{k}", "missing")) for k in keys]
            lines.append("| " + " | ".join([str(s), *values]) + " |")
    (folder / f"{output_prefix}diagnosis.md").write_text("\n".join(lines) + "\n")
    return not issues


def summarize_sweep(folder: Path, chunks: list[int], expected_steps: int) -> bool:
    records = []
    complete = True
    lines = [
        "# Fixed-GBS chunk-size comparison",
        "",
        "Configuration and run completion: comparison.md. All candidates share one baseline; GBS is unchanged.",
        "Primary window: step 30 onward. Step 10 onward is also shown to expose warmup sensitivity.",
        "Speedup = baseline total time / candidate total time. Above 1 is faster; below 1 is slower.",
        "Step time excludes job startup/model loading. Do not average per-step speedup ratios.",
        "",
        "| Chunk | Start step | Pairs | Training speedup | Step speedup | Step throughput change | Slower training steps | Max absolute loss difference (all steps) | Status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for size in chunks:
        arm = f"chunk{size}"
        paths = (folder / "baseline.log", folder / f"{arm}.log")
        valid = False
        status = "pending"
        if all(p.exists() for p in paths):
            valid = summarize(folder, expected_steps, chunkflow_arm=arm, output_prefix=f"{arm}-")
            base, chunk = (read_rows(p) for p in paths)
            valid = valid and all(
                base[s]["perf/actor_train_tokens"] > 0
                and base[s]["perf/actor_train_tokens"] == chunk[s]["perf/actor_train_tokens"]
                and all(d[s][m] > 0 for d in (base, chunk) for m in ("perf/actor_train_time", "perf/step_time"))
                for s in range(expected_steps)
            )
            status = "complete" if valid else "invalid: inspect diagnosis and token counts"
        complete = complete and valid
        for start in (30, 10):
            row = {"chunk": size, "start_step": start, "pairs": 0, "status": status}
            if valid:
                steps = range(start, expected_steps)
                train_x, step_x = (
                    sum(base[s][m] for s in steps) / sum(chunk[s][m] for s in steps)
                    for m in ("perf/actor_train_time", "perf/step_time")
                )
                loss_delta = max(abs(chunk[s]["train/loss"] - base[s]["train/loss"]) for s in range(expected_steps))
                slower = sum(chunk[s]["perf/actor_train_time"] > base[s]["perf/actor_train_time"] for s in steps)
                row.update(
                    pairs=len(steps),
                    train_speedup=train_x,
                    step_speedup=step_x,
                    step_throughput_change=step_x - 1,
                    train_slower_fraction=slower / len(steps),
                    max_abs_loss_delta=loss_delta,
                )
                lines.append(
                    f"| {size} | {start} | {len(steps)} | {train_x:.4f}x | {step_x:.4f}x | {step_x - 1:+.2%} | {slower}/{len(steps)} | {loss_delta:.6g} | {status} |"
                )
            else:
                lines.append(f"| {size} | {start} | — | — | — | — | — | — | {status} |")
            records.append(row)
    lines += [
        "",
        "Per-size chunk*-paired.csv preserves every loss/timing point; chunk*-diagnosis.md includes ClearML links.",
        "All sizes are shown, including regressions. Invalid or unequal-token runs are excluded from speedup calculations.",
        "Loss differences are reported, not automatically accepted as correct; matching token counts does not prove identical data.",
        "Slower-step counts use a strict time comparison without a noise tolerance.",
        "This is parameter screening. Shared baselines and sequential run order do not establish statistical significance.",
        "Confirm a promising size by rerunning baseline/ChunkFlow in reverse order with the same configuration, without profiler.",
    ]
    (folder / "sweep.md").write_text("\n".join(lines) + "\n")
    fields = (
        "chunk",
        "start_step",
        "pairs",
        "train_speedup",
        "step_speedup",
        "step_throughput_change",
        "train_slower_fraction",
        "max_abs_loss_delta",
        "status",
    )
    with (folder / "sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--baseline-arm", choices=("baseline", "chunkflow"), default="baseline")
    parser.add_argument("--chunkflow-arm", choices=("chunkflow", "chunkflow_pool"), default="chunkflow")
    parser.add_argument("--output-prefix", choices=("", "tail_pool-", "tail_pool-vs-unpooled-"), default="")
    parser.add_argument("--chunks", type=int, nargs="+")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.expected_steps is not None and args.expected_steps <= 0:
        parser.error("--expected-steps must be positive")
    if args.chunks is not None:
        if args.expected_steps is None or args.expected_steps <= 30:
            parser.error("--chunks requires --expected-steps > 30")
        if min(args.chunks) <= 0 or len(set(args.chunks)) != len(args.chunks):
            parser.error("--chunks must contain distinct positive sizes")
        if not summarize_sweep(args.folder, args.chunks, args.expected_steps) and args.require_complete:
            raise SystemExit("Incomplete sweep; inspect sweep.md")
        raise SystemExit(0)
    if not summarize(
        args.folder,
        args.expected_steps,
        baseline_arm=args.baseline_arm,
        chunkflow_arm=args.chunkflow_arm,
        output_prefix=args.output_prefix,
    ):
        raise SystemExit("Incomplete metrics; inspect diagnosis.md and paired.csv")
