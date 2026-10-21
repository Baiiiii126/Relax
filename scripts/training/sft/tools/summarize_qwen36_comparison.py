# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Summarize paired numerical differences and timing without third-party
dependencies."""

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


def summarize(folder: Path) -> None:
    base, chunk = (read_rows(folder / f"{arm}.log") for arm in ("baseline", "chunkflow"))
    steps = sorted(base.keys() & chunk.keys())
    fields = ["step", *[f"{arm}:{metric}" for metric in METRICS for arm in ("baseline", "chunkflow")]]
    with (folder / "paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for step in steps:
            row = {"step": step}
            row.update(
                {
                    f"{arm}:{key}": data[step].get(key)
                    for key in METRICS
                    for arm, data in (("baseline", base), ("chunkflow", chunk))
                }
            )
            writer.writerow(row)
    lines = [
        "# Loss and performance diagnosis",
        "",
        f"Matched step IDs: {len(steps)}.",
        "Matching step IDs and token counts does not prove identical sample contents.",
        "",
        "## Numerical differences",
        "",
        "| Step | Baseline loss | ChunkFlow loss | Absolute loss difference | Relative grad-norm difference |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in steps[:10]:
        b, c = base[step], chunk[step]
        lb, lc = b.get("train/loss"), c.get("train/loss")
        gb, gc = b.get("train/grad_norm"), c.get("train/grad_norm")
        delta = f"{abs(lc - lb):.6g}" if number(lb) and number(lc) else "missing"
        grad = f"{abs(gc - gb) / abs(gb):.2%}" if number(gb) and number(gc) and gb else "undefined"
        lines.append(f"| {step} | {lb} | {lc} | {delta} | {grad} |")
    lines += [
        "",
        "LR=0 isolates forward/backward differences from weight-update drift. Grad norms alone do not prove gradient alignment.",
        "",
        "## Timing sensitivity",
        "",
        "Speedup = sum(baseline time) / sum(ChunkFlow time). Report every window; do not select the best one.",
        "",
        "| Start step | Metric | Pairs | Baseline mean (s) | ChunkFlow mean (s) | Speedup |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for start in (0, 10, 30, 50):
        for metric in ("perf/actor_train_time", "perf/step_time"):
            valid = [
                s
                for s in steps
                if s >= start and all(number(d[s].get(metric)) and d[s][metric] > 0 for d in (base, chunk))
            ]
            if not valid:
                continue
            mismatches = [
                s
                for s in valid
                if not number(base[s].get("perf/actor_train_tokens"))
                or base[s].get("perf/actor_train_tokens") != chunk[s].get("perf/actor_train_tokens")
            ]
            if mismatches:
                lines.append(
                    f"\nWindow {start}, {metric}: refused speedup; missing/mismatched tokens at {mismatches}.\n"
                )
                continue
            bt, ct = (sum(d[s][metric] for s in valid) for d in (base, chunk))
            lines.append(
                f"| {start} | {metric} | {len(valid)} | {bt / len(valid):.4f} | {ct / len(valid):.4f} | {bt / ct:.4f}x |"
            )
    lines += [
        "",
        "Profiler runs are diagnostic only. LR=0 timing is not evidence of normal-training speedup.",
        "Missing steps are not filled. Check comparison.md for completion and experiment settings.",
    ]
    (folder / "diagnosis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
