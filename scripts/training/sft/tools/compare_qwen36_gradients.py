# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare diagnostic gradient samples on CPU, preserving rank-local parameter
identity."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def compare(folder: Path, rollouts: tuple[int, ...] = (0, 1, 2, 8)) -> None:
    import torch

    base, chunk = (folder / arm / "diagnostics" for arm in ("baseline", "chunkflow"))
    files = {p.name for p in base.glob("grad_*.pt")} | {p.name for p in chunk.glob("grad_*.pt")}
    issues, rows = [], []
    expected = {f"grad_r{rollout}_s0_rank{rank}.pt" for rollout in rollouts for rank in range(8)}
    issues.extend(f"Expected snapshot absent in both arms: {name}" for name in sorted(expected - files))
    for name in sorted(files):
        if not (base / name).exists() or not (chunk / name).exists():
            issues.append(f"Missing counterpart: {name}")
            continue
        b, c = (torch.load(p / name, map_location="cpu", weights_only=True) for p in (base, chunk))
        groups = defaultdict(lambda: defaultdict(float))
        for key in sorted(b.keys() | c.keys()):
            if key not in b or key not in c:
                issues.append(f"Missing parameter: {name}: {key}")
                continue
            x, y = b[key], c[key]
            if y.get("overwrite_main_grad", False):
                issues.append(f"ChunkFlow overwrite_main_grad is enabled: {name}: {key}")
            if any(x.get(k) != y.get(k) for k in ("shape", "group", "source", "dtype", "sample_count")):
                issues.append(f"Metadata mismatch: {name}: {key}")
                continue
            if x["missing"] or y["missing"]:
                issues.append(f"Missing gradient: {name}: {key} baseline={x['missing']} chunkflow={y['missing']}")
                continue
            gx, gy = x["sample"].double(), y["sample"].double()
            if not (
                math.isfinite(x["norm"])
                and math.isfinite(y["norm"])
                and torch.isfinite(gx).all()
                and torch.isfinite(gy).all()
                and torch.isfinite(x["weight_sample"]).all()
                and torch.isfinite(y["weight_sample"]).all()
            ):
                issues.append(f"Nonfinite values: {name}: {key}")
                continue
            if (x["norm"] == 0) != (y["norm"] == 0):
                issues.append(f"Zero/nonzero gradient mismatch: {name}: {key}")
            stats = groups[x["group"]]
            stats["parameters"] += 1
            stats["baseline_norm_sq"] += x["norm"] ** 2
            stats["chunkflow_norm_sq"] += y["norm"] ** 2
            stats["sample_baseline_sq"] += float(gx.square().sum())
            stats["sample_chunkflow_sq"] += float(gy.square().sum())
            stats["sample_error_sq"] += float((gy - gx).square().sum())
            stats["sample_dot"] += float((gx * gy).sum())
            stats["both_zero_samples"] += int(not torch.count_nonzero(gx) and not torch.count_nonzero(gy))
            if gx.numel():
                stats["max_weight_sample_diff"] = max(
                    stats["max_weight_sample_diff"], float((x["weight_sample"] - y["weight_sample"]).abs().max())
                )
        for group, stats in groups.items():
            bn, cn = math.sqrt(stats["baseline_norm_sq"]), math.sqrt(stats["chunkflow_norm_sq"])
            bx, cy = stats["sample_baseline_sq"], stats["sample_chunkflow_sq"]
            rows.append(
                {
                    "snapshot": name,
                    "group": group,
                    "parameters": int(stats["parameters"]),
                    "baseline_norm": bn,
                    "chunkflow_norm": cn,
                    "norm_ratio": cn / bn if bn else None,
                    "sample_relative_l2": math.sqrt(stats["sample_error_sq"] / bx) if bx else None,
                    "sample_cosine": stats["sample_dot"] / math.sqrt(bx * cy) if bx and cy else None,
                    "both_zero_samples": int(stats["both_zero_samples"]),
                    "max_weight_sample_diff": stats["max_weight_sample_diff"],
                }
            )
    if rows:
        with (folder / "gradients.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Gradient and backend diagnosis",
        "",
        "Gradients: after schedule finalization, before optimizer prepare/unscale/clip/update.",
        "Norms use full local tensors. Cosine and relative L2 use 64 fixed coordinates per parameter only.",
        "Norm ratio = ChunkFlow / baseline; sample relative L2 = ||ChunkFlow - baseline|| / ||baseline||.",
        "Samples can miss errors, especially in sparse experts; matching samples do not prove full alignment.",
        "Do not sum norms across ranks to reconstruct optimizer grad_norm (TP/EP replicas are not deduplicated).",
        "Timing includes diagnostic overhead and must not be used to claim speedup.",
        "",
        "## GDN Python call sources (GPU kernel dispatch requires the profiler trace)",
    ]
    for arm, path in (("baseline", base), ("chunkflow", chunk)):
        sources = set()
        for logfile in path.glob("backend_*.jsonl"):
            for line in logfile.read_text().splitlines():
                row = json.loads(line)
                sources.add(f"{row['callable']} — {row['source']} (wrapper: {row['wrapper_source']})")
        lines.append(f"\n{arm}: " + ("; ".join(sorted(sources)) or "MISSING"))
    lines += [
        "",
        "## Rank-local group comparison",
        "",
        "| Snapshot | Group | Norm ratio | Sample relative L2 | Sample cosine | Max weight sample difference |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [
            f"{row[k]:.6g}" if row[k] is not None else "undefined"
            for k in ("norm_ratio", "sample_relative_l2", "sample_cosine", "max_weight_sample_diff")
        ]
        lines.append(f"| {row['snapshot']} | {row['group']} | " + " | ".join(values) + " |")
    lines += ["", "## Missing or invalid data", *[f"- {issue}" for issue in issues]]
    if not files:
        lines.append("No gradient snapshots found; this is not a passing result.")
    (folder / "gradients.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rollouts", default="0,1,2,8")
    args = parser.parse_args()
    compare(args.folder, tuple(int(value) for value in args.rollouts.split(",")))
