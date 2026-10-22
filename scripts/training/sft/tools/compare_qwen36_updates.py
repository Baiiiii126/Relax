# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare sampled first-update gradients, FP32 master weights and Adam
states."""

import argparse
import csv
import math
import re
from pathlib import Path


def compare(folder: Path, candidate: str = "chunkflow") -> None:
    import torch

    fields = (
        "before_model",
        "before_master",
        "prepared_grad",
        "clipped_grad",
        "master_delta",
        "exp_avg",
        "exp_avg_sq",
        "model_delta",
    )
    rows, issues = [], []
    for rollout in (0, 1):
        for rank in range(8):
            name = f"update_r{rollout}_s0_rank{rank}.pt"
            paths = [folder / arm / "diagnostics" / name for arm in ("baseline", candidate)]
            if not all(p.exists() for p in paths):
                issues.append(f"Missing snapshot: {name}")
                continue
            loaded = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
            arms = []
            ranges = []
            for arm, data in zip(("baseline", candidate), loaded, strict=True):
                issues.extend(f"{arm}: {name}: {issue}" for issue in data["issues"])
                values = {field: {} for field in fields}
                values["before_model"] = data["before_model"]
                values["prepared_grad"] = data.get("prepared_grad", {})
                intervals = {}
                for record in data["optimizers"]:
                    intervals.update(record["ranges"])
                    for field in ("before_master", "clipped_grad", "exp_avg", "exp_avg_sq"):
                        values[field].update(record[field])
                    for key, before in record["before_master"].items():
                        values["master_delta"][key] = record["after_master"][key] - before
                for key, before in data["before_model"].items():
                    values["model_delta"][key] = data["after_model"][key] - before
                arms.append(values)
                ranges.append(intervals)
            if ranges[0] != ranges[1]:
                issues.append(f"Shard ranges differ: {name}; comparison skipped")
                continue
            for field in fields:
                b, c = (arm[field] for arm in arms)
                if not b or not c:
                    issues.append(f"Empty required field: {name} {field}")
                if b.keys() != c.keys():
                    issues.append(f"Missing parameter counterpart: {name} {field}")
                for key in sorted(b.keys() & c.keys()):
                    x, y = b[key].double(), c[key].double()
                    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                        issues.append(f"Invalid sample: {name} {field} {key}")
                        continue
                    xn, yn = float(x.norm()), float(y.norm())
                    layer = re.search(r"\.layers\.(\d+)\.", key)
                    rows.append(
                        {
                            "rollout": rollout,
                            "rank": rank,
                            "parameter": key,
                            "local_layer": int(layer[1]) if layer else -1,
                            "field": field,
                            "baseline_norm": xn,
                            "candidate_norm": yn,
                            "relative_l2": float((x - y).norm()) / xn if xn else None,
                            "sample_error_sq": float((x - y).square().sum()),
                            "cosine": float((x * y).sum()) / (xn * yn) if xn * yn else None,
                            "max_abs": float((x - y).abs().max()),
                            "nonzero_sign_mismatches": int(((x * y) < 0).sum()),
                        }
                    )
    suffix = "updates" if candidate == "chunkflow" else "updates-baseline-repeat"
    if rows:
        with (folder / f"{suffix}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# First-update diagnosis",
        "",
        f"Comparison: baseline vs {candidate}.",
        "64 fixed coordinates per local optimizer shard; model deltas use model-local coordinates.",
        "Sampled equality does not prove full-tensor equality. Local layer indices restart per PP stage.",
        "Prepared gradients are after unscale/preparation; clipped gradients are at the inner optimizer entry.",
        "Master deltas use FP32 Adam parameters, not rounded BF16 model weights.",
        "",
        "| Rollout | Field | Sample vectors | Relative L2 | Cosine | Max absolute difference |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rollout in (0, 1):
        for field in fields:
            subset = [r for r in rows if r["rollout"] == rollout and r["field"] == field]
            bn = sum(r["baseline_norm"] ** 2 for r in subset)
            cn = sum(r["candidate_norm"] ** 2 for r in subset)
            error = sum(r["sample_error_sq"] for r in subset)
            dot = sum(
                r["cosine"] * r["baseline_norm"] * r["candidate_norm"] for r in subset if r["cosine"] is not None
            )
            rel = f"{math.sqrt(error / bn):.6g}" if bn else "undefined"
            cosine = f"{dot / math.sqrt(bn * cn):.6g}" if bn * cn else "undefined"
            maximum = max((r["max_abs"] for r in subset), default=0)
            lines.append(f"| {rollout} | {field} | {len(subset)} | {rel} | {cosine} | {maximum:.6g} |")
    lines += [
        "",
        "## Largest sampled first-update differences",
        "",
        "| Rank | Parameter | Field | Relative L2 | Max absolute difference |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for field in ("prepared_grad", "master_delta"):
        candidates = [r for r in rows if r["rollout"] == 0 and r["field"] == field and r["relative_l2"] is not None]
        for row in sorted(candidates, key=lambda r: r["relative_l2"], reverse=True)[:10]:
            lines.append(
                f"| {row['rank']} | {row['parameter']} | {field} | {row['relative_l2']:.6g} | {row['max_abs']:.6g} |"
            )
    lines += ["", "## Coverage problems", *[f"- {issue}" for issue in issues]]
    if not rows:
        lines.append("No comparable snapshots; this is not a passing result.")
    (folder / f"{suffix}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--candidate", choices=("chunkflow", "baseline_repeat"), default="chunkflow")
    args = parser.parse_args()
    compare(args.folder, args.candidate)
