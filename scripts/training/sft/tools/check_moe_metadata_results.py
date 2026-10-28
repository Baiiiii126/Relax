# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail closed on logged numerical differences or increased CUDA allocator
peaks.

Compare an off-mode benefit directory with an a/b-mode benefit directory. This
checks logged values, not full gradients, sample identity or external CUDA
allocations. Use the two-GPU regression and unchanged production inputs as
well.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from summarize_qwen36_comparison import number, read_rows


def memory_rows(folder: Path, mode: str, expected_steps: int, expected_ranks: int) -> dict:
    result = {}
    for path in sorted(folder.glob("memory_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != expected_steps:
            raise ValueError(f"{path}: expected {expected_steps} memory records, got {len(rows)}")
        for row in rows:
            key = (row["rank"], row["rollout_id"], row["step_id"])
            if row["moe_metadata_mode"] != mode or key in result:
                raise ValueError(f"{path}: wrong metadata mode or duplicate step {key}")
            result[key] = row
    ranks = {key[0] for key in result}
    if len(ranks) != expected_ranks:
        raise ValueError(f"{folder}: expected {expected_ranks} ranks, got {len(ranks)}; collect files from every node")
    return result


def check(reference: Path, candidate: Path, mode: str, steps: int, ranks: int, atol: float, rtol: float) -> list[str]:
    issues = []
    for arm in ("baseline", "chunkflow"):
        base, test = (read_rows(folder / f"{arm}.log") for folder in (reference, candidate))
        required = set(range(steps))
        if set(base) != required or set(test) != required:
            issues.append(f"{arm}: missing or unexpected training steps")
            continue
        for step in sorted(required):
            for metric in ("train/loss", "train/grad_norm", "perf/actor_train_tokens"):
                a, b = base[step].get(metric), test[step].get(metric)
                tolerance = (0, 0) if metric == "perf/actor_train_tokens" else (atol, rtol)
                if (
                    not number(a)
                    or not number(b)
                    or not math.isclose(a, b, abs_tol=tolerance[0], rel_tol=tolerance[1])
                ):
                    issues.append(f"{arm} step={step} {metric}: {a} -> {b}")
        try:
            bm = memory_rows(reference / arm / "memory", "off", steps, ranks)
            tm = memory_rows(candidate / arm / "memory", mode, steps, ranks)
            if bm.keys() != tm.keys():
                raise ValueError(f"{arm}: memory rank/step coverage differs")
            for key in bm:
                for field in ("peak_allocated", "peak_reserved"):
                    a, b = bm[key][field], tm[key][field]
                    if not number(a) or not number(b) or a < 0 or b < 0 or b > a:
                        issues.append(f"{arm} {key} {field}: {a} -> {b}")
                for field in ("peak_allocated_since_previous_reset", "peak_reserved_since_previous_reset"):
                    a, b = bm[key]["before"][field], tm[key]["before"][field]
                    if not number(a) or not number(b) or a < 0 or b < 0 or b > a:
                        issues.append(f"{arm} {key} before.{field}: {a} -> {b}")
        except (ValueError, KeyError) as exc:
            issues.append(str(exc))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mode", choices=("a", "b"), required=True)
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    args = parser.parse_args()
    if (
        args.expected_steps < 1
        or args.expected_ranks < 1
        or any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol))
    ):
        parser.error("Counts must be positive and tolerances finite/nonnegative")
    try:
        issues = check(
            args.reference, args.candidate, args.mode, args.expected_steps, args.expected_ranks, args.atol, args.rtol
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Missing or invalid evidence: {exc}\n")
    if issues:
        parser.exit(1, "FAILED:\n" + "\n".join(issues[:50]) + f"\nTotal issues: {len(issues)}\n")
    print(
        "PASS: logged loss/grad norm/token counts and per-rank allocator peaks. External CUDA allocations and full production gradients are not checked."
    )


if __name__ == "__main__":
    main()
