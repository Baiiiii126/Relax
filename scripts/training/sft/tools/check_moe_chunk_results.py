# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed reporting for the eight-GPU MoE chunk optimization
experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from summarize_qwen36_comparison import number, read_rows


MODES = ("off", "dedup", "expert", "both")
ARMS = ("baseline", "chunkflow")


def fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = set()
    for folder in (
        "relax/backends/megatron",
        "scripts/training/sft",
        "scripts/models",
        "docker/patch/megatron",
        "tests/backends/megatron",
    ):
        paths.update(p for p in (root / folder).rglob("*") if p.suffix in (".py", ".sh", ".patch"))
    paths.update(
        root / name for name in ("relax/utils/arguments.py", "configs/env.yaml", "scripts/entrypoint/ray-job.sh")
    )
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_run(folder: Path, mode: str, steps: int, ranks: int = 8) -> dict:
    manifest = (folder / "comparison.md").read_text()
    if f"MoE chunk optimization: {mode} " not in manifest:
        raise ValueError(f"{folder}: optimization mode missing or wrong")
    result = {}
    for arm in ARMS:
        if f"{arm}: complete" not in manifest:
            raise ValueError(f"{folder}: {arm} not complete")
        rows = read_rows(folder / f"{arm}.log")
        if set(rows) != set(range(steps)):
            raise ValueError(f"{folder}/{arm}: missing or extra steps")
        for step, row in rows.items():
            for field in (
                "train/loss",
                "train/grad_norm",
                "perf/actor_train_tokens",
                "perf/actor_train_time",
                "perf/step_time",
            ):
                if not number(row.get(field)):
                    raise ValueError(f"{folder}/{arm}: nonfinite or missing {field} at {step}")
            if min(row["perf/actor_train_time"], row["perf/step_time"], row["perf/actor_train_tokens"]) <= 0:
                raise ValueError(f"{folder}/{arm}: invalid performance denominator")
        memory = {}
        for path in (folder / arm / "memory").glob("memory_*.jsonl"):
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(records) != steps:
                raise ValueError(f"{path}: wrong memory record count")
            for row in records:
                tag = re.fullmatch(r"rank(\d+)_dp\d+_tp\d+_pp\d+", row["rank"])
                if tag is None:
                    raise ValueError(f"{path}: malformed training rank tag")
                key = (int(tag[1]), row["rollout_id"], row["step_id"])
                if (
                    key in memory
                    or row.get("moe_chunk_mode") != mode
                    or row.get("moe_metadata_mode") != "off"
                    or row.get("arm") != arm
                ):
                    raise ValueError(f"{path}: duplicate memory record or wrong mode/arm")
                values = {k: row[k] for k in ("peak_allocated", "peak_reserved")}
                values.update(
                    {
                        f"before.{k}": row["before"][k]
                        for k in ("peak_allocated_since_previous_reset", "peak_reserved_since_previous_reset")
                    }
                )
                if any(not number(v) or v < 0 for v in values.values()):
                    raise ValueError(f"{path}: invalid memory values")
                memory[key] = values
        rank_ids = {k[0] for k in memory}
        if rank_ids != set(range(ranks)) or any(
            {k[1:] for k in memory if k[0] == rank} != {(s, 0) for s in range(steps)} for rank in rank_ids
        ):
            raise ValueError(f"{folder}/{arm}: incomplete per-rank memory evidence")
        device = {}
        stamps = {}
        start = float((folder / arm / "device-memory.start").read_text())
        end = float((folder / arm / "device-memory.end").read_text())
        if not number(start) or not number(end) or end <= start:
            raise ValueError(f"{folder}/{arm}: invalid device measurement interval")
        with (folder / arm / "device-memory.csv").open() as stream:
            for row in csv.reader(stream):
                if len(row) != 3:
                    raise ValueError(f"{folder}/{arm}: malformed device memory sample")
                gpu, mib = int(row[1].strip()), float(row[2].strip())
                if not number(mib) or mib < 0:
                    raise ValueError("Invalid device memory value")
                device[gpu] = max(device.get(gpu, 0), mib)
                stamps.setdefault(gpu, []).append(
                    datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                )
        if set(device) != set(range(ranks)):
            raise ValueError(f"{folder}/{arm}: missing device-memory samples")
        for gpu, times in stamps.items():
            if (
                len(times) < 2
                or times[0] > start
                or times[-1] < end
                or any(b <= a or b - a > 2 for a, b in zip(times, times[1:]))
            ):
                raise ValueError(f"{folder}/{arm}: incomplete device-memory time coverage on GPU {gpu}")
        result[arm] = dict(rows=rows, memory=memory, device=device)
    return result


def compare(left: dict, right: dict, label: str, *, memory: bool = True) -> list[str]:
    issues = []
    if left["rows"].keys() != right["rows"].keys():
        return [f"{label}: different step coverage"]
    for field in ("train/loss", "train/grad_norm", "perf/actor_train_tokens"):
        differences = [(abs(v[field] - right["rows"][s][field]), s) for s, v in left["rows"].items()]
        delta, step = max(differences)
        if delta != 0:
            issues.append(f"{label}: {field} max_abs_diff={delta:.12g} step={step}")
    if memory:
        if left["memory"].keys() != right["memory"].keys():
            issues.append(f"{label}: different memory coverage")
        else:
            increases = [
                (b - a, key, field)
                for key, values in left["memory"].items()
                for field, a in values.items()
                if (b := right["memory"][key][field]) > a
            ]
            if increases:
                issues.append(f"{label}: {len(increases)} allocator increases; largest={max(increases)}")
        if left["device"].keys() != right["device"].keys():
            issues.append(f"{label}: different device memory coverage")
        else:
            for gpu, value in left["device"].items():
                if right["device"][gpu] > value:
                    issues.append(f"{label}: GPU {gpu} sampled memory {value} -> {right['device'][gpu]} MiB")
    return issues


def report(root: Path, phase: str, steps: int, repeats: int) -> tuple[list[str], str]:
    issues, lines = (
        [],
        [
            f"# MoE chunk experiment: {phase}",
            "",
            "Exact logged loss/grad_norm/token checks; no tolerance relaxation.",
            "Allocator checks are per rank and step. Device memory is sampled every 250 ms and can miss brief peaks.",
            "Matching logged values is not proof of full-gradient or sample-identity equality.",
            "",
            "| Repeat | Mode | Arm | Steps | Train s | Step s | Allocated GiB | Reserved GiB | Last loss |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ],
    )
    speedups = {m: [] for m in MODES if m != "off"}
    for repeat in range(repeats):
        folder = root / phase / f"repeat{repeat}"
        modes = (*MODES, "off_repeat") if phase == "pilot" else MODES
        runs = {}
        for mode in modes:
            try:
                runs[mode] = read_run(folder / mode, "off" if mode == "off_repeat" else mode, steps)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append(str(exc))
                continue
            for arm, run in runs[mode].items():
                selected = [v for s, v in run["rows"].items() if s >= (30 if phase == "formal" else 0)]
                if not selected:
                    issues.append(f"{mode}/{arm}: no timed steps")
                    continue
                avg = lambda key: sum(v[key] for v in selected) / len(selected)
                peak = lambda kind: (
                    max(
                        v[field]
                        for v in run["memory"].values()
                        for field in (f"peak_{kind}", f"before.peak_{kind}_since_previous_reset")
                    )
                    / 2**30
                )
                lines.append(
                    f"| {repeat} | {mode} | {arm} | {steps} | {avg('perf/actor_train_time'):.6f} | "
                    f"{avg('perf/step_time'):.6f} | {peak('allocated'):.4f} | {peak('reserved'):.4f} | "
                    f"{run['rows'][steps - 1]['train/loss']:.9f} |"
                )
            issues.extend(
                compare(
                    runs[mode]["baseline"],
                    runs[mode]["chunkflow"],
                    f"repeat{repeat}/{mode} baseline vs chunk",
                    memory=False,
                )
            )
        if "off" not in runs:
            continue
        for mode, run in runs.items():
            if mode == "off":
                continue
            for arm in ARMS:
                issues.extend(compare(runs["off"][arm], run[arm], f"repeat{repeat}/{mode} vs off/{arm}"))
            if phase == "formal":
                total = lambda value: sum(v["perf/step_time"] for s, v in value["rows"].items() if s >= 30)
                patch_gain = total(runs["off"]["chunkflow"]) / total(run["chunkflow"]) - 1
                chunk_gain = total(run["baseline"]) / total(run["chunkflow"]) - 1
                overall_gain = total(runs["off"]["baseline"]) / total(run["chunkflow"]) - 1
                speedups[mode].append((patch_gain, chunk_gain, overall_gain))
                lines += [
                    "",
                    f"repeat{repeat}/{mode}: throughput vs off/chunk {patch_gain:+.2%}; "
                    f"vs same-mode baseline {chunk_gain:+.2%}; vs off/baseline {overall_gain:+.2%}.",
                ]
    if phase == "formal":
        for mode, gains in speedups.items():
            if len(gains) != repeats or any(min(g) <= 0 for g in gains):
                issues.append(f"{mode}: positive end-to-end gain not reproduced in every repeat/comparison")
    lines += [
        "",
        "Status: FAILED / NOT ACCEPTED"
        if issues
        else "Status: recorded checks passed; production acceptance still requires workload replication.",
        "",
    ]
    lines.extend(f"- {issue}" for issue in issues)
    return issues, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--phase", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.fingerprint:
        print(fingerprint(args.root))
        return
    if args.steps <= 0 or args.repeats <= 0 or (args.phase == "formal" and args.steps <= 30):
        parser.error("Require positive counts and >30 steps for formal performance measurements")
    issues, markdown = report(args.root, args.phase, args.steps, args.repeats)
    output = args.root / f"{args.phase}-report.md"
    output.write_text(markdown)
    print(markdown)
    print(f"Report: {output}")
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
