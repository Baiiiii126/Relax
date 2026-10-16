# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Stage controlled ChunkFlow experiments; training runs only on explicit
invocation.

screen: baseline + 4K + 8K, 110 steps each, fixed GBS=8 by default.
confirm: one explicitly selected chunk, three A/B repeats, 510 steps each.
profile: one baseline/CF pair with a short profiler window, no speedup report.
backend: shared baseline + CF-FA2 + CF-FA3 for one explicitly selected chunk.
DRY_RUN=1 prints the resolved plan without launching or creating files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import chunkflow_align as align
import chunkflow_diagnose as diagnose


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[3]
RUN_SCRIPT = TOOLS.parent / "run-qwen3-4B-8xgpu-chunkflow.sh"
ENTRYPOINT = ROOT / "scripts/entrypoint/ray-job.sh"


def integer(env: Mapping[str, str], key: str, default: int, minimum: int = 0) -> int:
    value = env.get(key, str(default))
    if not re.fullmatch(r"[0-9]+", value) or int(value) < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def configuration(phase: str, env: Mapping[str, str]) -> dict[str, Any]:
    backend = env.get("RELAX_CHUNKFLOW_FLASH_BACKEND", "fa2")
    if backend not in ("fa2", "fa3"):
        raise ValueError("RELAX_CHUNKFLOW_FLASH_BACKEND must be fa2 or fa3")
    for key, expected in {
        "TP_SIZE": "2",
        "PP_SIZE": "4",
        "SEQ_LENGTH": "32768",
        "MAX_TOKENS_PER_GPU": "32768",
        "PACK_POLICY": "staggered",
        "KEEP_LAST_K": "0",
    }.items():
        if key in env and env[key] != expected:
            raise ValueError(f"This controlled 8-GPU experiment requires {key}={expected}; unset the old override")
    if env.get("RAY_NO_WAIT"):
        raise ValueError("unset RAY_NO_WAIT: all experiment arms must finish sequentially")
    if env.get("RELAX_ENTRYPOINT_MODE"):
        raise ValueError("unset RELAX_ENTRYPOINT_MODE before running the experiment driver")
    if env.get("WORKING_DIR"):
        raise ValueError("run from the synced server checkout with WORKING_DIR unset")
    if env.get("CHUNKFLOW_PROFILE", "0") != "0":
        raise ValueError("unset CHUNKFLOW_PROFILE; use the separate 'profile' phase")
    if phase == "screen":
        if env.get("CHUNK_SIZE"):
            raise ValueError("screen uses CHUNK_SIZES (default '4096 8192'); unset CHUNK_SIZE")
        chunks_text = env.get("CHUNK_SIZES", "4096 8192").replace(",", " ").split()
    else:
        if not env.get("CHUNK_SIZE"):
            raise ValueError(f"{phase} requires CHUNK_SIZE selected after reading the screen report")
        if env.get("CHUNK_SIZES"):
            raise ValueError(f"{phase} uses one CHUNK_SIZE; unset CHUNK_SIZES")
        chunks_text = [env["CHUNK_SIZE"]]
    if not chunks_text or any(not re.fullmatch(r"[0-9]+", value) for value in chunks_text):
        raise ValueError("chunk sizes must be positive integers")
    chunks = [int(value) for value in chunks_text]
    if len(set(chunks)) != len(chunks) or any(c < 256 or c > 32768 or c % 256 for c in chunks):
        raise ValueError("chunk sizes must be distinct multiples of 256 in [256, 32768]")
    profile_start = integer(env, "PROFILE_START", 10)
    profile_end = integer(env, "PROFILE_END", 12)
    default_steps = {"screen": 110, "confirm": 510, "profile": profile_end + 2, "backend": 110}[phase]
    steps = integer(env, "STEPS", default_steps, 1)
    drop = integer(env, "DROP_FIRST", 10)
    if phase == "profile":
        if profile_end < profile_start or steps <= profile_end + 1:
            raise ValueError("need PROFILE_START <= PROFILE_END and STEPS > PROFILE_END + 1 to flush traces")
    elif steps - drop < 20:
        raise ValueError("need at least 20 measured steps after DROP_FIRST for block intervals")
    repeats = integer(env, "REPEATS", 3 if phase == "confirm" else 1, 1)
    if phase != "confirm" and repeats != 1:
        raise ValueError("REPEATS is only supported in the confirm phase")
    if env.get("DRY_RUN", "0") not in ("0", "1"):
        raise ValueError("DRY_RUN must be 0 or 1")
    dry_run = env.get("DRY_RUN", "0") == "1"
    if not env.get("MODEL_DIR") or not (env.get("PROMPT_DATA") or env.get("DATA_DIR")):
        raise ValueError("set MODEL_DIR and DATA_DIR (or PROMPT_DATA)")
    model = Path(env["MODEL_DIR"]).expanduser().resolve() / "Qwen3-4B"
    data = Path(env.get("PROMPT_DATA") or str(Path(env["DATA_DIR"]) / "chunkflow-mixed/train.parquet"))
    data = data.expanduser().resolve()
    if not dry_run and (not model.is_dir() or not data.is_file()):
        raise ValueError(f"required paths missing: model={model}, data={data}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    report = Path(env.get("REPORT_DIR", f"analysis/chunkflow-8gpu-{phase}-{run_id}")).expanduser().resolve()
    return {
        "phase": phase,
        "run_id": run_id,
        "report_dir": str(report),
        "model": str(model),
        "data": str(data),
        "gbs": integer(env, "GBS", 8, 1),
        "chunks": chunks,
        "flash_backend": backend,
        "backends": ["fa2", "fa3"] if phase == "backend" else [backend],
        "steps": steps,
        "drop_first": drop,
        "repeats": repeats,
        "seed": integer(env, "TRAIN_SEED", 1235 if phase == "confirm" else 1234),
        "tp": 2,
        "pp": 4,
        "dp": 1,
        "cp": 1,
        "capacity": 32768,
        "sequence_length": 32768,
        "pack_policy": "staggered",
        "keep_last_k": 0,
        "keep_last_k_meaning": "retain all chunk activations; no chunk forward recomputation",
        "profile_start": profile_start if phase == "profile" else None,
        "profile_end": profile_end if phase == "profile" else None,
        "dry_run": dry_run,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def code_hashes() -> dict[str, str]:
    paths = [
        RUN_SCRIPT,
        Path(__file__).resolve(),
        ENTRYPOINT,
        TOOLS / "chunkflow_align.py",
        TOOLS / "chunkflow_diagnose.py",
    ]
    paths.extend(sorted((ROOT / "relax/backends/megatron/chunkflow").glob("*.py")))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def arm_label(config: dict[str, Any], chunk: int, backend: str) -> str:
    if not chunk:
        return "baseline"
    suffix = f"-{backend}" if config["phase"] == "backend" or backend != "fa2" else ""
    return f"chunk{chunk}{suffix}"


def arm_environment(config: dict[str, Any], repeat: int, chunk: int, backend: str | None = None) -> dict[str, str]:
    backend = backend or config["flash_backend"]
    arm = arm_label(config, chunk, backend)
    name = f"cf8-{config['phase']}-{config['run_id']}-r{repeat}-gbs{config['gbs']}-{arm}"
    return {
        "MODEL_DIR": str(Path(config["model"]).parent),
        "PROMPT_DATA": config["data"],
        "TP_SIZE": "2",
        "PP_SIZE": "4",
        "GBS": str(config["gbs"]),
        "SEQ_LENGTH": "32768",
        "MAX_TOKENS_PER_GPU": "32768",
        "PACK_POLICY": "staggered",
        "KEEP_LAST_K": "0",
        "CHUNK_SIZE": str(chunk),
        "NUM_ROLLOUT": str(config["steps"]),
        "TRAIN_SEED": str(config["seed"] + repeat - 1),
        "EXP_NAME": name,
        "RUN_NAME": name,
        "LOG_FILE": str(Path(config["report_dir"]) / f"repeat{repeat}" / f"{arm}.log"),
        "RELAX_CHUNKFLOW_PIPELINE_UNTESTED": "1",
        "RELAX_CHUNKFLOW_FLASH_BACKEND": backend,
        "RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS": "0" if config["phase"] == "profile" else "1",
        "CHUNKFLOW_PROFILE": "1" if config["phase"] == "profile" else "0",
        "PROFILE_START": str(config["profile_start"] or 0),
        "PROFILE_END": str(config["profile_end"] or 0),
    }


def validate_log(path: Path, steps: int, chunk: int, profile: bool, backend: str | None = None) -> None:
    expected = set(range(steps))
    for metric in ("actor_train_time", "step_time"):
        timings = align._perf_rows(path, metric)
        actual = set(timings)
        if actual != expected:
            raise ValueError(f"{path.name}: {metric}: expected step IDs 0..{steps - 1}, got {sorted(actual)}")
        if any(
            not math.isfinite(align._pick(row, f"perf/{metric}")) or align._pick(row, f"perf/{metric}") <= 0
            for row in timings.values()
        ):
            raise ValueError(f"{path.name}: invalid {metric} timing")
    if chunk:
        if "chunked attention installed" not in path.read_text(errors="replace"):
            raise ValueError(f"{path.name}: chunked attention was not installed")
        if backend and f"chunkflow: flash attention backend={backend}" not in path.read_text(errors="replace"):
            raise ValueError(f"{path.name}: requested attention backend {backend} not confirmed in worker logs")
        if not profile and set(align._theory(path)) != expected:
            raise ValueError(f"{path.name}: missing per-step scheduling telemetry")


def validate_inputs(base: Path, cf: Path) -> None:
    left, right = align._perf_rows(base, "actor_train_time"), align._perf_rows(cf, "actor_train_time")
    if set(left) != set(right):
        raise ValueError("A/B step IDs differ")
    for step in left:
        a = align._pick(left[step], "perf/actor_train_tokens")
        b = align._pick(right[step], "perf/actor_train_tokens")
        if a is None or b is None or a != b or not math.isfinite(a) or a <= 0:
            raise ValueError(f"step {step}: missing or different token counts ({a} != {b})")


def analyze_pair(config: dict[str, Any], repeat: int, chunk: int, backend: str | None = None) -> dict[str, Any]:
    backend = backend or config["flash_backend"]
    folder = Path(config["report_dir"]) / f"repeat{repeat}"
    label = arm_label(config, chunk, backend)
    base, cf = folder / "baseline.log", folder / f"{label}.log"
    rows = align._pair(base, cf, "actor_train_time", "schedule")
    wall_rows = align._pair(base, cf, "step_time", "schedule")
    if len(rows) != config["steps"] or len(wall_rows) != config["steps"]:
        raise ValueError("paired steps incomplete; inspect the current run's logs")
    rows, wall_rows = rows[config["drop_first"] :], wall_rows[config["drop_first"] :]
    output = folder / label
    output.mkdir()
    with (output / "attribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Reuse the standalone tools' definitions; make no cross-configuration CI.
    report = diagnose.diagnose(diagnose.read_rows(output / "attribution.csv"), 0, 1000, 10, 1234)
    save_json(output / "diagnosis.json", report)
    (output / "diagnosis.md").write_text(diagnose.markdown(report))
    (output / "scatter.svg").write_text(align._scatter_svg(rows))
    summary = report["cases"]["0"]["summary"]
    result = {
        "repeat": repeat,
        "chunk_size": chunk,
        "flash_backend": backend,
        "seed": config["seed"] + repeat - 1,
        "summary": summary,
        "block_ci95": report["cases"]["0"]["block_ci95"],
        "step_time_speedup": align._aggregate_speedup(wall_rows),
        "baseline_log": str(base),
        "chunkflow_log": str(cf),
    }
    save_json(output / "attribution.json", result)
    return result


def write_report(config: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [
        "# ChunkFlow 8 卡实验结果",
        "",
        f"阶段：{config['phase']}；Qwen3-4B；GBS={config['gbs']}；TP=2 / PP=4 / DP=1 / CP=1；",
        f"容量=32768；staggered；K=0（保留全部激活）；每组 {config['steps']} step，去掉前 {config['drop_first']}。",
        "",
        "实际加速比 = baseline 总耗时 / ChunkFlow 总耗时；大于 1 表示提速。",
        "预测使用声明成本假设的调度模型，尚未按 GPU 算子成本校准。",
        "本表可在实验过程中增量生成；完整性以 manifest.json 的 status=complete 为准。",
        "",
        "| 轮次 | chunk | CF 后端 | 配对数 | 预测加速比 | 训练加速比 | 训练比 95% block 区间 | step_time 加速比 | 实际正收益率 |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for result in results:
        s, ci = result["summary"], result["block_ci95"].get("actual_x")
        interval = "未定义" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        lines.append(
            f"| {result['repeat']} | {result['chunk_size']} | {result['flash_backend']} | {s['paired_steps']} | "
            f"{s['aggregate_theory_x']:.4f} | {s['aggregate_actual_x']:.4f} | {interval} | "
            f"{result['step_time_speedup']:.4f} | {100 * s['actual_positive_fraction']:.1f}% |"
        )
    comparison = None
    if config["phase"] == "backend" and len(results) == 2:
        pair = {r["flash_backend"]: r for r in results}
        left, right = (align._perf_rows(Path(pair[b]["chunkflow_log"]), "actor_train_time") for b in ("fa2", "fa3"))
        direct = []
        for step in sorted(left)[config["drop_first"] :]:
            a, b = align._pick(left[step], "perf/actor_train_time"), align._pick(right[step], "perf/actor_train_time")
            direct.append(
                {
                    "step": step,
                    "base_time": a,
                    "cf_time": b,
                    "theory_gain_pct": 0,
                    "actual_gain_pct": 100 * (a / b - 1),
                }
            )
        comparison = {
            "fa3_vs_fa2_x": align._aggregate_speedup(direct),
            "block_ci95": diagnose.block_intervals(direct, 1000, 10, 1234),
        }
        ci = comparison["block_ci95"].get("actual_x")
        lines += [
            "",
            f"**FA3 相对 FA2 的训练加速比：{comparison['fa3_vs_fa2_x']:.4f}**（FA2 总时间 / FA3 总时间）。",
            f"95% block 区间：{ci}。大于 1 表示 FA3 更快；是否胜过 baseline 仍看上表。",
        ]
    for result in results:
        lines += ["", f"## 轮次 {result['repeat']}，chunk={result['chunk_size']}，{result['flash_backend']}", ""]
        s = result["summary"]
        # The same renderer is used by the original attribution tool.
        table = align._markdown_report(
            {
                "theory_source": "schedule",
                "drop_first_per_case": config["drop_first"],
                "pooled": s,
                "cases": [],
                "per_case": {},
            }
        )
        lines.append(table)
    lines += [
        "",
        "四格表以 speedup=1 为界；恰好相等计为中性。diagnosis.md 还包含按 M、token 数的分组。",
        "同轮不同 chunk 共享 baseline，不能把比较次数当独立样本数。区间不能消除运行顺序或机器状态偏差。",
        "筛选阶段发现的最好结果需要在 confirm 阶段的新 seed 上复验；本脚本不会自动开启下一阶段。",
        "若没有候选提速，下一步运行 profile 阶段定位开销；无需再重复同配置的长实验。",
        "每次运行的 ClearML 链接保存在 manifest.json 的 arms[].clearml_urls。",
    ]
    folder = Path(config["report_dir"])
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    save_json(folder / "summary.json", {"configuration": config, "results": results, "backend_comparison": comparison})


def run(config: dict[str, Any]) -> None:
    arms = []
    for repeat in range(1, config["repeats"] + 1):
        chunks = [0, *config["chunks"]]
        if config["phase"] == "confirm" and repeat % 2 == 0:
            chunks.reverse()
        for chunk in chunks:
            for backend in config["backends"] if chunk else ["fa2"]:
                arms.append(
                    {
                        "repeat": repeat,
                        "chunk": chunk,
                        "backend": backend,
                        "environment": arm_environment(config, repeat, chunk, backend),
                        "status": "pending",
                    }
                )
    print(
        json.dumps({"configuration": config, "run_order": [a["environment"]["RUN_NAME"] for a in arms]}, indent=2),
        flush=True,
    )
    if config["dry_run"]:
        print("DRY_RUN: no files created, no training submitted.")
        return
    folder = Path(config["report_dir"])
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "configuration": config,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status": git_output("status", "--short"),
        "data_sha256": sha256(Path(config["data"])),
        "code_sha256": code_hashes(),
        "status": "running",
        "arms": arms,
    }
    save_json(folder / "manifest.json", manifest)
    helper = [sys.executable, str(Path(__file__).resolve())]
    fail_command = shlex.join([*helper, "_fail", "--report-dir", str(folder)])
    trap = 'status=$?; if [ "$status" -ne 0 ]; then ' + fail_command + "; fi"
    commands = [
        "#!/bin/bash",
        "# Copyright (c) 2026 Relax Authors. All Rights Reserved.",
        "set -euo pipefail",
        "cd " + shlex.quote(str(ROOT)),
        "trap " + shlex.quote(trap) + " EXIT",
    ]
    for index, arm in enumerate(arms):
        commands.append(shlex.join([*helper, "_before", "--report-dir", str(folder), "--arm", str(index)]))
        commands.append(
            shlex.join(
                [
                    "env",
                    *[f"{key}={value}" for key, value in arm["environment"].items()],
                    "bash",
                    str(ENTRYPOINT),
                    str(RUN_SCRIPT),
                    "chunkflow" if arm["chunk"] else "baseline",
                ]
            )
        )
        commands.append(shlex.join([*helper, "_after", "--report-dir", str(folder), "--arm", str(index)]))
    commands.append(shlex.join([*helper, "_finish", "--report-dir", str(folder)]))
    script = folder / "execute.sh"
    script.write_text("\n".join(commands) + "\n")
    print("Use only the dedicated, idle Ray cluster: ray-job.sh cleans old jobs and Serve state.", flush=True)
    # Replace this process: no Python orchestrator may remain alive during
    # ray-job.sh's residual-Python cleanup. Helpers run only between arms.
    os.execv("/bin/bash", ["bash", str(script)])


def update_run(action: str, folder: Path, index: int | None) -> None:
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    config = manifest["configuration"]
    arm = manifest["arms"][index] if index is not None else None
    if action == "_fail":
        manifest["status"] = "failed"
        manifest.setdefault("error", "execution interrupted or training command failed; inspect the arm log")
        for entry in manifest["arms"]:
            if entry["status"] == "running":
                entry["status"] = "failed"
        save_json(path, manifest)
        return
    try:
        if action in ("_before", "_after"):
            if arm is None:
                raise ValueError("arm index required")
            if sha256(Path(config["data"])) != manifest["data_sha256"]:
                raise ValueError("dataset changed during this experiment; results are not comparable")
            if code_hashes() != manifest["code_sha256"]:
                raise ValueError("experiment code changed during this experiment")
        if action == "_before":
            if arm["status"] != "pending" or any(a["status"] != "complete" for a in manifest["arms"][:index]):
                raise ValueError("previous arm incomplete; refusing to start another training job")
            Path(arm["environment"]["LOG_FILE"]).parent.mkdir(exist_ok=True)
            arm.update(status="running", started_utc=datetime.now(timezone.utc).isoformat())
        elif action == "_after":
            log = Path(arm["environment"]["LOG_FILE"])
            validate_log(log, config["steps"], arm["chunk"], config["phase"] == "profile", arm["backend"])
            summary_path = folder / "summary.json"
            results = json.loads(summary_path.read_text())["results"] if summary_path.exists() else []
            for candidate in config["chunks"]:
                for backend in config["backends"]:
                    base, cf = log.parent / "baseline.log", log.parent / f"{arm_label(config, candidate, backend)}.log"
                    already = any(
                        r["repeat"] == arm["repeat"] and r["chunk_size"] == candidate and r["flash_backend"] == backend
                        for r in results
                    )
                    if base.is_file() and cf.is_file() and not already:
                        validate_inputs(base, cf)
                        if config["phase"] != "profile":
                            results.append(analyze_pair(config, arm["repeat"], candidate, backend))
                            write_report(config, results)
            arm.update(
                status="complete",
                finished_utc=datetime.now(timezone.utc).isoformat(),
                clearml_urls=sorted(
                    set(re.findall(r"https?://[^\s'\"<>]+/experiments/[^\s'\"<>]+", log.read_text(errors="replace")))
                ),
            )
            if config["phase"] == "profile":
                arm["trace_path_relative_to_worker_cwd"] = f"traces/{arm['environment']['RUN_NAME']}/train_trace"
        elif action == "_finish":
            if any(a["status"] != "complete" for a in manifest["arms"]):
                raise ValueError("experiment has incomplete arms")
            manifest["status"] = "complete"
            if config["phase"] == "profile":
                (folder / "profile.md").write_text(
                    "# Profiler 诊断运行\n\n本次计时含 profiler 开销，不用于判定收益。\n\n"
                    + "\n".join(
                        f"- worker cwd 下：`{a['trace_path_relative_to_worker_cwd']}`" for a in manifest["arms"]
                    )
                    + "\n\n目录由每个训练 worker 写入；查看其中的 *.pt.trace.json.gz 文件。\n"
                )
            print(f"REPORT_DIR={folder}")
            print(f"Read {folder / ('profile.md' if config['phase'] == 'profile' else 'summary.md')}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        if arm is not None:
            arm["status"] = "failed"
        raise
    finally:
        save_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "phase", choices=["screen", "confirm", "profile", "backend", "_before", "_after", "_finish", "_fail"]
    )
    parser.add_argument("--report-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arm", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.phase.startswith("_"):
            if args.report_dir is None:
                parser.error("internal action requires --report-dir")
            update_run(args.phase, args.report_dir, args.arm)
        else:
            run(configuration(args.phase, os.environ))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
