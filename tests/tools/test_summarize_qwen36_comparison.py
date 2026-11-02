# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import csv
import json
import runpy
from datetime import datetime
from pathlib import Path


def test_full_comparison_keeps_all_losses_and_uses_total_times(tmp_path: Path):
    summarize = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/summarize_qwen36_comparison.py")
    )["summarize"]
    for arm in ("baseline", "chunkflow"):
        lines = [f"url: https://example.org/{arm}"]
        for step in range(50):
            time = 10 if step % 2 else (2 if arm == "baseline" else 1)
            lines += [
                f"step {step}: {{'train/loss': {1 / (step + 1)}, 'train/grad_norm': 2}}",
                f"perf {step}: {{'perf/actor_train_time': {time}, 'perf/step_time': {time + 1}, 'perf/actor_train_tokens': 100}}",
            ]
        (tmp_path / f"{arm}.log").write_text("\n".join(lines))
    assert summarize(tmp_path, expected_steps=50)
    rows = list(csv.DictReader((tmp_path / "paired.csv").open()))
    assert len(rows) == 50 and rows[-1]["step"] == "49"
    report = (tmp_path / "diagnosis.md").read_text()
    assert "| 49 | 0.02 | 0.02 |" in report
    assert "| 0 | perf/actor_train_time | 50 | 6.0000 | 5.5000 | 1.0909x | +9.09% | +8.33% |" in report
    assert "https://example.org/baseline" in report and "https://example.org/chunkflow" in report

    path = tmp_path / "chunkflow.log"
    path.write_text("\n".join(line for line in path.read_text().splitlines() if " 37:" not in line))
    assert not summarize(tmp_path, expected_steps=50)
    rows = list(csv.DictReader((tmp_path / "paired.csv").open()))
    assert len(rows) == 50 and rows[37]["chunkflow:train/loss"] == ""
    report = (tmp_path / "diagnosis.md").read_text()
    assert "refused speedup; missing/invalid timing at [37]" in report
    assert "| 30 | perf/actor_train_time" not in report


def test_execution_comparison_separates_patch_gain_and_keeps_loss_differences(tmp_path: Path):
    summarize = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/summarize_qwen36_comparison.py")
    )["summarize"]
    for arm, timing, loss in (("baseline", 10, 1.0), ("chunkflow", 12, 1.01), ("chunkflow_exec", 11, 1.02)):
        row = {
            "train/loss": loss,
            "perf/actor_train_time": timing,
            "perf/step_time": timing,
            "perf/actor_train_tokens": 100,
        }
        (tmp_path / f"{arm}.log").write_text("".join(f"step {step}: {row}\n" for step in range(40)))
    for reference, expected in (("baseline", "0.9091x"), ("chunkflow", "1.0909x")):
        prefix = f"execution-vs-{reference}-"
        assert summarize(tmp_path, 40, baseline_arm=reference, chunkflow_arm="chunkflow_exec", output_prefix=prefix)
        assert expected in (tmp_path / f"{prefix}diagnosis.md").read_text()
        with (tmp_path / f"{prefix}paired.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 40 and rows[-1]["chunkflow:train/loss"] == "1.02"


def test_memory_report_keeps_rank_peaks_and_missing_measurements(tmp_path: Path):
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/summarize_qwen36_comparison.py")
    )
    folder = tmp_path / "expert_chunkflow"
    (folder / "memory").mkdir(parents=True)
    for rank in range(2):
        row = {
            "rank": f"rank{rank}",
            "peak_allocated": 1024**3 * (rank + 1),
            "peak_reserved": 3 * 1024**3,
            "before": {"peak_allocated_since_previous_reset": 4 * 1024**3 if rank == 0 else 0},
        }
        (folder / "memory" / f"memory_rank{rank}.jsonl").write_text(json.dumps(row) + "\n")
    timestamp = "2026/11/02 16:00:00.000"
    seconds = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S.%f").timestamp()
    (folder / "device-memory.start").write_text(str(seconds - 1))
    (folder / "device-memory.end").write_text(str(seconds + 1))
    (folder / "device-memory.csv").write_text(f"{timestamp}, 0, 5120\n{timestamp}, 1, 4096\n")
    lines = module["memory_report"](tmp_path, ("expert_baseline", "expert_chunkflow"), "experts-")
    assert "| expert_baseline | 0 | unmeasured | unmeasured | 0 | unmeasured |" in lines
    assert "| expert_chunkflow | 2 | 4.0000 | 3.0000 | 2 | 5.0000 |" in lines
    rows = list(csv.DictReader((tmp_path / "experts-memory.csv").open()))
    assert len(rows) == 4


def test_missing_arm_log_does_not_produce_speedup(tmp_path: Path):
    summarize = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/summarize_qwen36_comparison.py")
    )["summarize"]
    assert not summarize(
        tmp_path, 40, baseline_arm="expert_baseline", chunkflow_arm="expert_chunkflow", output_prefix="experts-"
    )
    report = (tmp_path / "experts-diagnosis.md").read_text()
    assert "Matched step IDs: 0" in report and "| 30 | perf/actor_train_time" not in report
