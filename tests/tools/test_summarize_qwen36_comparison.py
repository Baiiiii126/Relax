# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import csv
import runpy
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
