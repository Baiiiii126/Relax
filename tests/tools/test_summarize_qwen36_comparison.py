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


def test_tail_pool_comparison_exports_execution_and_refuses_missing_evidence(tmp_path: Path):
    summarize = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/summarize_qwen36_comparison.py")
    )["summarize"]
    for arm, timing in (("baseline", 12), ("chunkflow", 10), ("chunkflow_pool", 8)):
        row = {
            "train/loss": 1.0,
            "perf/actor_train_time": timing,
            "perf/step_time": timing + 1,
            "perf/actor_train_tokens": 100,
        }
        if arm == "chunkflow_pool":
            row.update(
                {
                    "train/chunkflow/tail_pool_enabled": 1,
                    "train/chunkflow/unpooled_units": 4,
                    "train/chunkflow/num_units": 3,
                    "train/chunkflow/tail_pool_saved_units": 1,
                }
            )
        (tmp_path / f"{arm}.log").write_text(f"step 0: {row}\n")
    assert summarize(tmp_path, 1, baseline_arm="chunkflow", chunkflow_arm="chunkflow_pool", output_prefix="tail_pool-")
    report = (tmp_path / "tail_pool-diagnosis.md").read_text()
    assert "1.2500x" in report and "## Tail pool execution" in report
    row = next(csv.DictReader((tmp_path / "tail_pool-paired.csv").open()))
    assert row["chunkflow:train/chunkflow/tail_pool_saved_units"] == "1"
    path = tmp_path / "chunkflow_pool.log"
    path.write_text(
        path.read_text().replace("'train/chunkflow/tail_pool_enabled': 1", "'train/chunkflow/tail_pool_enabled': 0")
    )
    assert not summarize(tmp_path, 1, chunkflow_arm="chunkflow_pool", output_prefix="tail_pool-")


def test_chunk_sweep_reuses_baseline_and_reports_regressions(tmp_path: Path):
    sweep = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/summarize_qwen36_comparison.py")
    )["summarize_sweep"]
    for arm in ("baseline", "chunk4096", "chunk8192"):
        lines = []
        for step in range(50):
            train_time = 10 if step % 2 else {"baseline": 2, "chunk4096": 1, "chunk8192": 4}[arm]
            row = {
                "train/loss": 1.0 + (0.01 if arm == "chunk8192" else 0),
                "perf/actor_train_time": train_time,
                "perf/step_time": train_time + 1,
                "perf/actor_train_tokens": 100,
            }
            lines.append(f"step {step}: {row}")
        (tmp_path / f"{arm}.log").write_text("\n".join(lines))
    assert not sweep(tmp_path, [4096, 8192, 16384], 50)
    rows = list(csv.DictReader((tmp_path / "sweep.csv").open()))
    assert [r["chunk"] for r in rows] == ["4096", "4096", "8192", "8192", "16384", "16384"]
    assert float(rows[0]["train_speedup"]) == 6 / 5.5
    assert float(rows[0]["step_speedup"]) == 7 / 6.5
    assert float(rows[2]["train_speedup"]) == 6 / 7
    assert float(rows[2]["train_slower_fraction"]) == 0.5
    assert rows[4]["status"] == "pending"
    assert len(list(csv.DictReader((tmp_path / "chunk8192-paired.csv").open()))) == 50
    assert sweep(tmp_path, [4096, 8192], 50)

    path = tmp_path / "chunk4096.log"
    path.write_text(path.read_text().replace("'perf/actor_train_tokens': 100", "'perf/actor_train_tokens': 99"))
    assert not sweep(tmp_path, [4096, 8192], 50)
    rows = list(csv.DictReader((tmp_path / "sweep.csv").open()))
    assert rows[0]["train_speedup"] == "" and rows[0]["status"].startswith("invalid")
    assert rows[2]["status"] == "complete"


def test_gate_report_checks_decisions_and_keeps_all_100_steps(tmp_path: Path):
    summarize = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/training/sft/tools/summarize_qwen36_comparison.py")
    )["summarize"]
    for arm in ("baseline", "chunkflow_gate"):
        lines = []
        for step in range(100):
            row = {
                "train/loss": 1.0,
                "perf/actor_train_time": 10,
                "perf/step_time": 12,
                "perf/actor_train_tokens": 100,
            }
            if arm == "chunkflow_gate":
                row.update(
                    {
                        "train/chunkflow/" + key: value
                        for key, value in {
                            "gate_selected": step % 2,
                            "gate_predicted_speedup": 1.3 if step % 2 else 1.2,
                            "gate_threshold_gain_pct": 20,
                            "gate_prediction_time_s": 0.002,
                            "gate_consensus_time_s": 0.003,
                        }.items()
                    }
                )
            lines.append(f"step {step}: {row}")
        (tmp_path / f"{arm}.log").write_text("\n".join(lines))
    assert summarize(tmp_path, 100, chunkflow_arm="chunkflow_gate", output_prefix="gate-")
    report = (tmp_path / "gate-diagnosis.md").read_text()
    assert "| 0 | 100 | 50 | 50 | 50.0% | 5.000 |" in report
    assert "| 50 | perf/actor_train_time | 50 |" in report
    rows = list(csv.DictReader((tmp_path / "gate-paired.csv").open()))
    assert len(rows) == 100 and rows[-1]["step"] == "99"
    assert rows[0]["chunkflow:train/chunkflow/gate_selected"] == "0"
    path = tmp_path / "chunkflow_gate.log"
    path.write_text(
        path.read_text().replace("'train/chunkflow/gate_selected': 0", "'train/chunkflow/gate_selected': 1", 1)
    )
    assert not summarize(tmp_path, 100, chunkflow_arm="chunkflow_gate", output_prefix="gate-")
    assert "Missing/inconsistent gate decisions at steps [0]" in (tmp_path / "gate-diagnosis.md").read_text()
