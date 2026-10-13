# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DPO_DIR = ROOT / "scripts" / "training" / "dpo"


def test_qwen36_dpo_scripts_have_required_runtime_flags():
    for name in (
        "run-qwen36-35B-A3B-dpo-8xgpu.sh",
        "run-qwen36-35B-A3B-dpo-32xgpu.sh",
    ):
        script = (DPO_DIR / name).read_text(encoding="utf-8")
        assert "--enable-weights-backuper" in script
        assert "--use-gloo-process-groups" in script
        assert '--load "${CHECKPOINT_DIR}"' in script
        assert '--save "${CHECKPOINT_DIR}"' in script


def test_qwen36_32gpu_dpo_topology_supports_ep8():
    script = (DPO_DIR / "run-qwen36-35B-A3B-dpo-32xgpu.sh").read_text(encoding="utf-8")
    assert "--tensor-model-parallel-size 1" in script
    assert "--pipeline-model-parallel-size 2" in script
    assert "--context-parallel-size 2" in script
    assert "--expert-model-parallel-size 8" in script
    assert "--max-tokens-per-gpu 8192" in script
    assert "--sequence-parallel" not in script
    assert "TP1/PP2/CP2/DP8/EP8" in script


def test_qwen36_32gpu_dpo_propagates_worker_environment():
    script = (DPO_DIR / "run-qwen36-35B-A3B-dpo-32xgpu.sh").read_text(encoding="utf-8")
    assert 'names.update(("PYTORCH_CUDA_ALLOC_CONF", "CLEARML_CONFIG_FILE"))' in script
    assert 'env["RELAX_PROPAGATE_ENV_VARS"]' in script
