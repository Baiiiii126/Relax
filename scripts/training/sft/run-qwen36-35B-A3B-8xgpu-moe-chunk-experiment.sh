#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One command, eight GPUs throughout. No two-GPU prerequisite.
# run: repeated production matrix; inspect curves without acceptance gates.
# check/pilot are optional, explicitly requested diagnostic runs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
PHASE="${1:-run}"
case "${PHASE}" in
   run|check|pilot) ;;
   *) echo "Usage: bash $0 [run|check|pilot]" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then echo "ERROR: too many arguments" >&2; exit 2; fi
if [ "${PHASE}" != check ]; then
   : "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the existing Qwen3.6 checkpoint}"
   : "${PROMPT_DATA:?Set PROMPT_DATA to the existing LongAlign JSONL file}"
   test -f "${HF_CHECKPOINT}/config.json"
   test -f "${PROMPT_DATA}"
fi
STEPS="${NUM_ROLLOUT:-100}"
REPEATS="${EXPERIMENT_REPEATS:-3}"
PILOT_STEPS="${PILOT_STEPS:-5}"
for value in "${STEPS}" "${REPEATS}" "${PILOT_STEPS}"; do
   if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: counts must be positive integers" >&2; exit 2; fi
done
if [ "${STEPS}" -le 30 ] || [ "${REPEATS}" -lt 3 ] || [ "${PILOT_STEPS}" -lt 2 ]; then
   echo "ERROR: require NUM_ROLLOUT>30, EXPERIMENT_REPEATS>=3, PILOT_STEPS>=2" >&2
   exit 2
fi
export GBS=8 NUM_GPUS=8 TP_SIZE=2 PP_SIZE=4 EP_SIZE=2
export CHUNK_SIZE=8192 CHUNK_SIZES='' KEEP_LAST_K=0 PACK_POLICY=staggered
export SEQ_LENGTH=32768 MAX_TOKENS_PER_GPU=32768 TRAIN_SEED=1235
export LEARNING_RATE=1e-5 GDN_RECOMPUTE=1 CPU_OPTIMIZER_OVERLAP=1 MOE_SHARED_EXPERT_OVERLAP=0
export QWEN36_ROUTER_DTYPE=fp64 RELAX_MOE_METADATA_MODE=off RELAX_MOE_CHUNK_MODE=off
export RELAX_CHUNKFLOW_FLASH_BACKEND=fa3 RELAX_CHUNKFLOW_PARENT_WINDOW=0
export QWEN36_TAIL_POOL_AB=0 QWEN36_REVERSE_ORDER=0 CHUNKFLOW_PROFILE=0
export QWEN36_MEMORY_CHECK=1 QWEN36_DEVICE_MEMORY_CHECK=1 RAY_NO_WAIT=0
# Training records memory without acceptance gates; standalone check can be strict.
if [ "${PHASE}" = check ]; then
   export RELAX_MOE_MEMORY_POLICY="${RELAX_MOE_MEMORY_POLICY:-report}"
else
   export RELAX_MOE_MEMORY_POLICY=report
fi
case "${RELAX_MOE_MEMORY_POLICY}" in
   strict|report) ;;
   *) echo "ERROR: RELAX_MOE_MEMORY_POLICY must be strict or report" >&2; exit 2 ;;
esac
export INPUT_KEY="${INPUT_KEY:-messages}" LABEL_KEY="${LABEL_KEY-}"
unset QWEN36_GATE_GAIN_PCT RELAX_CHUNKFLOW_GATE_GAIN_PCT
unset TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG CUDA_LAUNCH_BLOCKING

SUITE_DIR="${MOE_EXPERIMENT_DIR:-$(pwd)/analysis/qwen36-moe-chunk-$(date +%Y%m%d-%H%M%S)-$$}"
if [ -e "${SUITE_DIR}" ]; then echo "ERROR: result directory exists: ${SUITE_DIR}" >&2; exit 2; fi
mkdir -p "${SUITE_DIR}"
SUITE_DIR="$(cd "${SUITE_DIR}" && pwd)"
export SUITE_DIR
export MOE_EXPERIMENT_PHASE="${PHASE}"
REPORT="scripts/training/sft/tools/check_moe_chunk_results.py"
python3 "${REPORT}" "$(pwd)" --fingerprint > "${SUITE_DIR}/source.sha256"
python3 - <<'PY'
import importlib.metadata as md
import json
import os
import subprocess
from pathlib import Path
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("This script requires exactly 8 visible NVIDIA GPUs")
if md.version("transformer-engine") != "2.14.1":
    raise SystemExit("Expected Transformer Engine 2.14.1")
keys = ("HF_CHECKPOINT", "PROMPT_DATA", "GBS", "TP_SIZE", "PP_SIZE", "EP_SIZE", "CHUNK_SIZE", "TRAIN_SEED",
        "SEQ_LENGTH", "MAX_TOKENS_PER_GPU", "QWEN36_ROUTER_DTYPE", "LEARNING_RATE", "GDN_RECOMPUTE",
        "CPU_OPTIMIZER_OVERLAP", "MOE_SHARED_EXPERT_OVERLAP", "RELAX_CHUNKFLOW_FLASH_BACKEND", "INPUT_KEY", "LABEL_KEY",
        "RELAX_MOE_MEMORY_POLICY")
record = {k: os.environ.get(k) for k in keys}
record.update(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              observe_only=True,
              torch=torch.__version__, te=md.version("transformer-engine"), triton=md.version("triton"),
              gpus=[torch.cuda.get_device_name(i) for i in range(8)])
if os.environ["MOE_EXPERIMENT_PHASE"] != "check":
    import ray
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)
    try:
        nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
        if (len(nodes) != 1 or nodes[0]["Resources"]["GPU"] != 8
                or nodes[0]["NodeManagerAddress"] != ray.util.get_node_ip_address()):
            raise SystemExit("Device memory measurements require a single local Ray GPU node with exactly 8 GPUs")
        record["ray_gpu_node"] = nodes[0]["NodeManagerAddress"]
    finally:
        ray.shutdown()
(Path(os.environ["SUITE_DIR"]) / "config.json").write_text(json.dumps(record, indent=2) + "\n")
PY
echo "Experiment directory: ${SUITE_DIR}"
echo "Training comparisons are observational: loss, memory and performance differences do not block."
if [ "${PHASE}" = check ]; then
export RELAX_MOE_GATE_DIR="${SUITE_DIR}/operator-gate"
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   timeout --signal=TERM --kill-after=30s 1800s \
   torchrun --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 \
   -m pytest -x -vv -s -o faulthandler_timeout=300 tests/backends/megatron/test_moe_chunk_cuda.py \
   2>&1 | tee "${SUITE_DIR}/operator-gate.log"
python3 - <<'PY'
import json, os
from pathlib import Path
folder = Path(os.environ["RELAX_MOE_GATE_DIR"])
for rank in range(8):
    record = json.loads((folder / f"rank{rank}.json").read_text())
    if record.get("rank") != rank or record.get("passed") is not True:
        raise SystemExit(f"Operator gate missing or failed on rank {rank}")
    if record.get("memory_policy") != os.environ["RELAX_MOE_MEMORY_POLICY"]:
        raise SystemExit(f"Operator gate memory policy mismatch on rank {rank}")
PY
echo "Eight-GPU operator check passed: ${SUITE_DIR}"
exit 0
fi

run_mode() {
   local phase="$1" repeat="$2" name="$3" steps="$4" mode="$3"
   if [ "${name}" = off_repeat ]; then mode=off; fi
   if [ "$(python3 "${REPORT}" "$(pwd)" --fingerprint)" != "$(cat "${SUITE_DIR}/source.sha256")" ]; then
      echo "ERROR: source changed during the experiment" >&2; exit 1
   fi
   echo "START phase=${phase} repeat=${repeat} mode=${name} steps=${steps}"
   NUM_ROLLOUT="${steps}" RELAX_MOE_CHUNK_MODE="${mode}" \
      QWEN36_RESULT_DIR="${SUITE_DIR}/${phase}/repeat${repeat}/${name}" \
      QWEN36_REVERSE_ORDER="$((repeat % 2))" \
      bash scripts/training/sft/run-qwen36-35B-A3B-8xgpu-benefit.sh
   if [ "$(python3 "${REPORT}" "$(pwd)" --fingerprint)" != "$(cat "${SUITE_DIR}/source.sha256")" ]; then
      echo "ERROR: source changed during training" >&2; exit 1
   fi
}

if [ "${PHASE}" = pilot ]; then
   for mode in off off_repeat dedup expert both; do
      run_mode pilot 0 "${mode}" "${PILOT_STEPS}"
   done
   python3 "${REPORT}" "${SUITE_DIR}" --phase pilot --steps "${PILOT_STEPS}" --repeats 1 --observe-only
   exit 0
fi

for ((repeat=0; repeat<REPEATS; repeat++)); do
   case "$((repeat % 3))" in
      0) modes=(off dedup expert both) ;;
      1) modes=(both expert dedup off) ;;
      2) modes=(expert off both dedup) ;;
   esac
   for mode in "${modes[@]}"; do
      run_mode formal "${repeat}" "${mode}" "${STEPS}"
   done
done
python3 "${REPORT}" "${SUITE_DIR}" --phase formal --steps "${STEPS}" --repeats "${REPEATS}" --observe-only
