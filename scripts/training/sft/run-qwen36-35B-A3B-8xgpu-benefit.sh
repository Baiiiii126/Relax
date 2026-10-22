#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
if [ "$#" -ne 0 ] || [ "${RAY_NO_WAIT:-0}" != 0 ]; then
   echo "ERROR: this comparison runs sequentially; no arguments or RAY_NO_WAIT=1" >&2
   exit 2
fi
export EP_SIZE="${EP_SIZE:-2}"
export CHUNK_SIZE="${CHUNK_SIZE:-8192}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export TRAIN_SEED="${TRAIN_SEED:-1235}"
export CHUNKFLOW_PROFILE="${CHUNKFLOW_PROFILE:-0}"
case "${CHUNKFLOW_PROFILE}" in
   0) RUN_KIND=benefit ;;
   1) RUN_KIND=profile ;;
   *) echo "ERROR: CHUNKFLOW_PROFILE must be 0 or 1" >&2; exit 2 ;;
esac
export RELAX_CHUNKFLOW_PARENT_WINDOW="${RELAX_CHUNKFLOW_PARENT_WINDOW:-4}"
export MOE_SHARED_EXPERT_OVERLAP="${MOE_SHARED_EXPERT_OVERLAP:-0}"
export RELAX_CHUNKFLOW_KV_GRAD_FP32=0
export QWEN36_ROUTER_DTYPE="${QWEN36_ROUTER_DTYPE:-fp64}"
unset QWEN36_DIAGNOSTICS QWEN36_KV_GRAD_AB RELAX_TRAIN_PROFILE_DIR
unset RELAX_CHUNKFLOW_DIAGNOSTIC_DIR RELAX_CHUNKFLOW_GRAD_ROLLOUTS RELAX_CHUNKFLOW_LAYER_TRACE
unset RELAX_CHUNKFLOW_UPDATE_TRACE RELAX_CHUNKFLOW_CONV_TRACE RELAX_CHUNKFLOW_CONV_CHUNK_SIZE RELAX_CHUNKFLOW_TAIL_TRACE
export RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=0
export RELAX_CHUNKFLOW_FLASH_BACKEND="${RELAX_CHUNKFLOW_FLASH_BACKEND:-fa3}"
RUN_ID="qwen36-${RUN_KIND}-$(date +%Y%m%d-%H%M%S)-$$"
RESULT_DIR="$(pwd)/analysis/${RUN_ID}"
mkdir -p "${RESULT_DIR}"
cat > "${RESULT_DIR}/comparison.md" <<EOF
# Qwen3.6 baseline / ChunkFlow comparison

Commit: $(git rev-parse HEAD)
TP2 / PP4 / EP${EP_SIZE} / ETP1; chunk=${CHUNK_SIZE}; GBS=${GBS:-8}; seed=${TRAIN_SEED}; steps=${NUM_ROLLOUT}.
LR=${LEARNING_RATE:-1e-5}; GDN recompute=${GDN_RECOMPUTE:-1}; CPU optimizer overlap=${CPU_OPTIMIZER_OVERLAP:-1}.
MoE shared expert overlap: ${MOE_SHARED_EXPERT_OVERLAP} (both arms).
Router dtype: ${QWEN36_ROUTER_DTYPE:-fp32}.
ChunkFlow parent window: ${RELAX_CHUNKFLOW_PARENT_WINDOW} (maximum parents per pipeline drain; 1=previous schedule).
Data: ${PROMPT_DATA:-unset}; input key: ${INPUT_KEY:-question}; label key: ${LABEL_KEY-answer}.
Numerical probes and FP32 KV experiment disabled.
Both arms use the same data, seed and optimizer settings; only ChunkFlow is toggled.
Profiler: ${CHUNKFLOW_PROFILE}; schedule prediction is disabled. For unprofiled performance runs only, compare paired steps after the first 30 warmup steps using total-time ratios.
Both arms must be rerun for this topology; previous EP1 measurements are not this baseline.
EOF
if [ "${CHUNKFLOW_PROFILE}" = 1 ]; then
   cat > "${RESULT_DIR}/profile.md" <<EOF
# Qwen3.6 baseline / ChunkFlow profiler comparison

Configuration and completion: comparison.md. Profile steps: ${PROFILE_START:-34}..${PROFILE_END:-36} (0-based).
Profiler timing is diagnostic only; do not use it to claim speedup.
Each arm writes GPU traces (*.pt.trace.json.gz) and per-rank segments_*.jsonl below.
Host intervals include nested calls and waits; do not sum them as GPU compute time.
Use the GPU trace to distinguish GDN recomputation inside backward from the initial forward.
CUDA allocator peaks are per profiled step and include profiler effects; they exclude allocations outside PyTorch.
EOF
fi
for arm in baseline chunkflow; do
   if [ "${CHUNKFLOW_PROFILE}" = 1 ]; then
      export RELAX_TRAIN_PROFILE_DIR="${RESULT_DIR}/${arm}/train_trace"
      echo "- ${arm}: ${RELAX_TRAIN_PROFILE_DIR}" >> "${RESULT_DIR}/profile.md"
   fi
   RUN_NAME="${RUN_ID}-${arm}" LOG_FILE="${RESULT_DIR}/${arm}.log" \
      bash scripts/entrypoint/ray-job.sh \
      scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh "${arm}"
   echo "${arm}: complete" >> "${RESULT_DIR}/comparison.md"
done
if [ "${CHUNKFLOW_PROFILE}" = 0 ]; then
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/summarize_qwen36_comparison.py "${RESULT_DIR}" --expected-steps "${NUM_ROLLOUT}"
fi
echo "Comparison complete: ${RESULT_DIR}"
