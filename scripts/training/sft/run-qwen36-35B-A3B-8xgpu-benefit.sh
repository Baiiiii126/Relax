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
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export TRAIN_SEED="${TRAIN_SEED:-1235}"
export CHUNKFLOW_PROFILE=0
export RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=0
export RELAX_CHUNKFLOW_FLASH_BACKEND="${RELAX_CHUNKFLOW_FLASH_BACKEND:-fa3}"
RUN_ID="qwen36-benefit-$(date +%Y%m%d-%H%M%S)-$$"
RESULT_DIR="$(pwd)/analysis/${RUN_ID}"
mkdir -p "${RESULT_DIR}"
cat > "${RESULT_DIR}/comparison.md" <<EOF
# Qwen3.6 baseline / ChunkFlow comparison

Commit: $(git rev-parse HEAD)
TP2 / PP4 / EP${EP_SIZE} / ETP1; chunk=${CHUNK_SIZE}; GBS=${GBS:-8}; seed=${TRAIN_SEED}; steps=${NUM_ROLLOUT}.
LR=${LEARNING_RATE:-1e-5}; GDN recompute=${GDN_RECOMPUTE:-1}; CPU optimizer overlap=${CPU_OPTIMIZER_OVERLAP:-1}.
Router dtype: ${QWEN36_ROUTER_DTYPE:-fp32}.
Data: ${PROMPT_DATA:-unset}; input key: ${INPUT_KEY:-question}; label key: ${LABEL_KEY-answer}.
Update trace: ${RELAX_CHUNKFLOW_UPDATE_TRACE:-0}; convolution trace: ${RELAX_CHUNKFLOW_CONV_TRACE:-0}.
Diagnostic timing is not performance evidence.
Both arms use the same data and training configuration; only ChunkFlow is toggled.
Profiler and schedule prediction are disabled. Compare paired steps after the first 30 warmup steps using total-time ratios.
Both arms must be rerun for this topology; previous EP1 measurements are not this baseline.
EOF
arms=(baseline chunkflow)
if [ "${RELAX_CHUNKFLOW_UPDATE_TRACE:-0}" = 1 ] || [ "${RELAX_CHUNKFLOW_CONV_TRACE:-0}" = 1 ]; then
   arms=(baseline baseline_repeat chunkflow)
fi
for arm in "${arms[@]}"; do
   mode="${arm}"
   if [ "${mode}" = baseline_repeat ]; then mode=baseline; fi
   if [ "${QWEN36_DIAGNOSTICS:-0}" = 1 ]; then
      export RELAX_CHUNKFLOW_DIAGNOSTIC_DIR="${RESULT_DIR}/${arm}/diagnostics"
   else
      unset RELAX_CHUNKFLOW_DIAGNOSTIC_DIR RELAX_CHUNKFLOW_GRAD_ROLLOUTS RELAX_CHUNKFLOW_LAYER_TRACE RELAX_CHUNKFLOW_UPDATE_TRACE RELAX_CHUNKFLOW_CONV_TRACE RELAX_CHUNKFLOW_CONV_CHUNK_SIZE
   fi
   RUN_NAME="${RUN_ID}-${arm}" LOG_FILE="${RESULT_DIR}/${arm}.log" \
      bash scripts/entrypoint/ray-job.sh \
      scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh "${mode}"
   echo "${arm}: complete" >> "${RESULT_DIR}/comparison.md"
done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/summarize_qwen36_comparison.py "${RESULT_DIR}"
if [ "${QWEN36_DIAGNOSTICS:-0}" = 1 ]; then
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/compare_qwen36_gradients.py "${RESULT_DIR}" --rollouts "${RELAX_CHUNKFLOW_GRAD_ROLLOUTS:-0,1,2,8}"
   if [ "${RELAX_CHUNKFLOW_CONV_TRACE:-0}" = 1 ]; then
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/compare_qwen36_convolutions.py "${RESULT_DIR}"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/compare_qwen36_convolutions.py "${RESULT_DIR}" --candidate baseline_repeat
   fi
   if [ "${RELAX_CHUNKFLOW_UPDATE_TRACE:-0}" = 1 ]; then
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/compare_qwen36_updates.py "${RESULT_DIR}"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/compare_qwen36_updates.py "${RESULT_DIR}" --candidate baseline_repeat
   fi
   if [ "${RELAX_CHUNKFLOW_LAYER_TRACE:-0}" = 1 ]; then
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/compare_qwen36_layers.py "${RESULT_DIR}"
   fi
fi
echo "Comparison complete: ${RESULT_DIR}"
