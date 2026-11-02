#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Matched TE/tiles x baseline/ChunkFlow; no precision or performance gates.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
MODE="${1:-run}"
if [ "$#" -gt 1 ] || { [ "${MODE}" != run ] && [ "${MODE}" != profile ]; }; then
   echo "Usage: bash $0 {run|profile}" >&2; exit 2
fi
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the existing model directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the existing LongAlign file}"
test -f "${HF_CHECKPOINT}/config.json"
test -f "${PROMPT_DATA}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export GDN_RECOMPUTE="${GDN_RECOMPUTE:-1}"
EXPERT_BACKEND="${EXPERT_BACKEND:-mixed}"
DISPATCH_OVERLAP="${DISPATCH_OVERLAP:-1}"
case "${EXPERT_BACKEND}" in tiles128|tiles256|mixed) ;; *) echo "Invalid EXPERT_BACKEND" >&2; exit 2 ;; esac
case "${DISPATCH_OVERLAP}" in 0|1) ;; *) echo "Invalid DISPATCH_OVERLAP" >&2; exit 2 ;; esac
case "${GDN_RECOMPUTE}" in 0|1) ;; *) echo "Invalid GDN_RECOMPUTE" >&2; exit 2 ;; esac
if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || [ "${NUM_ROLLOUT}" -le 30 ]; then
   echo "ERROR: NUM_ROLLOUT must exceed the 30-step warmup" >&2; exit 2
fi
export GBS=8 NUM_GPUS=8 TP_SIZE=2 PP_SIZE=4 EP_SIZE=2
export CHUNK_SIZE=8192 KEEP_LAST_K=0 PACK_POLICY=staggered
export SEQ_LENGTH=32768 MAX_TOKENS_PER_GPU=32768 TRAIN_SEED=1235
export LEARNING_RATE=1e-5 CPU_OPTIMIZER_OVERLAP=1 MOE_SHARED_EXPERT_OVERLAP=0
export QWEN36_ROUTER_DTYPE=fp64 RELAX_CHUNKFLOW_FLASH_BACKEND=fa3
export RELAX_CHUNKFLOW_PARENT_WINDOW=0 RELAX_CHUNKFLOW_KV_GRAD_FP32=0
export RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=0 RAY_NO_WAIT=0
export RELAX_CHUNKFLOW_EXECUTION_OPT="${RELAX_CHUNKFLOW_EXECUTION_OPT:-0}"
export INPUT_KEY="${INPUT_KEY:-messages}" LABEL_KEY="${LABEL_KEY-}"
export CHUNKFLOW_PROFILE=0 RELAX_MOE_CAPTURE_LIMIT=0
if [ "${MODE}" = profile ]; then
   export CHUNKFLOW_PROFILE=1 RELAX_MOE_CAPTURE_LIMIT="${CAPTURE_LIMIT:-32}"
   export PROFILE_START="${PROFILE_START:-34}" PROFILE_END="${PROFILE_END:-36}"
fi
unset RELAX_ENTRYPOINT_MODE TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG CUDA_LAUNCH_BLOCKING

RUN_ID="qwen36-expert-tiles-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
RESULT_DIR="${QWEN36_RESULT_DIR:-$(pwd)/analysis/${RUN_ID}}"
if [ -e "${RESULT_DIR}" ]; then echo "ERROR: result directory exists: ${RESULT_DIR}" >&2; exit 2; fi
mkdir -p "${RESULT_DIR}" analysis
printf '%s\n' "${RESULT_DIR}" > analysis/expert-tiles-latest-path.txt
ARMS=(baseline chunkflow expert_baseline expert_chunkflow)
case "${EXPERIMENT_ARMS:-all}" in
   all) ;;
   optimized) ARMS=(expert_baseline expert_chunkflow) ;;
   *) echo "ERROR: EXPERIMENT_ARMS must be all or optimized" >&2; exit 2 ;;
esac
cat > "${RESULT_DIR}/comparison.md" <<REPORT
# Expert tiles training comparison

Commit: $(git rev-parse HEAD)
Arms: ${ARMS[*]}; steps=${NUM_ROLLOUT}; seed=${TRAIN_SEED}.
TP2/PP4/EP2/ETP1; GBS8; sequence=32768; chunk=8192.
All arms: BF16 / FP32 main_grad / FP64 router; GDN recompute=${GDN_RECOMPUTE}.
CPU optimizer offload and overlap on; shared expert overlap off.
ChunkFlow execution patch=${RELAX_CHUNKFLOW_EXECUTION_OPT} in both ChunkFlow arms.
Expert arms: backend=${EXPERT_BACKEND}; local-first dispatch=${DISPATCH_OVERLAP}.
No numerical, speed or memory gates. Missing steps or failed installation are failures.
Profile=${CHUNKFLOW_PROFILE}; capture limit=${RELAX_MOE_CAPTURE_LIMIT} per profiled step/rank.
Profile runs include overhead and must not be used as production speed measurements.
Timing reports discard the first 30 steps; inspect loss and peak-memory CSVs separately.
Local-first overlaps only forward dispatch with local-source FC1; CPU split sync and combine remain.
It adds one FC1 launch; it does not reduce remote payload bytes or guarantee peak-memory savings.
Data: ${PROMPT_DATA}; checkpoint: ${HF_CHECKPOINT}; input=${INPUT_KEY}; label=${LABEL_KEY}.
REPORT
echo "${#ARMS[@]} arms, ${NUM_ROLLOUT} steps each. Logs stream here and remain in ${RESULT_DIR}."
failed=0
for arm in "${ARMS[@]}"; do
   mode=baseline
   export RELAX_MOE_EXPERT_BACKEND=te RELAX_MOE_DISPATCH_OVERLAP=0
   case "${arm}" in *chunkflow) mode=chunkflow ;; esac
   case "${arm}" in expert_*) export RELAX_MOE_EXPERT_BACKEND="${EXPERT_BACKEND}" RELAX_MOE_DISPATCH_OVERLAP="${DISPATCH_OVERLAP}" ;; esac
   unset RELAX_TRAIN_PROFILE_DIR RELAX_TRAIN_MEMORY_DIR RELAX_DEVICE_MEMORY_DIR
   if [ "${MODE}" = profile ]; then
      export RELAX_TRAIN_PROFILE_DIR="${RESULT_DIR}/${arm}/train_trace"
   else
      export RELAX_TRAIN_MEMORY_DIR="${RESULT_DIR}/${arm}/memory" RELAX_DEVICE_MEMORY_DIR="${RESULT_DIR}/${arm}"
   fi
   echo "START ${arm}: backend=${RELAX_MOE_EXPERT_BACKEND}, overlap=${RELAX_MOE_DISPATCH_OVERLAP}, recompute=${GDN_RECOMPUTE}"
   if RUN_NAME="${RUN_ID}-${arm}" LOG_FILE="${RESULT_DIR}/${arm}.log" \
      bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh "${mode}"; then
      echo "${arm}: complete" | tee -a "${RESULT_DIR}/comparison.md"
   else
      status=$?
      echo "${arm}: failed (${status}); see ${arm}.log" | tee -a "${RESULT_DIR}/comparison.md"
      failed=1
      if [ "${status}" -eq 130 ] || [ "${status}" -eq 143 ]; then exit "${status}"; fi
   fi
done
if [ "${MODE}" = run ]; then
   REPORT=scripts/training/sft/tools/summarize_qwen36_comparison.py
   python3 "${REPORT}" "${RESULT_DIR}" --expected-steps "${NUM_ROLLOUT}" \
      --baseline-arm expert_baseline --chunkflow-arm expert_chunkflow --output-prefix experts- || failed=1
   if [ "${EXPERIMENT_ARMS:-all}" = all ]; then
      python3 "${REPORT}" "${RESULT_DIR}" --expected-steps "${NUM_ROLLOUT}" || failed=1
      python3 "${REPORT}" "${RESULT_DIR}" --expected-steps "${NUM_ROLLOUT}" \
         --baseline-arm baseline --chunkflow-arm expert_chunkflow --output-prefix production- || failed=1
   fi
fi
echo "Results: ${RESULT_DIR}; failed=${failed}"
exit "${failed}"
