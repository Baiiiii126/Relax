#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Two eight-GPU runs with GDN recomputation disabled in both arms.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
if [ "$#" -ne 0 ]; then echo "Usage: bash $0" >&2; exit 2; fi
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the existing model directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the existing LongAlign file}"
test -f "${HF_CHECKPOINT}/config.json"
test -f "${PROMPT_DATA}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || [ "${NUM_ROLLOUT}" -le 30 ]; then
   echo "ERROR: NUM_ROLLOUT must exceed 30 warmup steps" >&2
   exit 2
fi
export CHUNKFLOW_PROFILE="${CHUNKFLOW_PROFILE:-0}"
export QWEN36_REVERSE_ORDER="${QWEN36_REVERSE_ORDER:-0}"
for key in CHUNKFLOW_PROFILE QWEN36_REVERSE_ORDER; do
   case "${!key}" in
      0|1) ;;
      *) echo "ERROR: ${key} must be 0 or 1" >&2; exit 2 ;;
   esac
done

export GBS=8 NUM_GPUS=8 TP_SIZE=2 PP_SIZE=4 EP_SIZE=2
export CHUNK_SIZE=8192 KEEP_LAST_K=0 PACK_POLICY=staggered
export SEQ_LENGTH=32768 MAX_TOKENS_PER_GPU=32768 TRAIN_SEED=1235
export LEARNING_RATE=1e-5 GDN_RECOMPUTE=0 CPU_OPTIMIZER_OVERLAP=1 MOE_SHARED_EXPERT_OVERLAP=0
export QWEN36_ROUTER_DTYPE=fp64 RELAX_CHUNKFLOW_FLASH_BACKEND=fa3
export RELAX_CHUNKFLOW_PARENT_WINDOW=0
export RELAX_CHUNKFLOW_KV_GRAD_FP32=0 RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=0
export RAY_NO_WAIT=0
export INPUT_KEY="${INPUT_KEY:-messages}" LABEL_KEY="${LABEL_KEY-}"
unset RELAX_ENTRYPOINT_MODE TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG CUDA_LAUNCH_BLOCKING

RUN_ID="qwen36-chunkflow-norecompute-$(date +%Y%m%d-%H%M%S)-$$"
RESULT_DIR="${QWEN36_RESULT_DIR:-$(pwd)/analysis/${RUN_ID}}"
if [ -e "${RESULT_DIR}" ]; then echo "ERROR: result directory exists: ${RESULT_DIR}" >&2; exit 2; fi
mkdir -p "${RESULT_DIR}"
ARMS=(baseline chunkflow_exec)
if [ "${QWEN36_REVERSE_ORDER}" = 1 ]; then ARMS=(chunkflow_exec baseline); fi
cat > "${RESULT_DIR}/comparison.md" <<REPORT
# Qwen3.6 ChunkFlow comparison without GDN recomputation

Commit: $(git rev-parse HEAD)
Arms: ${ARMS[*]}; steps per arm: ${NUM_ROLLOUT}; seed: ${TRAIN_SEED}.
TP2 / PP4 / CP1 / EP2 / ETP1; GBS8; sequence=32768; chunk=8192; K=0.
LR=1e-5; BF16; router=FP64; GDN recompute=0 in both arms; CPU optimizer offload and overlap enabled.
Shared expert overlap and FP32 KV accumulation disabled.
Data: ${PROMPT_DATA}; checkpoint: ${HF_CHECKPOINT}; input=${INPUT_KEY}; label=${LABEL_KEY}.
Only chunkflow_exec enables the execution patch. Nonblocking host copies can still incur staging waits.
Profiler=${CHUNKFLOW_PROFILE}; profile steps=${PROFILE_START:-34}..${PROFILE_END:-36} (0-based).
Profiler timings include instrumentation overhead; use unprofiled step 30 onward for speed comparisons.
No numerical, memory or speed acceptance gates; inspect every loss and timing point in paired CSVs.
Failed arms retain their logs; no automatic recomputation fallback. A failed arm is not a valid speed comparison.
REPORT

echo "Two arms; ${NUM_ROLLOUT} steps each; GDN_RECOMPUTE=0; chunk=8192; router=fp64."
echo "Logs stream to this terminal and are saved in ${RESULT_DIR}/*.log"
failed=0
for arm in "${ARMS[@]}"; do
   mode=chunkflow
   export RELAX_CHUNKFLOW_EXECUTION_OPT=0
   if [ "${arm}" = baseline ]; then mode=baseline; fi
   if [ "${arm}" = chunkflow_exec ]; then export RELAX_CHUNKFLOW_EXECUTION_OPT=1; fi
   unset RELAX_TRAIN_PROFILE_DIR RELAX_TRAIN_MEMORY_DIR RELAX_DEVICE_MEMORY_DIR
   if [ "${CHUNKFLOW_PROFILE}" = 1 ]; then
      export RELAX_TRAIN_PROFILE_DIR="${RESULT_DIR}/${arm}/train_trace"
   else
      export RELAX_TRAIN_MEMORY_DIR="${RESULT_DIR}/${arm}/memory"
      export RELAX_DEVICE_MEMORY_DIR="${RESULT_DIR}/${arm}"
   fi
   echo "START arm=${arm} execution_opt=${RELAX_CHUNKFLOW_EXECUTION_OPT}"
   if RUN_NAME="${RUN_ID}-${arm}" LOG_FILE="${RESULT_DIR}/${arm}.log" \
      bash scripts/entrypoint/ray-job.sh \
      scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh "${mode}"; then
      echo "${arm}: complete" | tee -a "${RESULT_DIR}/comparison.md"
   else
      status=$?
      echo "${arm}: failed (exit ${status}); inspect ${RESULT_DIR}/${arm}.log" | tee -a "${RESULT_DIR}/comparison.md"
      failed=1
      if [ "${status}" -eq 130 ] || [ "${status}" -eq 143 ]; then exit "${status}"; fi
   fi
done
if [ "${CHUNKFLOW_PROFILE}" = 0 ] && [ "${failed}" = 0 ]; then
   REPORT="scripts/training/sft/tools/summarize_qwen36_comparison.py"
   python3 "${REPORT}" "${RESULT_DIR}" --expected-steps "${NUM_ROLLOUT}" \
      --chunkflow-arm chunkflow_exec
fi
echo "Results: ${RESULT_DIR}; failed=${failed}"
exit "${failed}"
