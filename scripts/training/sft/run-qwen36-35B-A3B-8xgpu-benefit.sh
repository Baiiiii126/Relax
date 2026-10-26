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
export GBS="${GBS:-8}"
export CHUNK_SIZE="${CHUNK_SIZE:-8192}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export TRAIN_SEED="${TRAIN_SEED:-1235}"
export CHUNKFLOW_PROFILE="${CHUNKFLOW_PROFILE:-0}"
export RELAX_TRAIN_PROFILE_RECORD_SHAPES="${RELAX_TRAIN_PROFILE_RECORD_SHAPES:-0}"
case "${CHUNKFLOW_PROFILE}" in
   0) RUN_KIND=benefit ;;
   1) RUN_KIND=profile ;;
   *) echo "ERROR: CHUNKFLOW_PROFILE must be 0 or 1" >&2; exit 2 ;;
esac
export RELAX_CHUNKFLOW_PARENT_WINDOW="${RELAX_CHUNKFLOW_PARENT_WINDOW:-0}"
ARMS=(baseline chunkflow)
case "${QWEN36_TAIL_POOL_AB:-0}" in
   0) ;;
   1) ARMS+=(chunkflow_pool) ;;
   *) echo "ERROR: QWEN36_TAIL_POOL_AB must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${QWEN36_REVERSE_ORDER:-0}" = 1 ]; then
   if [ "${QWEN36_TAIL_POOL_AB:-0}" = 1 ]; then
      ARMS=(chunkflow_pool chunkflow baseline)
   else
      ARMS=(chunkflow baseline)
   fi
fi
CHUNKS=()
if [ -n "${CHUNK_SIZES:-}" ]; then
   if [ "${CHUNKFLOW_PROFILE}" != 0 ] || [ "${QWEN36_TAIL_POOL_AB:-0}" != 0 ]; then
      echo "ERROR: CHUNK_SIZES requires CHUNKFLOW_PROFILE=0 and QWEN36_TAIL_POOL_AB=0" >&2
      exit 2
   fi
   if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || [ "${NUM_ROLLOUT}" -le 30 ]; then
      echo "ERROR: chunk sweep requires NUM_ROLLOUT > 30" >&2
      exit 2
   fi
   read -r -a CHUNKS <<< "${CHUNK_SIZES}"
   ARMS=(baseline)
   for chunk in "${CHUNKS[@]}"; do
      if ! [[ "${chunk}" =~ ^[1-9][0-9]*$ ]] || [ "${chunk}" -ge "${SEQ_LENGTH:-32768}" ]; then
         echo "ERROR: each CHUNK_SIZES value must be positive and smaller than SEQ_LENGTH" >&2
         exit 2
      fi
      for arm in "${ARMS[@]}"; do
         if [ "${arm}" = "chunk${chunk}" ]; then
            echo "ERROR: duplicate chunk size ${chunk}" >&2
            exit 2
         fi
      done
      ARMS+=("chunk${chunk}")
   done
   if [ "${#CHUNKS[@]}" -eq 0 ]; then
      echo "ERROR: CHUNK_SIZES contains no sizes" >&2
      exit 2
   fi
   export CHUNK_SIZE="${CHUNKS[0]}"
   if [ "${QWEN36_REVERSE_ORDER:-0}" = 1 ]; then
      REVERSED=()
      for ((i=${#ARMS[@]}-1; i>=0; i--)); do REVERSED+=("${ARMS[i]}"); done
      ARMS=("${REVERSED[@]}")
   fi
fi
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
Chunk sizes under comparison: ${CHUNK_SIZES:-${CHUNK_SIZE}}. GBS is fixed across all arms; each chunk uses the same baseline.
LR=${LEARNING_RATE:-1e-5}; GDN recompute=${GDN_RECOMPUTE:-1}; CPU optimizer overlap=${CPU_OPTIMIZER_OVERLAP:-1}.
MoE shared expert overlap: ${MOE_SHARED_EXPERT_OVERLAP} (both arms).
Router dtype: ${QWEN36_ROUTER_DTYPE:-fp32}.
ChunkFlow parent window: ${RELAX_CHUNKFLOW_PARENT_WINDOW} (0=all parents in each step; positive values cap parents per drain).
Data: ${PROMPT_DATA:-unset}; input key: ${INPUT_KEY:-question}; label key: ${LABEL_KEY-answer}.
Numerical probes and FP32 KV experiment disabled.
Arms in execution order: ${ARMS[*]}. Each starts from the same checkpoint, seed and optimizer settings.
chunkflow_pool enables bounded terminal-tail pooling; chunkflow disables it. Both use the same parent window and CPU preparation code.
Profiler: ${CHUNKFLOW_PROFILE}; schedule prediction is disabled. For unprofiled performance runs only, compare paired steps after the first 30 warmup steps using total-time ratios.
Both arms must be rerun for this topology; previous EP1 measurements are not this baseline.
EOF
if [ "${CHUNKFLOW_PROFILE}" = 1 ]; then
   cat > "${RESULT_DIR}/profile.md" <<EOF
# Qwen3.6 baseline / ChunkFlow profiler comparison

Configuration and completion: comparison.md. Profile steps: ${PROFILE_START:-34}..${PROFILE_END:-36} (0-based).
Profiler timing is diagnostic only; do not use it to claim speedup.
Record tensor shapes: ${RELAX_TRAIN_PROFILE_RECORD_SHAPES} (0=lean trace; 1=shape diagnostics).
Each arm writes GPU traces (*.pt.trace.json.gz) and per-rank segments_*.jsonl below.
Host intervals include nested calls and waits; do not sum them as GPU compute time.
thread_cpu_s counts CPU time on the calling thread; wall time minus thread CPU time includes waits and descheduling, not just GPU idle.
cpu_threads records intraop/interop thread counts and affinity size; it does not establish CPU quota or available capacity.
chunkflow.unit_prepare separates input slicing and metadata preparation from MoE kernel launches.
chunkflow.tail_pool.plan measures pooling decisions; chunkflow.batch.pooled includes its nested chunkflow.batch.pool_part calls.
chunkflow.gdn.kernel.conv/recurrent count actual per-segment kernel calls; chunkflow.gdn.tail_assemble marks pooled output concatenation/padding.
GDN kernel scopes also include calls during selective recomputation; they are host launch intervals, not GPU durations.
These kernel scopes are ChunkFlow-only; compare model.gdn_compute and GPU kernels across all three arms.
segments files distinguish tail_pool_enabled and record process CPU time plus visible cgroup counters before/after each selected step.
The visible cgroup may be shared with other processes; throttling/pressure deltas indicate resource pressure, not its owner or a proven cause.
Process CPU time covers all threads of one worker; do not sum shared cgroup deltas across ranks or treat PSI averages as step durations.
ChunkFlow window scopes identify parent ranges and prepare/warmup/steady/drain phases, with unit counts.
Warmup/drain include useful work; measure GPU idle and communication waits inside them to assess bubbles.
Both arms split MoE routing, dispatch, expert computation and combine; dispatcher DtoH synchronization is marked when available.
Missing MoE scopes mean the method was not installed; zero calls mean it was installed but unused.
MoE scopes describe forward calls; use autograd events and GPU kernels inside backward for backward costs.
Use the GPU trace to distinguish GDN recomputation inside backward from the initial forward.
CUDA allocator peaks are per profiled step and include profiler effects; they exclude allocations outside PyTorch.
EOF
fi
for arm in "${ARMS[@]}"; do
   mode="${arm}"
   export RELAX_CHUNKFLOW_TAIL_POOL=0
   if [ "${#CHUNKS[@]}" -gt 0 ] && [ "${arm}" != baseline ]; then
      mode=chunkflow
      export CHUNK_SIZE="${arm#chunk}"
   fi
   if [ "${arm}" = chunkflow_pool ]; then
      mode=chunkflow
      export RELAX_CHUNKFLOW_TAIL_POOL=1
   fi
   if [ "${CHUNKFLOW_PROFILE}" = 1 ]; then
      export RELAX_TRAIN_PROFILE_DIR="${RESULT_DIR}/${arm}/train_trace"
      echo "- ${arm}: ${RELAX_TRAIN_PROFILE_DIR}" >> "${RESULT_DIR}/profile.md"
   fi
   RUN_NAME="${RUN_ID}-${arm}" LOG_FILE="${RESULT_DIR}/${arm}.log" \
      bash scripts/entrypoint/ray-job.sh \
      scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh "${mode}"
   echo "${arm}: complete" >> "${RESULT_DIR}/comparison.md"
   if [ "${#CHUNKS[@]}" -gt 0 ]; then
      python3 scripts/training/sft/tools/summarize_qwen36_comparison.py "${RESULT_DIR}" \
         --expected-steps "${NUM_ROLLOUT}" --chunks "${CHUNKS[@]}"
   fi
done
if [ "${CHUNKFLOW_PROFILE}" = 0 ]; then
   if [ "${#CHUNKS[@]}" -eq 0 ]; then
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/training/sft/tools/summarize_qwen36_comparison.py "${RESULT_DIR}" --expected-steps "${NUM_ROLLOUT}"
   else
      python3 scripts/training/sft/tools/summarize_qwen36_comparison.py "${RESULT_DIR}" \
         --expected-steps "${NUM_ROLLOUT}" --chunks "${CHUNKS[@]}" --require-complete
   fi
   if [ "${QWEN36_TAIL_POOL_AB:-0}" = 1 ]; then
      python3 scripts/training/sft/tools/summarize_qwen36_comparison.py "${RESULT_DIR}" \
         --expected-steps "${NUM_ROLLOUT}" --chunkflow-arm chunkflow_pool --output-prefix tail_pool-
      python3 scripts/training/sft/tools/summarize_qwen36_comparison.py "${RESULT_DIR}" \
         --expected-steps "${NUM_ROLLOUT}" --baseline-arm chunkflow --chunkflow-arm chunkflow_pool \
         --output-prefix tail_pool-vs-unpooled-
   fi
fi
echo "Comparison complete: ${RESULT_DIR}"
