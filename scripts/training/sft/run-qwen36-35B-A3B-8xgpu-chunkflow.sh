#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.6 GDN validation: run baseline and chunkflow separately via ray-job.sh.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODE="${1:-chunkflow}"
if [ "$#" -gt 1 ] || { [ "${MODE}" != baseline ] && [ "${MODE}" != chunkflow ]; }; then
   echo "Usage: bash scripts/entrypoint/ray-job.sh $0 {baseline|chunkflow}" >&2
   exit 2
fi
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:+${MODEL_DIR}/Qwen3.6-35B-A3B}}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT or MODEL_DIR containing Qwen3.6-35B-A3B}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR:+${DATA_DIR}/chunkflow-mixed/train.parquet}}"
PROMPT_DATA="${PROMPT_DATA:?Set PROMPT_DATA or DATA_DIR containing chunkflow-mixed/train.parquet}"
GBS="${GBS:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${SEQ_LENGTH}}"
LOGITS_CHUNK_SIZE="${LOGITS_CHUNK_SIZE:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
KEEP_LAST_K="${KEEP_LAST_K:-0}"
TRAIN_SEED="${TRAIN_SEED:-1235}"
EP_SIZE="${EP_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GDN_RECOMPUTE="${GDN_RECOMPUTE:-1}"
CPU_OPTIMIZER_OVERLAP="${CPU_OPTIMIZER_OVERLAP:-1}"
MOE_SHARED_EXPERT_OVERLAP="${MOE_SHARED_EXPERT_OVERLAP:-0}"
export RELAX_MOE_METADATA_MODE="${RELAX_MOE_METADATA_MODE:-off}"
export RELAX_MOE_CHUNK_MODE="${RELAX_MOE_CHUNK_MODE:-off}"
case "${RELAX_MOE_CHUNK_MODE}" in
   off|dedup|expert|both) ;;
   *) echo "ERROR: RELAX_MOE_CHUNK_MODE must be off, dedup, expert or both" >&2; exit 2 ;;
esac
if [ "${RELAX_MOE_CHUNK_MODE}" != off ] && [ "${RELAX_MOE_METADATA_MODE}" != off ]; then
   echo "ERROR: MoE chunk experiments require RELAX_MOE_METADATA_MODE=off" >&2
   exit 2
fi
case "${RELAX_MOE_METADATA_MODE}" in
   off|a|b) ;;
   *) echo "ERROR: RELAX_MOE_METADATA_MODE must be off, a or b" >&2; exit 2 ;;
esac
DIAGNOSTIC_ARGS=()
case "${GDN_RECOMPUTE}" in
   1) DIAGNOSTIC_ARGS+=(--recompute-granularity selective --recompute-modules gdn) ;;
   0) ;;
   *) echo "ERROR: GDN_RECOMPUTE must be 0 or 1" >&2; exit 2 ;;
esac
case "${CPU_OPTIMIZER_OVERLAP}" in
   1) DIAGNOSTIC_ARGS+=(--overlap-cpu-optimizer-d2h-h2d) ;;
   0) ;;
   *) echo "ERROR: CPU_OPTIMIZER_OVERLAP must be 0 or 1" >&2; exit 2 ;;
esac
case "${MOE_SHARED_EXPERT_OVERLAP}" in
   1) DIAGNOSTIC_ARGS+=(--moe-shared-expert-overlap) ;;
   0) ;;
   *) echo "ERROR: MOE_SHARED_EXPERT_OVERLAP must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${EP_SIZE}" != 1 ] && [ "${EP_SIZE}" != 2 ]; then
   echo "ERROR: EP_SIZE must be 1 or 2 for TP2 / PP4 / ETP1 on 8 GPUs" >&2
   exit 2
fi
for key in GBS NUM_ROLLOUT SEQ_LENGTH MAX_TOKENS_PER_GPU LOGITS_CHUNK_SIZE CHUNK_SIZE; do
   if ! [[ "${!key}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: ${key} must be a positive integer" >&2
      exit 2
   fi
done
for key in KEEP_LAST_K TRAIN_SEED; do
   if ! [[ "${!key}" =~ ^(0|[1-9][0-9]*)$ ]]; then
      echo "ERROR: ${key} must be a nonnegative integer" >&2
      exit 2
   fi
done
PROFILE_ARGS=()
export RELAX_TRAIN_PROFILE_RECORD_SHAPES="${RELAX_TRAIN_PROFILE_RECORD_SHAPES:-0}"
case "${RELAX_TRAIN_PROFILE_RECORD_SHAPES}" in
   0|1) ;;
   *) echo "ERROR: RELAX_TRAIN_PROFILE_RECORD_SHAPES must be 0 or 1" >&2; exit 2 ;;
esac
case "${CHUNKFLOW_PROFILE:-0}" in
   0) ;;
   1)
      PROFILE_START="${PROFILE_START:-34}"
      PROFILE_END="${PROFILE_END:-36}"
      if ! [[ "${PROFILE_START}" =~ ^(0|[1-9][0-9]*)$ && "${PROFILE_END}" =~ ^(0|[1-9][0-9]*)$ ]] ||
         [ "${PROFILE_END}" -lt "${PROFILE_START}" ] || [ "${NUM_ROLLOUT}" -le "$((PROFILE_END + 1))" ]; then
         echo "ERROR: require 0 <= PROFILE_START <= PROFILE_END and NUM_ROLLOUT > PROFILE_END + 1" >&2
         exit 2
      fi
      PROFILE_ARGS=(--use-pytorch-profiler --profile-target train_overall
         --profile-step-start "${PROFILE_START}" --profile-step-end "${PROFILE_END}")
      ;;
   *) echo "ERROR: CHUNKFLOW_PROFILE must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${NUM_GPUS:-8}" != 8 ] || [ "${TP_SIZE:-2}" != 2 ] || [ "${PP_SIZE:-4}" != 4 ] ||
   [ "${PACK_POLICY:-staggered}" != staggered ]; then
   echo "ERROR: this example requires 8 GPUs, TP2 / PP4 / CP1 and staggered packing" >&2
   exit 2
fi
if [ "${MAX_TOKENS_PER_GPU}" -lt "${SEQ_LENGTH}" ]; then
   echo "ERROR: MAX_TOKENS_PER_GPU must cover SEQ_LENGTH to keep the intended long samples" >&2
   exit 2
fi
if [ "${MODE}" = chunkflow ] && [ "${CHUNK_SIZE}" -ge "${SEQ_LENGTH}" ]; then
   echo "ERROR: CHUNK_SIZE must be smaller than SEQ_LENGTH to exercise state transfer" >&2
   exit 2
fi
if [ ! -f "${HF_CHECKPOINT}/config.json" ] || [ ! -f "${PROMPT_DATA}" ]; then
   echo "ERROR: missing checkpoint config or SFT data: ${HF_CHECKPOINT}/config.json ; ${PROMPT_DATA}" >&2
   exit 1
fi
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   echo "ERROR: launch this script through scripts/entrypoint/ray-job.sh" >&2
   exit 2
fi
source "${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../../models}/qwen36-35B-A3B.sh"

CHUNKFLOW_ARGS=()
ARM=baseline
if [ "${MODE}" = chunkflow ]; then
   ARM="chunk${CHUNK_SIZE}-k${KEEP_LAST_K}"
   CHUNKFLOW_ARGS=(--use-chunkflow --chunkflow-chunk-size "${CHUNK_SIZE}"
      --chunkflow-pack-policy staggered --chunkflow-keep-last-k "${KEEP_LAST_K}" --chunkflow-pipeline 4)
fi
now=$(date "+%Y-%m-%d-%H:%M:%S")
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/chunkflow-qwen36}"
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-${ARM}-ep${EP_SIZE}-gbs${GBS}-cap${MAX_TOKENS_PER_GPU}-s${NUM_ROLLOUT}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-${now}}"
LOG_FILE="${LOG_FILE:-log/${RUN_NAME}.log}"

export RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1
export RELAX_CHUNKFLOW_FLASH_BACKEND="${RELAX_CHUNKFLOW_FLASH_BACKEND:-fa3}"
export RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS="${RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS:-0}"
export RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY=1
export RELAX_CHUNKFLOW_PARENT_WINDOW="${RELAX_CHUNKFLOW_PARENT_WINDOW:-0}"
export RELAX_CHUNKFLOW_TAIL_POOL="${RELAX_CHUNKFLOW_TAIL_POOL:-0}"
case "${RELAX_CHUNKFLOW_TAIL_POOL}" in
   0|1) ;;
   *) echo "ERROR: RELAX_CHUNKFLOW_TAIL_POOL must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${RELAX_CHUNKFLOW_TAIL_POOL}" = 1 ] && [ "${KEEP_LAST_K}" != 0 ]; then
   echo "ERROR: tail pooling requires KEEP_LAST_K=0" >&2
   exit 2
fi
if ! [[ "${RELAX_CHUNKFLOW_PARENT_WINDOW}" =~ ^(0|[1-9][0-9]*)$ ]]; then
   echo "ERROR: RELAX_CHUNKFLOW_PARENT_WINDOW must be nonnegative (0 uses all step parents)" >&2
   exit 2
fi
case "${RELAX_CHUNKFLOW_FLASH_BACKEND}" in
   fa2|fa3) ;;
   *) echo "ERROR: RELAX_CHUNKFLOW_FLASH_BACKEND must be fa2 or fa3" >&2; exit 2 ;;
esac
case "${RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS}" in
   0|1) ;;
   *) echo "ERROR: RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${MODE}" = baseline ]; then
   unset RELAX_CHUNKFLOW_GATE_GAIN_PCT
elif [ -n "${RELAX_CHUNKFLOW_GATE_GAIN_PCT:-}" ]; then
   export RELAX_CHUNKFLOW_GATE_GAIN_PCT
   python3 -c 'import math, os; x=float(os.environ["RELAX_CHUNKFLOW_GATE_GAIN_PCT"]); assert math.isfinite(x) and x >= 0, "Invalid gate gain percentage"'
fi
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
env = json.loads(os.environ["RUNTIME_ENV_JSON"])
for key in ("RELAX_CHUNKFLOW_PIPELINE_UNTESTED", "RELAX_CHUNKFLOW_FLASH_BACKEND",
            "RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS", "RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY",
            "RELAX_CHUNKFLOW_PARENT_WINDOW", "RELAX_CHUNKFLOW_TAIL_POOL", "RELAX_TRAIN_PROFILE_RECORD_SHAPES",
            "RELAX_MOE_METADATA_MODE", "RELAX_MOE_CHUNK_MODE"):
    env.setdefault("env_vars", {})[key] = os.environ[key]
if os.environ.get("RELAX_TRAIN_PROFILE_DIR"):
    env.setdefault("env_vars", {})["RELAX_TRAIN_PROFILE_DIR"] = os.environ["RELAX_TRAIN_PROFILE_DIR"]
else:
    env.setdefault("env_vars", {}).pop("RELAX_TRAIN_PROFILE_DIR", None)
for key in ("RELAX_TRAIN_MEMORY_DIR", "RELAX_CHUNKFLOW_GATE_GAIN_PCT", "RELAX_CHUNKFLOW_DIAGNOSTIC_DIR", "RELAX_CHUNKFLOW_GRAD_ROLLOUTS", "RELAX_CHUNKFLOW_LAYER_TRACE",
            "RELAX_CHUNKFLOW_UPDATE_TRACE", "RELAX_CHUNKFLOW_KV_GRAD_FP32", "RELAX_CHUNKFLOW_TAIL_TRACE", "RELAX_CHUNKFLOW_CONV_TRACE", "RELAX_CHUNKFLOW_CONV_CHUNK_SIZE"):
    if os.environ.get(key):
        env.setdefault("env_vars", {})[key] = os.environ[key]
    else:
        env.setdefault("env_vars", {}).pop(key, None)
print(json.dumps(env))
')
export RUNTIME_ENV_JSON

TRAIN_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --bf16
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key "${INPUT_KEY:-question}"
   --num-rollout "${NUM_ROLLOUT}"
   --global-batch-size "${GBS}"
   --micro-batch-size 1
   --use-dynamic-batch-size
   --seq-length "${SEQ_LENGTH}"
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --sft-oversize-strategy skip
   --sft-invalid-multimodal-strategy skip
   --calculate-per-token-loss
   --cross-entropy-loss-fusion
   --sft-chunked-logits
   --sft-logits-chunk-size "${LOGITS_CHUNK_SIZE}"
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 1
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size 1
   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0
   --optimizer adam
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --lr "${LEARNING_RATE}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
   --seed "${TRAIN_SEED}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-clearml
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${RUN_NAME}"
)
if [ -n "${LABEL_KEY-answer}" ]; then
   TRAIN_ARGS+=(--label-key "${LABEL_KEY-answer}")
fi
SUBMIT_ARGS=(--address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}")
case "${RAY_NO_WAIT:-0}" in
   0|"") ;;
   1) SUBMIT_ARGS+=(--no-wait) ;;
   *) echo "ERROR: RAY_NO_WAIT must be 0 or 1" >&2; exit 2 ;;
esac
if [ -n "${WORKING_DIR:-}" ]; then
   SUBMIT_ARGS+=(--working-dir "${WORKING_DIR}")
fi
mkdir -p "$(dirname -- "${LOG_FILE}")"
echo "MODE=${MODE} TP=2 PP=4 CP=1 EP=${EP_SIZE} GBS=${GBS} STEPS=${NUM_ROLLOUT} SEED=${TRAIN_SEED} ARM=${ARM}"
echo "LR=${LEARNING_RATE} GDN_RECOMPUTE=${GDN_RECOMPUTE} CPU_OPTIMIZER_OVERLAP=${CPU_OPTIMIZER_OVERLAP}"
echo "MOE_SHARED_EXPERT_OVERLAP=${MOE_SHARED_EXPERT_OVERLAP}"
echo "RELAX_MOE_METADATA_MODE=${RELAX_MOE_METADATA_MODE}"
echo "RELAX_MOE_CHUNK_MODE=${RELAX_MOE_CHUNK_MODE}"
echo "CHUNKFLOW_GATE_GAIN_PCT=${RELAX_CHUNKFLOW_GATE_GAIN_PCT:-disabled}"
echo "CHUNKFLOW_TAIL_POOL=${RELAX_CHUNKFLOW_TAIL_POOL} PARENT_WINDOW=${RELAX_CHUNKFLOW_PARENT_WINDOW}"
echo "LOG_FILE=${LOG_FILE}"
if [ "${CHUNKFLOW_PROFILE:-0}" = 1 ]; then
   echo "PROFILE_ROLLOUTS=${PROFILE_START}..${PROFILE_END} (0-based; includes profiler overhead)"
   echo "PROFILE_RECORD_SHAPES=${RELAX_TRAIN_PROFILE_RECORD_SHAPES}"
   echo "Worker output: ${RELAX_TRAIN_PROFILE_DIR:-traces/${RUN_NAME}/train_trace}/{*.pt.trace.json.gz,segments_*.jsonl}"
fi
memory_sampler_pid=""
stop_memory_sampler() {
   if [ -n "${memory_sampler_pid}" ]; then
      kill "${memory_sampler_pid}" 2>/dev/null || true
      wait "${memory_sampler_pid}" 2>/dev/null || true
      memory_sampler_pid=""
   fi
}
if [ -n "${RELAX_DEVICE_MEMORY_DIR:-}" ]; then
   if [ "${RAY_NO_WAIT:-0}" = 1 ]; then
      echo "ERROR: device memory sampling requires a blocking training run" >&2; exit 2
   fi
   # ray-job.sh has completed cleanup. Refuse to include surviving GPU contexts
   # from a previous arm in this arm's peak measurement.
   for ((attempt=0; attempt<30; attempt++)); do
      active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
      if [ -z "${active_gpu_pids}" ]; then break; fi
      sleep 1
   done
   if [ -n "${active_gpu_pids}" ]; then
      echo "ERROR: GPU contexts still active before measurement: ${active_gpu_pids}" >&2; exit 1
   fi
   mkdir -p "${RELAX_DEVICE_MEMORY_DIR}"
   nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 250 \
      > "${RELAX_DEVICE_MEMORY_DIR}/device-memory.csv" 2> "${RELAX_DEVICE_MEMORY_DIR}/device-memory.stderr" &
   memory_sampler_pid=$!
   trap stop_memory_sampler EXIT
   sleep 1
   kill -0 "${memory_sampler_pid}"
   python3 -c 'import time; print(time.time())' > "${RELAX_DEVICE_MEMORY_DIR}/device-memory.start"
fi
ray job submit "${SUBMIT_ARGS[@]}" --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DIAGNOSTIC_ARGS[@]}" "${PROFILE_ARGS[@]}" "${CHUNKFLOW_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

if [ -n "${memory_sampler_pid}" ]; then
   python3 -c 'import time; print(time.time())' > "${RELAX_DEVICE_MEMORY_DIR}/device-memory.end"
   sleep 1
   if ! kill -0 "${memory_sampler_pid}"; then
      echo "ERROR: device memory sampler stopped before training completed" >&2; exit 1
   fi
   stop_memory_sampler
fi

if [ "${RAY_NO_WAIT:-0}" != 1 ]; then
   if ! grep -q "Actor training completed step $((NUM_ROLLOUT - 1))/${NUM_ROLLOUT}" "${LOG_FILE}"; then
      echo "ERROR: expected training steps not completed; inspect ${LOG_FILE}" >&2
      exit 1
   fi
   for feature in dedup expert; do
      if [ "${RELAX_MOE_CHUNK_MODE}" = "${feature}" ] || [ "${RELAX_MOE_CHUNK_MODE}" = both ]; then
         if ! grep -q "MoE chunk optimization installed: ${feature}" "${LOG_FILE}"; then
            echo "ERROR: requested MoE optimization not installed: ${feature}" >&2
            exit 1
         fi
      fi
   done
   if [ "${MODE}" = chunkflow ]; then
      for pattern in 'stateful GDN installed on [1-9][0-9]* layer' 'chunked attention installed on [1-9][0-9]* layer'; do
         if ! grep -Eq "chunkflow: ${pattern}" "${LOG_FILE}"; then
            echo "ERROR: ChunkFlow installation not confirmed: ${pattern}" >&2
            exit 1
         fi
      done
   elif grep -Eq 'chunkflow: (stateful GDN|chunked attention) installed' "${LOG_FILE}"; then
      echo "ERROR: baseline unexpectedly installed ChunkFlow" >&2
      exit 1
   fi
fi
