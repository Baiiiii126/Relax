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
if [ "${NUM_GPUS:-8}" != 8 ] || [ "${TP_SIZE:-2}" != 2 ] || [ "${PP_SIZE:-4}" != 4 ] ||
   [ "${PACK_POLICY:-staggered}" != staggered ]; then
   echo "ERROR: this example requires 8 GPUs, TP2 / PP4 / CP1 / EP1 and staggered packing" >&2
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
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-${ARM}-gbs${GBS}-cap${MAX_TOKENS_PER_GPU}-s${NUM_ROLLOUT}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-${now}}"
LOG_FILE="${LOG_FILE:-log/${RUN_NAME}.log}"

export RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1
export RELAX_CHUNKFLOW_FLASH_BACKEND="${RELAX_CHUNKFLOW_FLASH_BACKEND:-fa3}"
export RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS="${RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS:-0}"
export RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY=1
case "${RELAX_CHUNKFLOW_FLASH_BACKEND}" in
   fa2|fa3) ;;
   *) echo "ERROR: RELAX_CHUNKFLOW_FLASH_BACKEND must be fa2 or fa3" >&2; exit 2 ;;
esac
case "${RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS}" in
   0|1) ;;
   *) echo "ERROR: RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS must be 0 or 1" >&2; exit 2 ;;
esac
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
env = json.loads(os.environ["RUNTIME_ENV_JSON"])
for key in ("RELAX_CHUNKFLOW_PIPELINE_UNTESTED", "RELAX_CHUNKFLOW_FLASH_BACKEND",
            "RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS", "RELAX_CHUNKFLOW_TELEMETRY_LOCAL_ONLY"):
    env.setdefault("env_vars", {})[key] = os.environ[key]
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
   --label-key "${LABEL_KEY:-answer}"
   --num-rollout "${NUM_ROLLOUT}"
   --global-batch-size "${GBS}"
   --micro-batch-size 1
   --use-dynamic-batch-size
   --seq-length "${SEQ_LENGTH}"
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --sft-oversize-strategy skip
   --calculate-per-token-loss
   --cross-entropy-loss-fusion
   --sft-chunked-logits
   --sft-logits-chunk-size "${LOGITS_CHUNK_SIZE}"
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity selective
   --recompute-modules gdn
   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0
   --optimizer adam
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --lr 1e-5
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
echo "MODE=${MODE} TP=2 PP=4 CP=1 EP=1 GBS=${GBS} STEPS=${NUM_ROLLOUT} SEED=${TRAIN_SEED} ARM=${ARM}"
echo "LOG_FILE=${LOG_FILE}"
ray job submit "${SUBMIT_ARGS[@]}" --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "${CHUNKFLOW_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

if [ "${RAY_NO_WAIT:-0}" != 1 ]; then
   if ! grep -q "Actor training completed step $((NUM_ROLLOUT - 1))/${NUM_ROLLOUT}" "${LOG_FILE}"; then
      echo "ERROR: expected training steps not completed; inspect ${LOG_FILE}" >&2
      exit 1
   fi
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
