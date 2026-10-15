#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B SFT ChunkFlow benefit A/B, 8 GPUs, TP2 x PP4 x DP1.
#
#   bash scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh baseline
#   bash scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh chunkflow
set -exo pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1}"
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
# Micro-batch count M is a property of the data: the samples in one step are
# first-fit packed into bins of MAX_TOKENS_PER_GPU (see
# relax/utils/data/data.py:get_minimum_num_micro_batch_size). Raise GBS for more samples
# per step; lower MAX_TOKENS_PER_GPU for more, smaller micro-batches. Defaults
# reproduce the original single-sample behaviour (M = 1).
GBS="${GBS:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${SEQ_LENGTH}}"

TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-4}"
if [ $((36 % PP_SIZE)) -ne 0 ]; then
   echo "ERROR: PP_SIZE=${PP_SIZE} must divide Qwen3-4B's 36 layers" >&2
   exit 2
fi
NUM_GPUS=$((TP_SIZE * PP_SIZE))
MODE="${1:-chunkflow}"
case "${MODE}" in
   baseline)
      CHUNK_SIZE=0
      ;;
   chunkflow)
      # Keep this below the benchmark dataset's 16K-32K sequences so this arm
      # exercises real multi-chunk execution.
      CHUNK_SIZE="${CHUNK_SIZE:-128}"
      ;;
   *)
      echo "Usage: bash $0 {baseline|chunkflow}" >&2
      exit 2
      ;;
esac
PACK_POLICY="${PACK_POLICY:=staggered}"
KEEP_LAST_K="${KEEP_LAST_K:=0}"
if [ "${CHUNK_SIZE}" -gt 0 ]; then
   CHUNKFLOW_ARGS=(--use-chunkflow --chunkflow-chunk-size ${CHUNK_SIZE} --chunkflow-pack-policy "${PACK_POLICY}")
   ARM="chunk${CHUNK_SIZE}-${PACK_POLICY}"
   CHUNKFLOW_ARGS+=(--chunkflow-pipeline ${PP_SIZE})
   ARM="${ARM}-pp${PP_SIZE}"
   if [ "${KEEP_LAST_K}" -gt 0 ]; then
      CHUNKFLOW_ARGS+=(--chunkflow-keep-last-k ${KEEP_LAST_K})
      ARM="${ARM}-k${KEEP_LAST_K}"
   fi
else
   CHUNKFLOW_ARGS=()
   ARM="baseline"
fi

PROJECT_NAME="${PROJECT_NAME:-Relax/dev/chunkflow-benefit}"
EXP_NAME="${EXP_NAME:-qwen3-4b-sft-${ARM}}"

if [ ! -e "${MODEL_DIR}/Qwen3-4B" ]; then
   echo "ERROR: ${MODEL_DIR}/Qwen3-4B does not exist. Set MODEL_DIR / DATA_DIR, e.g." >&2
   echo "  export MODEL_DIR=/path/to/models" >&2
   echo "  export DATA_DIR=/path/to/data" >&2
   exit 1
fi
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/chunkflow-longtail/train.parquet}"
if [ ! -e "${PROMPT_DATA}" ]; then
   echo "ERROR: ${PROMPT_DATA} does not exist." >&2
   echo "Build it with scripts/training/sft/tools/build_chunkflow_longtail_sft.py, or set PROMPT_DATA." >&2
   exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   --num-rollout ${NUM_ROLLOUT}
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key question
   --label-key answer
   --global-batch-size ${GBS}
   --use-dynamic-batch-size
   --seq-length ${SEQ_LENGTH}
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --sft-oversize-strategy skip
)

export RELAX_CHUNKFLOW_PIPELINE_UNTESTED="${RELAX_CHUNKFLOW_PIPELINE_UNTESTED:=1}"
export RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS="${RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS:-0}"

export RUNTIME_ENV_JSON=$(python3 -c '
import json, os
env = json.loads(os.environ["RUNTIME_ENV_JSON"])
target = env.setdefault("env_vars", {})
target["RELAX_CHUNKFLOW_PIPELINE_UNTESTED"] = os.environ["RELAX_CHUNKFLOW_PIPELINE_UNTESTED"]
target["RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS"] = os.environ["RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS"]
print(json.dumps(env))
')

TP_ARGS=(--tensor-model-parallel-size ${TP_SIZE})
if [ "${TP_SIZE}" -gt 1 ]; then
   TP_ARGS+=(--sequence-parallel)
fi

PERF_ARGS=(
   "${TP_ARGS[@]}"
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --no-rope-fusion
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

MISC_ARGS=(
   --seed 1234
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CLEARML_ARGS=(
   --use-clearml
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

LOG_FILE="${LOG_FILE:-log/${EXP_NAME}-gpu${NUM_GPUS}-${now}.log}"
mkdir -p "$(dirname -- "${LOG_FILE}")"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
echo "LOG_FILE=${LOG_FILE}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\": [1, 0], \"actor\": [1, ${NUM_GPUS}]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${CHUNKFLOW_ARGS[@]}" \
    "${CLEARML_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_FILE}"

if [ -z "${RAY_NO_WAIT:-}" ] && [ "${CHUNK_SIZE}" -gt 0 ]; then
   if ! grep -q "chunked attention installed" "${LOG_FILE}"; then
      echo "FAIL: chunkflow never installed -- this arm proves nothing." >&2
      exit 1
   fi
fi
