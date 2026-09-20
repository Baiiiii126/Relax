#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU RL training with a chunkflow toggle.
#
# Same as run-qwen3-4B-8xgpu.sh, plus one switch:
#
#   CHUNK_SIZE=0    -> plain run (baseline)
#   CHUNK_SIZE=2048 -> chunked attention on
#
# Run both, then overlay the two `train/loss` curves in ClearML
# (project "Relax/dev/chunkflow"). Chunked attention is numerically equivalent,
# so the curves should track each other within RL sampling noise -- they will
# NOT be identical step by step (rollout samples at temperature 1, so the arms
# see different data), and near-identical early steps drifting apart later is
# expected from Adam amplification, not a chunkflow bug.
#
#   CHUNK_SIZE=0    bash scripts/training/text/run-qwen3-4B-8xgpu-chunkflow.sh
#   CHUNK_SIZE=2048 bash scripts/training/text/run-qwen3-4B-8xgpu-chunkflow.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/chunkflow}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=100}"

# 0 = baseline arm (chunkflow off). >0 = tokens per chunk.
CHUNK_SIZE="${CHUNK_SIZE:=2048}"
if [ "${CHUNK_SIZE}" -gt 0 ]; then
   CHUNKFLOW_ARGS=(--use-chunkflow --chunkflow-chunk-size ${CHUNK_SIZE})
   ARM="chunk${CHUNK_SIZE}"
else
   CHUNKFLOW_ARGS=()
   ARM="baseline"
fi

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   # No --load/--save: both arms start from the same HF checkpoint, and neither
   # inherits the other's state.
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=(
   --skip-eval-before-train
   --log-passrate
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   # chunkflow's StateStore is local to one sequence, so CP must stay 1.
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 30720
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.8
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-chunkflow-${ARM}-gpu8-${now}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

mkdir -p log
LOG_FILE="log/qwen3-4b-chunkflow-${ARM}-gpu8-${now}.log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}'\
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${CHUNKFLOW_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_FILE}"

# A chunkflow arm whose patch never fired would look identical to the baseline
# and read as a pass, so check the arm was actually chunked.
if [ "${CHUNK_SIZE}" -gt 0 ] && ! grep -q "chunked attention installed" "${LOG_FILE}"; then
   echo "FAIL: chunkflow never installed -- this arm proves nothing." >&2
   exit 1
fi
