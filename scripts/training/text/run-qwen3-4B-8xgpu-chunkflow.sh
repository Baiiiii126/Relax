#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU chunkflow correctness A/B.
#
# Chunked attention is numerically equivalent to unchunked attention, so the
# point of this run is to show the training loss is unchanged. Comparing two
# ordinary RL runs cannot show that: rollout samples at temperature 1, so the
# two arms would train on different data and the loss difference would be
# dominated by sampling noise, not by chunkflow.
#
# So the rollout is done ONCE and replayed. Three phases:
#
#   MODE=dump                    bash scripts/training/text/run-qwen3-4B-8xgpu-chunkflow.sh
#   MODE=train CHUNK_SIZE=0      bash scripts/training/text/run-qwen3-4B-8xgpu-chunkflow.sh
#   MODE=train CHUNK_SIZE=2048   bash scripts/training/text/run-qwen3-4B-8xgpu-chunkflow.sh
#
# `dump` generates and saves the rollout data (sglang only, no training). The
# two `train` arms replay that exact data (--load-debug-rollout-data implies
# --debug-train-only, so sglang never starts), differ in one flag pair, and
# start from the same HF checkpoint. Any per-step loss difference between them
# is attributable to chunked attention and nothing else.
#
# Run the three sequentially -- each uses all 8 GPUs.
#
# This checks correctness only. Chunked attention does not reduce memory or
# pipeline bubbles on its own; that needs the per-chunk backward schedule,
# which is a separate step.

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
NUM_ROLLOUT="${NUM_ROLLOUT:=3}"

# Where the dumped rollout lives. Both train arms must read the same one.
ROLLOUT_DUMP_DIR="${ROLLOUT_DUMP_DIR:-${EXP_DIR}/chunkflow-rollout-dump}"
MODE="${MODE:=train}"
# 0 = baseline arm (chunkflow off). >0 = tokens per chunk.
CHUNK_SIZE="${CHUNK_SIZE:=2048}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   # No --rollout-shuffle: the dump must be reproducible from the prompt file.

   --rm-type dapo
   --reward-key score

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
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
   # No --use-kl-loss: its coefficient is 0 here anyway, and dropping it keeps
   # the reference model out of a run that only needs the actor.
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

if [ "${MODE}" = "dump" ]; then
   mkdir -p "${ROLLOUT_DUMP_DIR}"
   ARM="dump"
   RESOURCE='{"rollout": [1, 8]}'
   MODE_ARGS=(
      --debug-rollout-only
      --save-debug-rollout-data "${ROLLOUT_DUMP_DIR}/{rollout_id}.pt"
   )
elif [ "${CHUNK_SIZE}" -gt 0 ]; then
   ARM="chunk${CHUNK_SIZE}"
   RESOURCE='{"actor": [1, 8]}'
   MODE_ARGS=(
      --load-debug-rollout-data "${ROLLOUT_DUMP_DIR}/{rollout_id}.pt"
      --use-chunkflow
      --chunkflow-chunk-size ${CHUNK_SIZE}
   )
else
   ARM="baseline"
   RESOURCE='{"actor": [1, 8]}'
   MODE_ARGS=(
      --load-debug-rollout-data "${ROLLOUT_DUMP_DIR}/{rollout_id}.pt"
   )
fi

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-chunkflow-${ARM}-gpu8-${now}
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "${RESOURCE}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
    --skip-eval-before-train \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-chunkflow-${ARM}-gpu8-${now}.log
