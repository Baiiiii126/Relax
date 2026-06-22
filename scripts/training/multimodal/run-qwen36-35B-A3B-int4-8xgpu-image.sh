#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B 16xGPU (2-node) fully sync training script for DAPO math dataset.
#
# Usage:
#   bash scripts/training/text/run-qwen36-35B-A3B-16xgpu-sync.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=300}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.6-35B-A3B-int4
   --ref-load ${MODEL_DIR}/Qwen3.6-35B-A3B
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   #--load ${MODEL_DIR}/Qwen3.6-35B-A3B-int4-save
   --save ${MODEL_DIR}/Qwen3.6-35B-A3B-int4-save
   --save-interval 200
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --multimodal-keys '{"image":"image"}'
   --system-prompt "${SYSTEM_PROMPT}"
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --global-batch-size 512
   --rollout-max-response-len 8192
   --rollout-max-prompt-len 2048
   --rollout-temperature 1.0
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 500
   --eval-prompt-data aime ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --qkv-format bshd
   # --micro-batch-size 1 # avoid OOM
   --use-dynamic-batch-size
   --max-tokens-per-gpu 6144
   --log-probs-max-tokens-per-gpu 40960

   --no-rope-fusion
)

GRPO_ARGS=(
   --advantage-estimator grpo
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
   --clip-grad 1.0

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   # NOTE(wuhuan): to avoid algorithm performance degradation
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-load-format dummy
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen36-35B-A3B-8x-sync-${now}
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
if [ ${MODE} = "async" ]; then
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
       ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 -m relax.entrypoints.train \
       --resource '{"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}'\
       --max-staleness 2 \
       --num-data-storage-units 1 \
       --num-iters-per-train-update 16 \
       --fully-async \
       --use-health-check \
       "${MODEL_ARGS[@]}" \
       "${CKPT_ARGS[@]}" \
       "${ROLLOUT_ARGS[@]}" \
       "${OPTIMIZER_ARGS[@]}" \
       "${GRPO_ARGS[@]}" \
       "${WANDB_ARGS[@]}" \
       "${PERF_ARGS[@]}" \
       "${SGLANG_ARGS[@]}" \
       "${MISC_ARGS[@]}"  2>&1 | tee log/qwen36-35B-A3B-int4-gpu8-async-${now}.log
else
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
       "${SGLANG_ARGS[@]}" \
       "${MISC_ARGS[@]}"  2>&1 | tee log/qwen36-35B-A3B-int4-gpu8-colocate-${now}.log
fi
