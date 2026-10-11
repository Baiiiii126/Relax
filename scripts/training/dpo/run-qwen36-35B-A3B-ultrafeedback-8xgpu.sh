#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.6-35B-A3B DPO on UltraFeedback, 8xGPU single-node.
#
# Relax side of the Relax <-> ms-swift DPO alignment experiment. Mirrors
# swift/run_dpo.sh exactly:
#   model      Qwen3.6-35B-A3B (full fine-tuning)
#   data       UltraFeedback preference pairs, 4096 train / 512 eval
#   steps      100 (1600 pairs = 0.39 epoch) -- the comparison run
#   batch      global 16
#   lr         1e-6, cosine over 100 steps, 5% warmup
#   max_length 2048
#   beta       0.1
#   shuffle    OFF on both sides (--no-dataset-shuffle)
#   clearml    project DPO-Align/ultrafeedback
#
# DPO v1 requires TP=CP=PP=1, so the 35B MoE is sharded with expert
# parallelism only (EP=8, DP=1) and the optimizer is CPU-offloaded.
#
# Usage:
#   DATA_DIR=<dir containing ultrafeedback/> HF_CHECKPOINT=<Qwen3.6-35B-A3B> \
#     bash scripts/training/dpo/run-qwen36-35B-A3B-ultrafeedback-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=DPO-Align/ultrafeedback}"
EXP_NAME="${EXP_NAME:=relax-qwen3.6-35b-ultrafeedback}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
DATA_DIR="${DATA_DIR:=${EXP_DIR}}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${EXP_DIR}/Qwen3.6-35B-A3B}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/ultrafeedback/ultrafeedback_train.parquet}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-${DATA_DIR}/ultrafeedback/ultrafeedback_eval.parquet}"

# 4096 train pairs / 16 global batch = 256 optimizer steps == 1 epoch.
# The comparison run only needs the first 100 steps (1600 pairs, 0.39 epoch);
# both sides must therefore consume the SAME leading 1600 rows, which is why
# shuffling is disabled on both (see --no-dataset-shuffle below).
NUM_ROLLOUT="${NUM_ROLLOUT:=100}"
SEQ_LENGTH="${SEQ_LENGTH:=2048}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --enable-weights-backuper
   # No --load / --save: this alignment run does not persist checkpoints
   # (every save path in Relax is guarded by `args.save is not None`).
)

DPO_ARGS=(
   --loss-type dpo
   --dpo-beta "${DPO_BETA:-0.1}"

   # Alignment: consume rows in file order so both frameworks see the same
   # samples at the same step. ms-swift cannot reproduce Relax's permutation,
   # so the only common ground is "no shuffle on either side". Mirrors
   # `--dataset_shuffle false --train_dataloader_shuffle false` on the swift
   # side. --seed matches swift's data_seed for good measure.
   --no-dataset-shuffle
   --seed "${SEED:-42}"

   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --preference-chosen-key chosen
   --preference-rejected-key rejected
   --preference-max-length "${SEQ_LENGTH}"
   --preference-max-completion-length "${SEQ_LENGTH}"

   --seq-length "${SEQ_LENGTH}"
   --num-rollout "${NUM_ROLLOUT}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --use-gloo-process-groups
)

EVAL_ARGS=(
   --eval-prompt-data ultrafeedback "${EVAL_PROMPT_DATA}"
   --eval-interval "${EVAL_INTERVAL:-25}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style cosine
   --lr-warmup-fraction "${WARMUP_FRACTION:-0.05}"
   --min-lr 0
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.999
   --clip-grad 1.0

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex

   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0

   --colocate
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${DPO_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee log/${EXP_NAME}-${now}.log
