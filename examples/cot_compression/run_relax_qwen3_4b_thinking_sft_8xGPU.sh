#!/bin/bash
#
# Qwen3-4B-Thinking-2507 SFT on k1.15_llmgrading.json, 8xGPU, Relax ray-submit launch.
#
# Usage:
#   bash examples/cot_compression/run_relax_qwen3_4b_thinking_sft_8xGPU.sh
#

set -ex
set -o pipefail

unset NCCL_NVLS_ENABLE

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

# Paths (override via env if needed)
RELAX="${RELAX}"
MEGATRON="${MEGATRON:-/root/Megatron-LM/}"

MODEL_DIR="${MODEL_DIR}"
TMP_DIR="${TMP_DIR}"
PROMPT_DATA="${PROMPT_DATA}"

EXP_NAME="${EXP_NAME:-qwen3-4b-thinking-sft-k115-llmgrading-relax-gpu8}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/k1.15_llmgrading}"
SAVE_ROOT="${SAVE_ROOT:-${TMP_DIR}/output/Qwen3-4b-Think/full/sft_k1.15_llmgrading_relax}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${EXP_NAME}}"

# Source the standard Relax local entrypoint (Ray head, env vars, RUNTIME_ENV_JSON).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi

RAY_ADDRESS="${RAY_ADDRESS:-http://${MASTER_ADDR:-127.0.0.1}:8265}"

MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 5000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
   --seq-length 32768
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load      "${MODEL_DIR}"
   --load          "${MODEL_DIR}"
   --save          "${SAVE_DIR}"

   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --save-interval 50
   --num-epoch 6
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key instruction
   --label-key output

   --global-batch-size 32
   --use-dynamic-batch-size

   --max-tokens-per-gpu 32768
   --balance-data
   --sft-prefetch-num-workers 8
   --sft-oversize-strategy truncate_right
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --data-parallel-sharding-strategy optim
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --min-lr 1e-6
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.999
   --adam-eps 1e-8
   --clip-grad 1.0
   --no-rope-fusion
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --bf16
   --attention-backend flash
   --cross-entropy-fusion-impl te
   --log-interval 1
   --distributed-timeout-minutes 3000000
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --recompute-loss-function
   --log-probs-chunk-size 256
   --use-health-check
   --no-save-rng
)

mkdir -p "${SAVE_DIR}" "${TMP_DIR}/log"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --max-staleness 1 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${TMP_DIR}/log/${EXP_NAME}-${now}.log"
