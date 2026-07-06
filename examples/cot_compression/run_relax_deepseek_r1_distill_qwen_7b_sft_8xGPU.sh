#!/bin/bash
#
# DeepSeek-R1-Distill-Qwen-7B SFT on k1.15_llmgrading.json, 8xGPU,
# Relax ray-submit launch.
#
# Usage:
#   bash examples/cot_compression/run_relax_deepseek_r1_distill_qwen_7b_sft_8xGPU.sh


set -ex
set -o pipefail

unset NCCL_NVLS_ENABLE

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

# Paths (override via env if needed)
MEGATRON="${MEGATRON:-/root/Megatron-LM/}"

MODEL_DIR="${MODEL_DIR}"
TMP_DIR="${TMP_DIR}"
PROMPT_DATA="${PROMPT_DATA}"

EXP_NAME="${EXP_NAME:-deepseek-r1-distill-qwen-7b-sft-k115-llmgrading-relax-gpu8}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/k1.15_llmgrading}"
SAVE_ROOT="${SAVE_ROOT:-${TMP_DIR}/output/DeepSeek-R1-Distill-Qwen-7B/full/sft_k1.15_llmgrading_relax}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${EXP_NAME}}"

# Source the standard Relax local entrypoint (Ray head, env vars, RUNTIME_ENV_JSON).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi

RAY_ADDRESS="${RAY_ADDRESS:-http://${MASTER_ADDR:-127.0.0.1}:8265}"

# Compatible with DeepSeek chat template.
SFT_CHAT_TEMPLATE_FILE="${TMP_DIR}/.deepseek_r1_sft_chat_template.jinja"
cat > "${SFT_CHAT_TEMPLATE_FILE}" <<'JINJA'
{%- if messages and messages[0]['role'] == 'system' -%}{{- messages[0]['content'] -}}{%- endif -%}{{- bos_token -}}{%- for message in messages -%}{%- if message['role'] == 'user' -%}{{- '<｜User｜>' + message['content'] -}}{%- endif -%}{%- if message['role'] == 'assistant' -%}{{- '<｜Assistant｜>' -}}{% generation %}{{- message['content'] + '<｜end▁of▁sentence｜>' -}}{% endgeneration %}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<｜Assistant｜>' -}}{%- endif -%}
JINJA
APPLY_CHAT_TEMPLATE_KWARGS=$(python3 -c '
import json, sys
tmpl = open(sys.argv[1]).read().rstrip("\n")
sys.stdout.write(json.dumps({"chat_template": tmpl}, ensure_ascii=False))
' "${SFT_CHAT_TEMPLATE_FILE}")

MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 3584
   --ffn-hidden-size 18944
   --num-attention-heads 28
   --group-query-attention
   --num-query-groups 4
   --use-rotary-position-embeddings
   --disable-bias-linear
   # Qwen2-only: keep bias on QKV projections (drops out the silent ~0-shift
   # bug from MLP/output bias getting re-added).
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --vocab-size 152064
   --kv-channels 128
   --untie-embeddings-and-output-weights
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

   # Override the model's native chat_template.
   --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"

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
