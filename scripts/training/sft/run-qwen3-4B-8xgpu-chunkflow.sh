#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B SFT with a chunkflow toggle, 8xGPU single-node.
#
# SFT is the right vehicle for checking chunked attention: the data is fixed, so
# unlike RL (temperature-1 sampling) the two arms see identical tokens and the
# loss curves should overlay to within bf16 noise. Any real divergence is
# chunkflow's doing.
#
#   CHUNK_SIZE=0    bash scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh  # baseline
#   CHUNK_SIZE=1024 bash scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh  # chunkflow
#
# CHUNK_SIZE defaults to 1024 -- NOT 2048 -- and that matters. chunkflow splits
# per sequence, so a chunk size above a sample's length leaves it as a single
# chunk, making chunked attention a no-op: the curves would match trivially and
# prove nothing. OpenMathReasoning's CoT solutions run to thousands of tokens,
# so 1024 cuts every sample into several chunks. If you point PROMPT_DATA at
# short data instead (plain GSM8K is ~195 tokens/sample), drop the chunk size to
# ~128 for the same reason.
#
# One shard of OpenMathReasoning is enough (~22k samples); grab
#   data/cot-00000-of-00144.parquet
# from https://huggingface.co/datasets/nvidia/OpenMathReasoning and put it at
# $DATA_DIR/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet
# (the path the repo's other SFT scripts already expect).

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

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/chunkflow-sft}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=50}"

# 0 = baseline arm (chunkflow off). >0 = tokens per chunk.
CHUNK_SIZE="${CHUNK_SIZE:=1024}"
if [ "${CHUNK_SIZE}" -gt 0 ]; then
   CHUNKFLOW_ARGS=(--use-chunkflow --chunkflow-chunk-size ${CHUNK_SIZE})
   ARM="chunk${CHUNK_SIZE}"
else
   CHUNKFLOW_ARGS=()
   ARM="baseline"
fi

# Fail here rather than deep inside transformers: a non-existent --hf-checkpoint
# reaches ``cached_file`` as-is, which then treats the path as a Hub repo id and
# dies with "Repo id must be in the form 'repo_name' or 'namespace/repo_name'".
if [ ! -e "${MODEL_DIR}/Qwen3-4B" ]; then
   echo "ERROR: ${MODEL_DIR}/Qwen3-4B does not exist. Set MODEL_DIR / DATA_DIR, e.g." >&2
   echo "  export MODEL_DIR=/mnt/tidal-nj01/usr/yuzhe1/baihaonan/models" >&2
   echo "  export DATA_DIR=/mnt/tidal-nj01/usr/yuzhe1/baihaonan/data" >&2
   exit 1
fi
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
if [ ! -e "${PROMPT_DATA}" ]; then
   echo "ERROR: ${PROMPT_DATA} does not exist." >&2
   echo "Download one shard of nvidia/OpenMathReasoning (cot split) there, or set PROMPT_DATA." >&2
   exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   # No --save: this run only produces curves.
   --num-epoch 1
)

# --rollout-batch-size defaults to global_batch_size // n_samples_per_prompt
# (= 256 here), which makes args.train_iters == num_rollout, so NUM_ROLLOUT is
# exactly the optimizer-step count.
SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key problem
   --label-key generated_solution
   --global-batch-size 256
   --num-rollout ${NUM_ROLLOUT}
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --sft-oversize-strategy skip
   --balance-data
)

EVAL_ARGS=(
   --eval-size 0.01
   --eval-interval 50
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
   --cross-entropy-loss-fusion
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

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-sft-chunkflow-${ARM}-gpu8-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

mkdir -p log
LOG_FILE="log/qwen3-4b-sft-chunkflow-${ARM}-gpu8-${now}.log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${CHUNKFLOW_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_FILE}"

# A chunkflow arm whose patch never fired would look identical to the baseline
# and read as a pass, so check the arm was actually chunked.
if [ "${CHUNK_SIZE}" -gt 0 ] && ! grep -q "chunked attention installed" "${LOG_FILE}"; then
   echo "FAIL: chunkflow never installed -- this arm proves nothing." >&2
   exit 1
fi

# ClearML failures never fail the job, so confirm the curves actually went up.
if ! grep -q "ClearML adapter initialized" "${LOG_FILE}"; then
   echo "WARNING: ClearML adapter never initialized -- this run uploaded no curves." >&2
   grep -m1 "Failed to initialize ClearML adapter" "${LOG_FILE}" >&2 || true
   exit 1
fi
grep -m1 "Saving clearml log to project" "${LOG_FILE}" || true