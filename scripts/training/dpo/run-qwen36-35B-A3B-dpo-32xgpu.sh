#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.6-35B-A3B DPO on 32 GPUs with TP1/PP2/CP2/DP8/EP8.
# Usage: bash scripts/entrypoint/spmd-multinode.sh \
#   scripts/training/dpo/run-qwen36-35B-A3B-dpo-32xgpu.sh

set -ex
set -o pipefail

export no_proxy="localhost,127.0.0.1,10.0.0.0/8,${no_proxy:-}"
export NO_PROXY="$no_proxy"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "$(cd -- "${SCRIPT_DIR}/../../entrypoint" && pwd)/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/dpo/qwen3.6-35B-A3B}"
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-dpo-32xgpu}"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${CLEARML_CONFIG_FILE:?Set CLEARML_CONFIG_FILE to a shared readable file}"
[ -r "${CLEARML_CONFIG_FILE}" ] || { echo "Unreadable CLEARML_CONFIG_FILE: ${CLEARML_CONFIG_FILE}" >&2; exit 1; }

if [ "${RELAX_ENTRYPOINT_MODE:-}" = "spmd-multinode" ]; then
    export RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-${MASTER_ADDR:-127.0.0.1}}:8265}"
    export RAY_NO_WAIT=""
else
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs HTTP endpoint}"
fi

RUNTIME_ENV_JSON="$(python3 -c '
import json, os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
names = {
    name.strip()
    for source in (env.get("RELAX_PROPAGATE_ENV_VARS", ""), os.environ.get("RELAX_PROPAGATE_ENV_VARS", ""))
    for name in source.split(",")
    if name.strip()
}
names.update(("PYTORCH_CUDA_ALLOC_CONF", "CLEARML_CONFIG_FILE"))
env.update({name: os.environ[name] for name in ("PYTORCH_CUDA_ALLOC_CONF", "CLEARML_CONFIG_FILE")})
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(sorted(names))
print(json.dumps(runtime))
')"
export RUNTIME_ENV_JSON

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.6-35B-A3B}"
PREFERENCE_DATA="${PREFERENCE_DATA:-${DATA_DIR}/ultrafeedback/ultrafeedback_train.jsonl}"
CHECKPOINT_DIR="${SAVE_DIR}/${EXP_NAME}"
[ -r "${PREFERENCE_DATA}" ] || {
    echo "Missing preference data: ${PREFERENCE_DATA}" >&2
    echo "Prepare it with scripts/data/prepare_ultrafeedback_preferences.py --output-dir <dir>." >&2
    exit 1
}

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --enable-weights-backuper
   --load "${CHECKPOINT_DIR}"
   --save "${CHECKPOINT_DIR}"
   --save-interval "${SAVE_INTERVAL:-25}"
   --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-2}"
)

DPO_ARGS=(
   --loss-type dpo
   --dpo-beta "${DPO_BETA:-0.1}"
   --prompt-data "${PREFERENCE_DATA}"
   --input-key prompt
   --no-dataset-shuffle
   --seed "${SEED:-42}"
   --preference-max-length 8192
   --preference-max-completion-length 4096
   --seq-length 8192
   --global-batch-size 64
   --num-epoch 1
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --use-gloo-process-groups
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 2
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --actor-num-nodes 4
   --actor-num-gpus-per-node 8
   --resource '{"sft":[1,0],"actor":[1,32]}'
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --no-rope-fusion
   --attention-backend flash
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-tensorboard
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${DPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
