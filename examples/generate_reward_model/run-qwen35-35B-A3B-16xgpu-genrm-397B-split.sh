#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B 16xGPU (2-node) fully colocate training script for DAPO math
# dataset, using Qwen3.5-397B-A17B as the Generative Reward Model (GenRM).
#
# Mode: SPLIT-BUNDLE — rollout on 8 GPUs, GenRM on the other 8 GPUs. Both run
# in parallel; inline reward is fired as each sample finishes generating.
# Suitable for long-tail / agentic rollouts where the GenRM's spare GPUs would
# otherwise be idle waiting for the tail. See README for tradeoffs vs the
# defer/swap script.
#
# Usage:
#   bash examples/generate_reward_model/run-qwen35-35B-A3B-16xgpu-genrm-397B-split.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math-genrm}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   # --load ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu_genrm397B_split/
   --save ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu_genrm397B_split/
   --save-interval 1000
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo-genrm
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --use-fault-tolerance
   --balance-data
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   # --log-probs-max-tokens-per-gpu 40960

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
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

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   # NOTE(wuhuan): to avoid algorithm performance degradation
   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

# Rollout engine for the 35B-A3B actor. Runs on its dedicated 8-GPU bundle
# (split from GenRM's 8 GPUs via rollout_num_gpus + genrm_num_gpus == actor_total).
# Only shares GPUs with actor (which offloads during rollout), so mem_fraction
# can be generous.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

# GenRM engine (Qwen3.5-397B-A17B FP8) on dedicated 8-GPU bundle. Only shares
# GPUs with actor (which offloads during judging); rollout is on the other 8.
# TP=DP=EP=8: shared_expert per-partition = 1024/8 = 128, matches FP8 block_n.
# mem_fraction 0.85: 397B FP8 weights ~50GB/GPU + KV; actor CUDA ctx fits in
# the remaining ~15% since actor offloads its weights during judge phase.
# disable_cuda_graph: judge is short-generation only (max 1024 tokens, temp 0.1),
# so no meaningful throughput loss; avoids capture_bs=[0] assert when the KV pool
# gets squeezed to 0 by graph-buffer reservation.
GENRM_ARGS=(
   --genrm-model-path ${MODEL_DIR}/Qwen3.5-397B-A17B-FP8/
   --genrm-num-gpus 8
   --genrm-num-gpus-per-engine 8
   --genrm-engine-config '{"context_length": 10240, "mem_fraction_static": 0.85, "dp_size": 8, "ep_size": 8, "enable_dp_attention": true, "moe_dense_tp_size": 1, "enable_dp_lm_head": true, "disable_cuda_graph": true, "server_concurrency": 1024, "watchdog_timeout": 3600}'
   --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 32, "chat_template_kwargs": {"enable_thinking": false}}'
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-35B-A3B-GRPO-GenRM397B-split-gpu16-${now}
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

# Tune NCCL/DeepEP timeouts for the heavier 397B GenRM engine.
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

# ============================================
# Resource Configuration (SPLIT-BUNDLE):
# 16 GPU (2 nodes). Actor runs on all 16; rollout and GenRM split into two
# dedicated 8-GPU bundles (rollout_num_gpus + genrm_num_gpus == actor_total
# triggers split-bundles colocate in relax/utils/arguments.py:2887).
#   - rollout (35B-A3B) on 8 GPUs, TP=8 SGLang engine
#   - GenRM   (397B-A17B FP8) on the other 8 GPUs, TP=DP=EP=8, dp-attention
# Reward is fired inline per-sample during rollout (rm-type=dapo-genrm), so
# GenRM overlaps with rollout generation. Best when the rollout has long tails
# (agentic multi-turn, high variance response length) — the tail hides GenRM
# latency and the two 8-GPU shards stay busy in parallel.
# ============================================

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 8], "genrm": [1, 8]}' \
   --colocate \
   --max-staleness 0 \
   --rollout-num-gpus 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${GENRM_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-35B-A3B-GRPO-GenRM397B-split-gpu16-${now}.log
