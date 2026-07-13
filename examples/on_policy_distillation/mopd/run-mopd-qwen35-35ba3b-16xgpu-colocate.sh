#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Multi-Teacher OPD (MOPD), 2-node 16 GPU colocate:
#   student : Qwen3.5-35B-A3B,  actor(16) + rollout(8) colocate, TP=4 PP=2 EP=8
#   teacher : one per data_source, shares the actor GPU pool (colocate)
#     dapo-math-17k       -> Qwen3.6-27B  (text, 4 GPU TP=4)
#     multimodal-open-r1  -> Qwen3.5-27B  (VL,   4 GPU TP=4)
#
# Colocate GPU layout (2 nodes × 8 GPU, one shared placement group):
#   training : student actor uses ALL 16 GPUs (TP=4 PP=2 EP=8, DP=2)
#   rollout  : GPU 0-7 student rollout (TP=8) | GPU 8-15 teachers
#   teachers : GPU 8-11 text Qwen3.6-27B (TP=4) | GPU 12-15 VL Qwen3.5-27B (TP=4)
# Constraint (enforced): rollout_gpus(8) + teacher_gpus(8) == actor_gpus(16).
# Teachers live inside the actor placement group and offload/onload in lock-step
# with training. Training the student on all 16 GPUs (TP=4 PP=2) is what fixes
# the 8-GPU grad-norm OOM — no dedicated teacher node needed.
#
# Per-sample rm_type is embedded in extra_info by prepare_data_35b.py;
# no global --rm-type is needed (fallback: "dapo").
#
# Usage:
#   bash run-mopd-qwen35-35ba3b-16xgpu.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_OPD_PREEXPANDED_PATCH"

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/mopd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mopd-qwen35-35ba3b-16xgpu-${now}}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-35B-A3B}"
TEXT_TEACHER_MODEL_NAME="${TEXT_TEACHER_MODEL_NAME:-Qwen3.6-27B}"
VL_TEACHER_MODEL_NAME="${VL_TEACHER_MODEL_NAME:-Qwen3.5-27B}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/MOPD-35B/train.parquet}"
# Derive eval set from PROMPT_SET's directory so overriding PROMPT_SET alone is
# enough. Use the small 50-sample balanced subset (25 text + 25 VL) for fast
# monitoring; point EVAL_SET at test.parquet for a full 1254-sample eval.
EVAL_SET="${EVAL_SET:-${PROMPT_SET%/*}/test_small.parquet}"

# GPU allocation (colocate: rollout + teacher == actor):
#   student colocate: 16 GPU actor (TP=4, PP=2, EP=8 → DP=2); rollout uses 8 GPU
#   teacher total:     8 GPU  (4 per teacher, TEACHER_NUM_GPUS_PER_ENGINE=4 → TP=4 each)
ACTOR_GPUS="${ACTOR_GPUS:-16}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
TEACHER_GPUS="${TEACHER_GPUS:-8}"
TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-4}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${STUDENT_MODEL_NAME}/"
   --megatron-to-hf-mode bridge
   # --save "${EXP_DIR}/save/mopd-${STUDENT_MODEL_NAME}/"
   # --save-interval 100
)

# global_batch_size = samples per optimizer step; must divide
# total_samples = rollout_batch_size(16) × n_samples_per_prompt(8) = 128.
# GBS=128 → num_rollout_minis=1; 128 samples / dp=2 = 64 per DP rank ✓.
# NOTE: GBS does NOT include DP (dp=ACTOR_GPUS/(TP×PP)=16/(4×2)=2).
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --apply-chat-template
   --rollout-shuffle

   --multimodal-keys '{"image":"images"}'

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128

   --log-passrate
   --use-fault-tolerance
   --use-streaming-dataset
)

# Teacher routes: data_source → HF checkpoint path.
# TEACHER_NUM_GPUS_PER_ENGINE=4 → each teacher gets 1 replica (TP=4).
TEACHER_ROUTES="{\"dapo-math-17k\":\"${MODEL_DIR}/${TEXT_TEACHER_MODEL_NAME}/\",\"multimodal-open-r1\":\"${MODEL_DIR}/${VL_TEACHER_MODEL_NAME}/\"}"

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-only-reward
   --opd-kl-coef 0.2
   --opd-per-token-clip 2.0
   --opd-teacher-key data_source
   --opd-teacher-routes "${TEACHER_ROUTES}"
   --teacher-num-gpus-per-engine "${TEACHER_NUM_GPUS_PER_ENGINE}"
   # student_sampled (default) keeps the teacher's per-position output to a single
   # logprob, so the 8192-token logprob forward is far lighter than top-k=64. That
   # frees enough memory to raise the static pool back to 0.65 for larger prefill
   # batching (avoids the 1-seq-at-a-time slowdown), while chunked-prefill=4096
   # bounds the per-chunk logits allocation to stay clear of OOM.
   --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.65}"
   --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
   --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-128}"
   --teacher-sglang-disable-cuda-graph
   --opd-log-prob-min-clamp -10.0
   --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-600}"
   --opd-teacher-image-key images
   --use-rollout-logprobs
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
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
   --accumulate-allreduce-grads-in-fp32
   --moe-router-load-balancing-type "none"
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXPERIMENT_NAME}"
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data mopd-35b "${EVAL_SET}"
   --eval-global-batch-size 128
   --n-samples-per-eval-prompt 1
   --eval-temperature 0.0
   --eval-max-response-len 8192
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

# Student on 16 GPU: TP=4, PP=2, EP=8, ETP=1, CP=1. Mirrors the proven text recipe
# scripts/training/text/run-qwen35-35B-A3B-16xgpu.sh:
#   - PP=2 splits the layers into 2 pipeline stages → ~half the per-GPU weight +
#     activation vs a single-stage layout (this is what removes the 8-GPU
#     grad-norm OOM).
#   - TP=4 shards each layer across 4 GPUs; EP=8 shards all 256 experts (32/GPU).
#   - DP = 16/(TP4×PP2) = 2.
# Recompute kept ON for extra headroom (MOPD adds the student-logprob forward).
# Teachers share the actor placement group (colocate); rollout SGLang: 1 engine
# × TP=8 over the front-8 rollout region.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # 10240 = max prompt (~2048) + max response (8192): just fits the longest
   # single sequence per microbatch without extra packing, minimizing peak
   # logits/activation memory. log-probs forward is capped the same so the OPD
   # student-logprob pass does not materialize an oversized [tokens, vocab] tensor.
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 10240
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   # 0.6 (down from 0.7): colocate leaves ~7.7GB of actor residual on each GPU
   # after offload; at 0.7 the static pool + residual left only ~7.4GB free and
   # the --use-rollout-logprobs logits tensor (~7.5GB for a large prefill batch)
   # OOMed by ~50MB at step 15. 0.6 frees ~8GB → ~2x headroom on that alloc.
   --sglang-mem-fraction-static 0.6
   # Hard cap on concurrent requests so the per-forward logprob logits tensor
   # (positions x vocab) stays bounded regardless of over-sampling burst.
   --sglang-max-running-requests 128
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
   --sglang-disable-cuda-graph
)

# Partial rollout: over-sample prompts and abort the slowest (longest) in-flight
# generations once the batch fills; aborted sequences continue next step
# (off-policy tokens masked). Caps the number of near-max-length (8192) sequences
# in each training microbatch -> bounds the optimizer-step activation peak that
# was OOMing grad-norm at step ~28, WITHOUT truncating responses (they finish
# across steps). Mirrors the working 35B GRPO reference. OPD-compatible: aborted
# samples skip teacher scoring.
PARTIAL_ROLLOUT_ARGS=(
   --partial-rollout
   --over-sampling-batch-size 24
   --mask-offpolicy-in-partial-rollout
   --partial-rollout-max-aborted-count 3
)

RESOURCE_JSON="{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}"

python3 -m relax.entrypoints.train \
    --resource "${RESOURCE_JSON}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --offload \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${PARTIAL_ROLLOUT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "mopd-qwen35-35ba3b-${now}.log"
