#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU chunkflow correctness A/B, one command:
#
#   bash scripts/training/text/run-qwen3-4B-8xgpu-chunkflow.sh
#
# Runs three phases back to back on one Ray cluster:
#
#   1. dump      rollout only, saved to $ROLLOUT_DUMP_DIR
#   2. baseline  train on that dump with chunkflow OFF, saving per-step grad norms
#   3. chunk     train on the same dump with chunkflow ON, asserting each grad
#                norm matches the baseline's (--ci-load-grad-norm, rel_tol 1%)
#
# Phase 3 aborts the script if any grad norm diverges, so the run itself is the
# verdict -- no log eyeballing required. The per-step `step N: {...}` lines from
# phases 2 and 3 are diffed at the end for the finer-grained picture.
#
# The rollout is done once and replayed because sampling runs at temperature 1:
# two ordinary RL runs would train on different data and the loss gap would be
# sampling noise, not chunkflow. --load-debug-rollout-data implies
# --debug-train-only, so sglang never starts in phases 2 and 3 and both arms are
# actor-only, differing in exactly one flag pair.
#
# Single phases, if a rerun is needed:
#   MODE=dump  ... ; MODE=train CHUNK_SIZE=0 ... ; MODE=train CHUNK_SIZE=2048 ...
#
# This checks correctness only. Chunked attention does not reduce memory or
# pipeline bubbles on its own; that needs the per-chunk backward schedule.

set -e
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

# Both train phases must read the same dump and compare the same grad norms.
ROLLOUT_DUMP_DIR="${ROLLOUT_DUMP_DIR:-${EXP_DIR}/chunkflow-rollout-dump}"
GRAD_NORM_DIR="${GRAD_NORM_DIR:-${EXP_DIR}/chunkflow-grad-norm}"
MODE="${MODE:=all}"
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

GRAD_NORM_TEMPLATE="${GRAD_NORM_DIR}/{role}_{rollout_id}_{step_id}.pt"

mkdir -p log "${ROLLOUT_DUMP_DIR}" "${GRAD_NORM_DIR}"

run_phase() {
   local phase="$1"
   local arm resource
   local -a mode_args

   case "${phase}" in
      dump)
         arm="dump"
         resource='{"rollout": [1, 8]}'
         mode_args=(
            --debug-rollout-only
            --save-debug-rollout-data "${ROLLOUT_DUMP_DIR}/{rollout_id}.pt"
         )
         ;;
      baseline)
         arm="baseline"
         resource='{"actor": [1, 8]}'
         mode_args=(
            --load-debug-rollout-data "${ROLLOUT_DUMP_DIR}/{rollout_id}.pt"
            --ci-save-grad-norm "${GRAD_NORM_TEMPLATE}"
         )
         ;;
      chunk)
         arm="chunk${CHUNK_SIZE}"
         resource='{"actor": [1, 8]}'
         # --ci-load-grad-norm asserts against the baseline's grad norms, so a
         # divergence aborts this phase instead of hiding in a log.
         mode_args=(
            --load-debug-rollout-data "${ROLLOUT_DUMP_DIR}/{rollout_id}.pt"
            --ci-load-grad-norm "${GRAD_NORM_TEMPLATE}"
            --use-chunkflow
            --chunkflow-chunk-size ${CHUNK_SIZE}
         )
         ;;
      *)
         echo "unknown phase: ${phase}" >&2
         return 1
         ;;
   esac

   local log_file="log/qwen3-4b-chunkflow-${arm}-gpu8-${now}.log"
   echo "=== chunkflow phase: ${phase} -> ${log_file} ==="
   ( set -x
   ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
      ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 -m relax.entrypoints.train \
      --resource "${resource}" \
      --max-staleness 0 \
      --num-data-storage-units 1 \
      --colocate \
      --skip-eval-before-train \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${GRPO_ARGS[@]}" \
      --use-clearml \
      --use-metrics-service \
      --tb-project-name "${PROJECT_NAME}" \
      --tb-experiment-name "qwen3-4b-chunkflow-${arm}-gpu8-${now}" \
      "${PERF_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${mode_args[@]}" \
      "${MISC_ARGS[@]}" ) 2>&1 | tee "${log_file}"

   if [ "${phase}" = "chunk" ]; then
      if ! grep -q "chunked attention installed" "${log_file}"; then
         echo "FAIL: chunkflow never installed -- this arm proves nothing." >&2
         return 1
      fi
   fi
}

case "${MODE}" in
   all)
      run_phase dump
      run_phase baseline
      run_phase chunk
      echo "=== per-step metrics: baseline vs chunk${CHUNK_SIZE} ==="
      diff -y --width 200 \
         <(grep -oE "step [0-9]+: \{.*\}" "log/qwen3-4b-chunkflow-baseline-gpu8-${now}.log") \
         <(grep -oE "step [0-9]+: \{.*\}" "log/qwen3-4b-chunkflow-chunk${CHUNK_SIZE}-gpu8-${now}.log") \
         || true
      echo "=== PASS: grad norms matched the baseline at every step ==="
      ;;
   dump) run_phase dump ;;
   train)
      if [ "${CHUNK_SIZE}" -gt 0 ]; then run_phase chunk; else run_phase baseline; fi
      ;;
   *)
      echo "MODE must be one of: all, dump, train" >&2
      exit 1
      ;;
esac
