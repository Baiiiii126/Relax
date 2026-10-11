#!/bin/bash
# ms-swift side of the Relax <-> swift 32xGPU "64k" DPO alignment experiment.
#
# Mirrored by
#   scripts/training/dpo/run-pevek-dpo-qwen36-35B-A3B-64k-32xgpu.sh
#
# Alignment contract (identical on both sides):
#   model       Qwen3.6-35B-A3B (full fine-tuning)
#   data        UltraFeedback preference pairs, 4096 train (leading rows)
#   global batch 64 pairs  (relax --global-batch-size 64)
#   steps       64 (1 epoch; 4096 / 64)
#   lr          1e-6, CONSTANT (no warmup) -- relax --lr-decay-style constant
#   beta        0.1
#   max_length  8192
#   shuffle     OFF on both sides
#   adam        beta1 0.9, beta2 0.98  -- relax defaults
#
# Why GBS is 64: ms-swift's global batch is
#   per_device_train_batch_size x world_size x gradient_accumulation_steps,
# all integers, so with 32 GPUs it must be a multiple of 32. The Relax side
# was set to 64 (32 * 2) so both sides can express the same value.
#
# Why shuffle is off: neither framework can reproduce the other's permutation,
# so the only common ground is consuming the leading rows of an identically
# ordered file (see swift/run_dpo_8xgpu_ultrafeedback.sh for the full note).
#
# Known residual mismatch (cannot be closed from the config): swift computes
# DPO logprobs in bf16 while Relax computes them in fp32, putting a
# ~+/-0.05-0.1 floor on per-step loss agreement.
#
# Multi-node: launch this on all 4 nodes (8 GPUs each = 32). Set NNODES,
# NODE_RANK, MASTER_ADDR, MASTER_PORT and WORKSPACE_ROOT on every node.
set -ex

# WORKSPACE_ROOT holds venv-swift/, models/, data/ and exps/.
ROOT="${WORKSPACE_ROOT:-}"
if [ -z "${ROOT}" ]; then
    echo "WORKSPACE_ROOT is not set." >&2
    echo "Point it at the directory containing venv-swift/, models/, data/, exps/, e.g." >&2
    echo "  WORKSPACE_ROOT=/mnt/tidal-nj01/usr/yuzhe1/baihaonan bash \"\${BASH_SOURCE[0]}\"" >&2
    exit 1
fi
for d in venv-swift models data; do
    [ -d "${ROOT}/${d}" ] || { echo "WORKSPACE_ROOT=${ROOT} has no ${d}/" >&2; exit 1; }
done

export NNODES="${NNODES:-4}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is not set}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export CLEARML_PROJECT="${CLEARML_PROJECT:-DPO-Align/ultrafeedback}"
export CLEARML_TASK="${CLEARML_TASK:-swift-qwen3.6-35b-uf-64k-32xgpu}"

# qs_inject's model profiling deadlocks under deepspeed zero3 (see the 8xgpu
# script). Turn the profiler off; the rest of qs_inject stays active.
export QS_INJECT_MODEL_PROFILE="${QS_INJECT_MODEL_PROFILE:-off}"

NPROC_PER_NODE=8 \
"${ROOT}/venv-swift/bin/swift" rlhf \
    --rlhf_type dpo \
    --model "${ROOT}/models/Qwen3.6-35B-A3B" \
    --tuner_type full \
    --dataset "${ROOT}/data/ultrafeedback/swift_train.jsonl" \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 64 \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --beta 0.1 \
    --save_strategy no \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir "${ROOT}/exps/swift-dpo-uf-64k-32xgpu" \
    --dataloader_num_workers 2 \
    --dataset_num_proc 1 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --padding_free true \
    --report_to clearml \
    --run_name "${CLEARML_TASK}" \
    2>&1 | tee "${ROOT}/exps/swift-dpo-uf-64k-32xgpu.log"
