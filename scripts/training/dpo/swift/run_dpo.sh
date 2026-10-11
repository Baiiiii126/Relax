#!/bin/bash
# ms-swift side of the Relax <-> swift DPO alignment experiment.
#
# Mirrored by
#   scripts/training/dpo/run-qwen36-35B-A3B-ultrafeedback-8xgpu.sh
#
# Model / data / hyper-parameters are identical on both sides:
#   model      Qwen3.6-35B-A3B (full fine-tuning)
#   data       UltraFeedback preference pairs, 4096 train / 512 eval
#   steps      100 (max_steps; 1600 pairs = 0.39 epoch) -- the comparison run
#   batch      per_device 2 x 8 GPUs = global 16
#   lr         1e-6, cosine over 100 steps, 5% warmup
#   max_length 2048 (99.8% of branches fit untruncated)
#   beta       0.1
#   shuffle    OFF on both sides -- see below
#   clearml    project DPO-Align/ultrafeedback
#
# Why shuffle is off: swift shuffles with data_seed=42 via HF datasets +
# a torch sampler; Relax shuffles via its own two-stage IndexManager
# (RandomState seed -> permutation, then torch.randperm(epoch)). Neither can
# reproduce the other's permutation, and at 100 steps the two runs would
# consume different 1600-row SUBSETS, not just different orders. Row order is
# the only common ground, so both sides disable shuffling and consume the
# leading 1600 rows of an identically-ordered file (verified byte-identical).
#
# Known residual mismatch (cannot be closed from the config): swift computes
# the DPO logprobs in bf16 (lm_head logits are never upcast), so per-sequence
# logps land on a 0.5-1.0 nat grid, while Relax computes them in fp32. That
# puts a ~+/-0.05-0.1 floor on per-step loss agreement.
#
# Deviations from the reference swift script (Qwen2.5-7B / zh-en-emoji):
#   * model + dataset swapped to the ones used by the 200-step Relax run
#   * NPROC_PER_NODE 4 -> 8     (35B needs zero3 spread over 8 GPUs)
#   * learning_rate 1e-5 -> 1e-6 (keep the loss from collapsing early)
#   * rpo_alpha 0.1 -> 0         (Relax has no RPO term)
#   * max_length 8192 -> 2048
set -ex

# WORKSPACE_ROOT is the directory holding venv-swift/, models/, data/ and exps/.
# It is NOT derivable from this script's location: the script lives in the Relax
# repo, while the swift venv and the datasets live in the experiment workspace.
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

export CLEARML_PROJECT="${CLEARML_PROJECT:-DPO-Align/ultrafeedback}"
export CLEARML_TASK="${CLEARML_TASK:-swift-qwen3.6-35b-ultrafeedback}"

# qs_inject's model profiling instantiates the model with `from_config` on the
# meta device; under deepspeed zero3 that enters initialize_weights_zero3 and
# deadlocks in a cuda synchronize. Turn the profiler off (the rest of qs_inject
# stays active).
export QS_INJECT_MODEL_PROFILE="${QS_INJECT_MODEL_PROFILE:-off}"

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
"${ROOT}/venv-swift/bin/swift" rlhf \
    --rlhf_type dpo \
    --model "${ROOT}/models/Qwen3.6-35B-A3B" \
    --tuner_type full \
    --dataset "${ROOT}/data/ultrafeedback/swift_train.jsonl" \
    --val_dataset "${ROOT}/data/ultrafeedback/swift_eval.jsonl" \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --beta 0.1 \
    --eval_steps 25 \
    --save_strategy no \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir "${ROOT}/exps/swift-dpo-align" \
    --dataloader_num_workers 2 \
    --dataset_num_proc 1 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --padding_free true \
    --report_to clearml \
    --run_name "${CLEARML_TASK}" \
    2>&1 | tee "${ROOT}/exps/swift-dpo-align.log"
