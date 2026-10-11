#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.6-35B-A3B 32xGPU DPO (Direct Preference Optimization) 64k 长序列训练脚本.
# 目标集群: pevek. Ray 集群由外部管理 —— 不要 kill ray 或新建集群.
#
# 拓扑 (32 GPU): TP1 / PP2 / CP16 / EP8, DP = 32 / (1*2*16) = 1
#   40 层 -> 每 PP stage 20 层. 数据集 UltraFeedback, 短句对,
#   目标是"每步 64k token": GBS=76 × ~840 tok/对 ≈ 63.8k.
#
# Usage:
#   bash examples/algorithms/run-pevek-dpo-qwen36-35B-A3B-64k-32xgpu.sh

set -ex
set -o pipefail

# 内网设了 http_proxy, 会把访问 Ray dashboard(127.0.0.1:8265)/集群内地址的请求
# 劫持到 squid 返回 503, 导致 ray-job.sh 枚举节点失败. 这里绕过代理.
export no_proxy="localhost,127.0.0.1,10.0.0.0/8,${no_proxy:-}"
export NO_PROXY="$no_proxy"

# DPO loss 需要把 [tokens, 248320] 的 fp32 logits 物化, 长序列下显存吃紧;
# 开 expandable_segments 减少碎片, 并透传到 Ray actor.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}PYTORCH_CUDA_ALLOC_CONF"

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_NAME="${PROJECT_NAME:-Relax/dpo/qwen3.6-35B-A3B}"
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-dpo-uf64ktok-32xgpu}"
EXP_DIR="${EXP_DIR:-/mnt/tidalfs-bdsz01/dataset/pevek/baihaonan}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}/model}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}/data}"

mkdir -p "${EXP_DIR}/logs"

# ---------------------------------------------------------------------------
# 集群与 ClearML
# ---------------------------------------------------------------------------
# Ray head 地址由 pevek 集群提供, 按实际环境覆盖.
export RAY_ADDRESS="${RAY_ADDRESS:?RAY_ADDRESS 未设置: 请指向 pevek 集群的 Ray head}"
export RAY_NO_WAIT="${RAY_NO_WAIT:-1}"

# MetricsService 可能落在别的节点, 必须用共享路径下的 clearml.conf.
export CLEARML_CONFIG_FILE="${CLEARML_CONFIG_FILE:-${EXP_DIR}/clearml.conf}"
if [ ! -r "${CLEARML_CONFIG_FILE}" ]; then
    echo "ClearML config is not readable: ${CLEARML_CONFIG_FILE}" >&2
    exit 1
fi
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CLEARML_CONFIG_FILE"

# ---------------------------------------------------------------------------
# 模型 / checkpoint
# ---------------------------------------------------------------------------
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.6-35B-A3B}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   # DPO 需要冻结的 reference policy 算 ref logprob; --ref-load 复用现成的
   # ActorFwd / ROLES.reference 机制 (relax/components/actor_fwd.py).
   --ref-load "${HF_CHECKPOINT}"
   --load "${EXP_DIR}/checkpoints/${EXP_NAME}"
   --save "${EXP_DIR}/checkpoints/${EXP_NAME}"
   --save-interval 50
   --max-actor-ckpt-to-keep 2
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

# ---------------------------------------------------------------------------
# 偏好数据 (chosen / rejected 成对样本)
# ---------------------------------------------------------------------------
# 由 scripts/data/prepare_longreward_preferences.py 从 LongReward-10k 生成,
# 走 Relax 的显式偏好 schema (prompt / chosen / rejected), 故 --input-key prompt.
PREFERENCE_DATA="${DATA_DIR}/ultrafeedback/ultrafeedback_train.jsonl"
if [ ! -r "${PREFERENCE_DATA}" ]; then
    echo "偏好数据不存在: ${PREFERENCE_DATA}" >&2
    echo "先运行: python3 scripts/data/prepare_longreward_preferences.py \\" >&2
    echo "          --input ${DATA_DIR}/LongReward-10k/dpo-glm4-9b.jsonl --output ${PREFERENCE_DATA}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# DPO 超参
# ---------------------------------------------------------------------------
# DPO 由 `38db5feb 【Task.31】feat(dpo): add pair-aware DPO training (#257)` 引入,
# 已内建: relax/engine/sft/dataset/preference.py 数据集、relax/backends/megatron/loss.py
# 的 dpo_loss_function、以及 --loss-type dpo / --dpo-beta.
# 64k 长序列所需的 CP/PP 支持在本分支补齐 (loss.py 的 `_preference_branch_sums`
# 与 runtime.py 的 `validate_preference_args`).
DPO_ARGS=(
   --loss-type dpo
   --dpo-beta 0.1
   --prompt-data "${PREFERENCE_DATA}"
   --input-key prompt
   # 单分支最长 ~60.4k token, 上限取 64k; 回答最长 ~1.4k, 给 4k 足够.
   --preference-max-length 8192
   --preference-max-completion-length 4096
)

# ---------------------------------------------------------------------------
# 并行配置 (32 GPU)
# ---------------------------------------------------------------------------
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 2
   --context-parallel-size 16
   --expert-model-parallel-size 8
   --sequence-parallel

   --actor-num-nodes 4
   --actor-num-gpus-per-node 8

   # 离线目标必须显式声明 resource: sft producer 是 CPU-only, actor 占满 32 卡.
   --resource '{"sft":[1,0],"actor":[1,32]}'
   # 每步的偏好"对"数 (一对 = chosen+rejected); DP=2, 每 rank 4 对.
   --global-batch-size 76
   # 正式跑 1 个 epoch (4096 对 / batch 76 ≈ 54 步).
   --num-epoch 1

   # UltraFeedback 短句对 (平均 ~840 tok/对); 每步 76 对 ≈ 64k token.
   --seq-length 8192
   --micro-batch-size 1
   --use-dynamic-batch-size
   # 偏好目标按"对"计: 一对 = chosen + rejected, prompt 计两次.
   # --max-tokens-per-gpu 是"每 rank"预算, microbatch 总预算 = 该值 * CP16.
   # 8192 * 16 = 131072 >= 最大的一对 (120.6k) → 没有样本被判 oversize.
   --max-tokens-per-gpu 4096

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-rope-fusion
   --attention-backend flash
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # 1e-6 是仓库对大模型训练的惯例 (scripts/training/text/ 下 51/53 用 1e-6);
   # 5e-7 是 0.6B DPO 示例的默认值, 对 35B 偏小.
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

# ---------------------------------------------------------------------------
# 指标 (ClearML)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 提交
# ---------------------------------------------------------------------------
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
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
   "${MISC_ARGS[@]}" 2>&1 | tee "${EXP_DIR}/logs/${EXP_NAME}-${now}.log"
