#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.6-35B-A3B 32xGPU DPO (Direct Preference Optimization) 64k 长序列训练脚本.
# 目标集群: pevek. Ray 集群由外部管理 —— 不要 kill ray 或新建集群.
#
# 拓扑 (32 GPU): TP1 / PP2 / CP8 / EP8, DP = 32 / (1*2*8) = 2
#   40 层 -> 每 PP stage 20 层; 64k 序列经 CP8 切分为 8k/rank.
#
# Usage:
#   bash examples/algorithms/run-pevek-dpo-qwen36-35B-A3B-64k-32xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

# ---------------------------------------------------------------------------
# 集群与 ClearML
# ---------------------------------------------------------------------------
# Ray head 地址由 pevek 集群提供, 按实际环境覆盖.
export RAY_ADDRESS="${RAY_ADDRESS:?RAY_ADDRESS 未设置: 请指向 pevek 集群的 Ray head}"
export RAY_NO_WAIT="${RAY_NO_WAIT:-1}"

# MetricsService 可能落在别的节点, 必须用共享路径下的 clearml.conf.
export CLEARML_CONFIG_FILE="${CLEARML_CONFIG_FILE:-${SCRIPT_DIR}/../../clearml.conf}"
if [ ! -r "${CLEARML_CONFIG_FILE}" ]; then
    echo "ClearML config is not readable: ${CLEARML_CONFIG_FILE}" >&2
    exit 1
fi
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CLEARML_CONFIG_FILE"

PROJECT_NAME="${PROJECT_NAME:-Relax/dpo/qwen3.6-35B-A3B}"
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-dpo-64k-32xgpu}"
EXP_DIR="${EXP_DIR:-/mnt/tidalfs-bdsz01/dataset/pevek/baihaonan}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}/model}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}/data}"

mkdir -p "${EXP_DIR}/logs"

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
# TODO(agent): 该路径当前为空. DPO 需要成对的 chosen/rejected 偏好数据,
# 现有 SFTStreamingDataset (relax/engine/sft/dataset/streaming.py) 是单流数据集,
# 不能直接复用. 需要先准备数据并确认读取组件.
PREFERENCE_DATA="${DATA_DIR}/dpo/preference-64k.jsonl"
if [ ! -r "${PREFERENCE_DATA}" ]; then
    echo "偏好数据不存在: ${PREFERENCE_DATA}" >&2
    echo "DPO 需要 chosen/rejected 成对样本, 请先准备该数据集." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# DPO 超参
# ---------------------------------------------------------------------------
# TODO(agent): 以下参数在 Relax 中尚不存在, 需先实现 DPO 训练模式.
#   relax/core/registry.py:83-153 的 ALGOS 目前只有 grpo/gspo/sapo/cispo/m2po/
#     reinforce_plus_plus(_baseline)/rloo/sft/ppo —— 没有 dpo.
#   relax/utils/arguments.py:1867-1913 的 --loss-type 与 --advantage-estimator
#     的 choices 里也没有 dpo, 且没有 --dpo-beta.
# 补齐所需改动 (参考 relax/components/sft.py 的组织方式):
#   1) 新增 relax/components/dpo.py: 偏好对数据流 + DPO loss
#      loss = -log sigmoid( beta * [ (logp_w - logp_w_ref) - (logp_l - logp_l_ref) ] )
#   2) relax/core/registry.py: ALGOS 增加 "dpo" -> {dpo: DPO, reference: ActorFwd}
#   3) relax/utils/arguments.py: add_algo_arguments 增加 --loss-type dpo 与 --dpo-beta
# 在 1)-3) 落地前, 本脚本无法运行 (会在参数解析阶段报错).
DPO_ARGS=(
   --loss-type dpo
   --dpo-beta 0.1
   --preference-data "${PREFERENCE_DATA}"
   --reference-free 0
)

# ---------------------------------------------------------------------------
# 并行配置 (32 GPU)
# ---------------------------------------------------------------------------
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 2
   --context-parallel-size 8
   --expert-model-parallel-size 8
   --sequence-parallel

   --actor-num-nodes 4
   --actor-num-gpus-per-node 8

   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   # CP > 1 时 arguments.py 强制要求开启 (relax/utils/arguments.py:3389-3390)
   --calculate-per-token-loss

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-rope-fusion
   --attention-backend flash
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
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
