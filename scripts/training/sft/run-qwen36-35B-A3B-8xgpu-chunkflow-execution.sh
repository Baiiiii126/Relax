#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Three sequential eight-GPU runs: baseline, original ChunkFlow, optimized ChunkFlow.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
if [ "$#" -ne 0 ]; then echo "Usage: bash $0" >&2; exit 2; fi
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the existing model directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the existing LongAlign file}"
test -f "${HF_CHECKPOINT}/config.json"
test -f "${PROMPT_DATA}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || [ "${NUM_ROLLOUT}" -le 30 ]; then
   echo "ERROR: NUM_ROLLOUT must exceed 30 warmup steps" >&2
   exit 2
fi

export GBS=8 NUM_GPUS=8 TP_SIZE=2 PP_SIZE=4 EP_SIZE=2
export CHUNK_SIZE=8192 CHUNK_SIZES='' KEEP_LAST_K=0 PACK_POLICY=staggered
export SEQ_LENGTH=32768 MAX_TOKENS_PER_GPU=32768 TRAIN_SEED=1235
export LEARNING_RATE=1e-5 GDN_RECOMPUTE=1 CPU_OPTIMIZER_OVERLAP=1 MOE_SHARED_EXPERT_OVERLAP=0
export QWEN36_ROUTER_DTYPE=fp64 RELAX_CHUNKFLOW_FLASH_BACKEND=fa3
export RELAX_CHUNKFLOW_PARENT_WINDOW=0 QWEN36_TAIL_POOL_AB=0
export RELAX_MOE_METADATA_MODE=off RELAX_MOE_CHUNK_MODE=off
export QWEN36_EXECUTION_AB=1 RAY_NO_WAIT=0
export CHUNKFLOW_PROFILE="${CHUNKFLOW_PROFILE:-0}"
if [ "${CHUNKFLOW_PROFILE}" = 0 ]; then
   export QWEN36_MEMORY_CHECK=1 QWEN36_DEVICE_MEMORY_CHECK=1
else
   export QWEN36_MEMORY_CHECK=0 QWEN36_DEVICE_MEMORY_CHECK=0
fi
export INPUT_KEY="${INPUT_KEY:-messages}" LABEL_KEY="${LABEL_KEY-}"
unset QWEN36_GATE_GAIN_PCT RELAX_CHUNKFLOW_GATE_GAIN_PCT RELAX_ENTRYPOINT_MODE
unset TORCH_DISTRIBUTED_DEBUG NCCL_DEBUG CUDA_LAUNCH_BLOCKING
export QWEN36_RESULT_DIR="${QWEN36_RESULT_DIR:-$(pwd)/analysis/qwen36-chunkflow-execution-$(date +%Y%m%d-%H%M%S)-$$}"
echo "Three arms; ${NUM_ROLLOUT} steps each; chunk=8192; router=fp64; precision settings unchanged."
echo "Logs stream to this terminal and are saved in ${QWEN36_RESULT_DIR}/*.log"
echo "No loss, memory or speed acceptance gates. Compare step 30 onward in execution-vs-*-diagnosis.md."
bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
