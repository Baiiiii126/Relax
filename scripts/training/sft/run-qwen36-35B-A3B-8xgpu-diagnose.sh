#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
if [ "$#" != 1 ]; then
   echo "Usage: bash $0 {numerics|profile|localize|router64|performance|update|conv|tail|attention}" >&2
   exit 2
fi
if [ "$1" != attention ]; then unset QWEN36_KV_GRAD_AB; fi
if [ "$1" != tail ] && [ "$1" != attention ]; then unset RELAX_CHUNKFLOW_TAIL_TRACE; fi
if [ "$1" != update ]; then unset RELAX_CHUNKFLOW_UPDATE_TRACE; fi
if [ "$1" != conv ]; then unset RELAX_CHUNKFLOW_CONV_TRACE RELAX_CHUNKFLOW_CONV_CHUNK_SIZE; fi
export EP_SIZE="${EP_SIZE:-2}" CHUNK_SIZE="${CHUNK_SIZE:-8192}" GBS="${GBS:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-40}" TRAIN_SEED="${TRAIN_SEED:-1235}"
export GDN_RECOMPUTE="${GDN_RECOMPUTE:-1}" CPU_OPTIMIZER_OVERLAP="${CPU_OPTIMIZER_OVERLAP:-1}"
case "$1" in
   tail|attention)
      if [ "$1" = attention ]; then export QWEN36_KV_GRAD_AB=1; fi
      unset RELAX_CHUNKFLOW_LAYER_TRACE RELAX_CHUNKFLOW_CONV_TRACE RELAX_CHUNKFLOW_CONV_CHUNK_SIZE
      export NUM_ROLLOUT=1 LEARNING_RATE=0 QWEN36_DIAGNOSTICS=1 QWEN36_ROUTER_DTYPE=fp64
      export RELAX_CHUNKFLOW_GRAD_ROLLOUTS=0 RELAX_CHUNKFLOW_TAIL_TRACE=1 KEEP_LAST_K=0
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
      ;;
   conv)
      unset RELAX_CHUNKFLOW_LAYER_TRACE
      export NUM_ROLLOUT=1 LEARNING_RATE=0 QWEN36_DIAGNOSTICS=1 QWEN36_ROUTER_DTYPE=fp64
      export RELAX_CHUNKFLOW_GRAD_ROLLOUTS=0 RELAX_CHUNKFLOW_CONV_TRACE=1
      export RELAX_CHUNKFLOW_CONV_CHUNK_SIZE="${CHUNK_SIZE}"
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
      ;;
   update)
      unset RELAX_CHUNKFLOW_LAYER_TRACE
      export NUM_ROLLOUT=3 LEARNING_RATE=1e-5 QWEN36_DIAGNOSTICS=1 QWEN36_ROUTER_DTYPE=fp64
      export RELAX_CHUNKFLOW_GRAD_ROLLOUTS=0,1 RELAX_CHUNKFLOW_UPDATE_TRACE=1
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
      ;;
   performance)
      unset RELAX_CHUNKFLOW_LAYER_TRACE
      export NUM_ROLLOUT="${PERF_STEPS:-100}" LEARNING_RATE=1e-5 QWEN36_DIAGNOSTICS=1
      export RELAX_CHUNKFLOW_GRAD_ROLLOUTS=0
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
      ;;
   localize|router64)
      if [ "$1" = router64 ]; then export QWEN36_ROUTER_DTYPE=fp64; fi
      export NUM_ROLLOUT=1 LEARNING_RATE=0 QWEN36_DIAGNOSTICS=1 KEEP_LAST_K=0
      export RELAX_CHUNKFLOW_GRAD_ROLLOUTS=0 RELAX_CHUNKFLOW_LAYER_TRACE=1
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
      ;;
   numerics)
      unset RELAX_CHUNKFLOW_LAYER_TRACE
      export LEARNING_RATE=0 QWEN36_DIAGNOSTICS=1
      export RELAX_CHUNKFLOW_GRAD_ROLLOUTS=0,1,2,8
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
      ;;
   profile)
      unset RELAX_CHUNKFLOW_LAYER_TRACE
      export LEARNING_RATE=1e-5 PROFILE_START=28 PROFILE_END=32
      export PROFILE_RUN_ID="qwen36-profile-$(date +%Y%m%d-%H%M%S)-$$"
      export REPORT_DIR="$(pwd)/analysis/${PROFILE_RUN_ID}"
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-profile.sh"
      ;;
   *) echo "ERROR: mode must be numerics, profile, localize, router64, performance, update, conv, tail or attention" >&2; exit 2 ;;
esac
