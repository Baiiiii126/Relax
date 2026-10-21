#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
if [ "$#" != 1 ]; then
   echo "Usage: bash $0 {numerics|profile}" >&2
   exit 2
fi
export EP_SIZE="${EP_SIZE:-2}" CHUNK_SIZE="${CHUNK_SIZE:-8192}" GBS="${GBS:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-40}" TRAIN_SEED="${TRAIN_SEED:-1235}"
export GDN_RECOMPUTE="${GDN_RECOMPUTE:-1}" CPU_OPTIMIZER_OVERLAP="${CPU_OPTIMIZER_OVERLAP:-1}"
case "$1" in
   numerics)
      export LEARNING_RATE=0 QWEN36_DIAGNOSTICS=1
      export RELAX_CHUNKFLOW_GRAD_ROLLOUTS=0,1,2,8
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
      ;;
   profile)
      export LEARNING_RATE=1e-5 PROFILE_START=28 PROFILE_END=32
      export PROFILE_RUN_ID="qwen36-profile-$(date +%Y%m%d-%H%M%S)-$$"
      export REPORT_DIR="$(pwd)/analysis/${PROFILE_RUN_ID}"
      exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-profile.sh"
      ;;
   *) echo "ERROR: mode must be numerics or profile" >&2; exit 2 ;;
esac
