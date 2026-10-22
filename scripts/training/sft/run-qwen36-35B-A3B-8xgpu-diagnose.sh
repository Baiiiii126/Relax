#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ "$#" != 1 ] || [ "$1" != performance ]; then
   echo "Usage: bash $0 performance (legacy operator diagnostics removed)" >&2
   exit 2
fi
export NUM_ROLLOUT="${PERF_STEPS:-50}" LEARNING_RATE=1e-5
exec bash "${SCRIPT_DIR}/run-qwen36-35B-A3B-8xgpu-benefit.sh"
