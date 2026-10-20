#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Sequential experiments on the user's dedicated, idle 8-GPU Ray cluster.
# Usage: bash scripts/training/sft/tools/run_chunkflow_8gpu.sh screen
#        CHUNK_SIZE=4096 bash scripts/training/sft/tools/run_chunkflow_8gpu.sh confirm
#        CHUNK_SIZE=4096 bash scripts/training/sft/tools/run_chunkflow_8gpu.sh profile
#        CHUNK_SIZE=4096 bash scripts/training/sft/tools/run_chunkflow_8gpu.sh backend
#        CHUNK_SIZE=4096 GATE_GAIN_PCT=140 bash scripts/training/sft/tools/run_chunkflow_8gpu.sh gate
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../../.."
exec python3 "${SCRIPT_DIR}/chunkflow_8gpu.py" "$@"
