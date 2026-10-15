#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Simulated sample proportions, not measured production statistics.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR containing Qwen3-4B}"
: "${DATA_DIR:?Set DATA_DIR for the generated dataset}"
python3 "${SCRIPT_DIR}/build_chunkflow_longtail_sft.py" \
   --tokenizer "${MODEL_DIR}/Qwen3-4B" \
   --output "${DATA_DIR}/chunkflow-mixed/train.parquet" \
   --synthetic-only --seed 1234 \
   --short-count 3500 --medium-count 1000 --long-count 400 --tail-count 100 \
   --synthetic-ultralong-count 25
