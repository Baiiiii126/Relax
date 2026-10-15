#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Focused attribution experiment. Run from the repository root with MODEL_DIR
# and DATA_DIR (or PROMPT_DATA) set. DRY_RUN=1 prints commands without submission.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/../run-qwen3-4B-8xgpu-chunkflow.sh"
ENTRYPOINT="${SCRIPT_DIR}/../../../entrypoint/ray-job.sh"
STEPS="${STEPS:-510}"
GBS="${GBS:-8}"
DROP_FIRST="${DROP_FIRST:-10}"
REPORT_DIR="${REPORT_DIR:-analysis/chunkflow-attribution-$(date +%Y%m%d-%H%M%S)-$$}"

if ! [[ "${STEPS}" =~ ^[0-9]+$ && "${DROP_FIRST}" =~ ^[0-9]+$ ]] ||
   [ "${STEPS}" -le "${DROP_FIRST}" ]; then
   echo "ERROR: STEPS and DROP_FIRST must be nonnegative integers, with STEPS > DROP_FIRST" >&2
   exit 2
fi
if [ -n "${RAY_NO_WAIT:-}" ]; then
   echo "ERROR: unset RAY_NO_WAIT; attribution arms must finish sequentially" >&2
   exit 2
fi
if ! [[ "${GBS}" =~ ^[1-9][0-9]*$ ]]; then
   echo "ERROR: GBS must be a positive integer" >&2
   exit 2
fi
if [ -z "${PROMPT_DATA:-}" ]; then
   if [ -z "${DATA_DIR:-}" ]; then
      echo "ERROR: set DATA_DIR or PROMPT_DATA to the mixed-length dataset" >&2
      exit 2
   fi
   export PROMPT_DATA="${DATA_DIR}/chunkflow-mixed/train.parquet"
fi
if [ -z "${DRY_RUN:-}" ]; then
   if [ ! -e "${MODEL_DIR:-}/Qwen3-4B" ] || [ ! -f "${PROMPT_DATA}" ]; then
      echo "ERROR: MODEL_DIR/Qwen3-4B and PROMPT_DATA must exist" >&2
      exit 2
   fi
   mkdir -p -- "$(dirname -- "${REPORT_DIR}")"
   # Never append to or overwrite a previous experiment's artifacts.
   mkdir -- "${REPORT_DIR}" 2>/dev/null || {
      echo "ERROR: REPORT_DIR must be a new directory with an existing parent: ${REPORT_DIR}" >&2
      exit 2
   }
fi

# Keep the data, seed (1234 in the run script), layout and batch identical.
# 2048 was the best observed candidate; positive speedup is not yet established.
export GBS MAX_TOKENS_PER_GPU=32768 SEQ_LENGTH=32768
export TP_SIZE=2 PP_SIZE=4 PACK_POLICY=staggered KEEP_LAST_K=0
export NUM_ROLLOUT="${STEPS}"
export RELAX_CHUNKFLOW_SCHEDULE_ANALYSIS=1
export RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1
BASELINE_LOG="${REPORT_DIR}/baseline.log"
CHUNKFLOW_LOG="${REPORT_DIR}/chunk2048.log"

echo "GBS=${GBS} TP=2 PP=4 chunk=2048; ${STEPS} steps per arm; drop first ${DROP_FIRST} paired steps"
echo "PROMPT_DATA=${PROMPT_DATA}"
echo "REPORT_DIR=${REPORT_DIR}"
for mode in baseline chunkflow; do
   if [ "${mode}" = baseline ]; then
      export CHUNK_SIZE=0 LOG_FILE="${BASELINE_LOG}"
   else
      export CHUNK_SIZE=2048 LOG_FILE="${CHUNKFLOW_LOG}"
   fi
   export EXP_NAME="cfattribution-gbs${GBS}-cap32768-chunk${CHUNK_SIZE}-s${STEPS}"
   if [ -n "${DRY_RUN:-}" ]; then
      printf 'DRY RUN: EXP_NAME=%q CHUNK_SIZE=%q LOG_FILE=%q bash %q %q %q\n' \
         "${EXP_NAME}" "${CHUNK_SIZE}" "${LOG_FILE}" "${ENTRYPOINT}" "${RUN_SCRIPT}" "${mode}"
   else
      bash "${ENTRYPOINT}" "${RUN_SCRIPT}" "${mode}"
   fi
done

ANALYSIS=(python3 "${SCRIPT_DIR}/chunkflow_align.py"
   --baseline "${BASELINE_LOG}" --chunkflow "${CHUNKFLOW_LOG}"
   --drop-first "${DROP_FIRST}" --theory-source schedule
   --output-prefix "${REPORT_DIR}/attribution")
if [ -n "${DRY_RUN:-}" ]; then
   printf 'DRY RUN:'
   printf ' %q' "${ANALYSIS[@]}"
   printf '\n'
else
   # A successful submission alone does not prove that all requested steps ran.
   python3 - "${SCRIPT_DIR}/chunkflow_align.py" "${BASELINE_LOG}" "${CHUNKFLOW_LOG}" "${STEPS}" <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("chunkflow_align", sys.argv[1])
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)
rows = align._pair(Path(sys.argv[2]), Path(sys.argv[3]), "actor_train_time", "schedule")
if len(rows) != int(sys.argv[4]):
    raise SystemExit(f"ERROR: expected {sys.argv[4]} paired steps, found {len(rows)}; inspect this run's logs")
PY
   "${ANALYSIS[@]}" | tee "${REPORT_DIR}/summary.txt"
   echo "Read ${REPORT_DIR}/attribution.md for the attribution table."
fi
