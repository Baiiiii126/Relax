#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Drive the ChunkFlow step-scheduling benefit A/B: for each sweep point, run the
# baseline arm once and the chunkflow arm for every chunk size. See
# docs/draft/chunkflow-scheduling-benefit.md for the theory and how to read the
# results.
#
# Usage:
#   DRY_RUN=1 bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh   # print, do not submit
#   bash scripts/training/sft/tools/run_chunkflow_benefit_sweep.sh
#
# Env:
#   MODEL_DIR / DATA_DIR   required by the run script (see it for defaults)
#   DATA_PROFILE           tail (default), longtail, or smoke
#   POINTS                 override the selected matrix; "gbs:capacity:steps:chunk,chunk,..."
#   NUM_ROLLOUT            NOT used here -- each point sets its own step count
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/../run-qwen3-4B-8xgpu-chunkflow.sh"
ENTRYPOINT="${SCRIPT_DIR}/../../../entrypoint/ray-job.sh"

if [ ! -f "${RUN_SCRIPT}" ]; then
   echo "ERROR: ${RUN_SCRIPT} not found" >&2
   exit 1
fi
if [ ! -f "${ENTRYPOINT}" ]; then
   echo "ERROR: ${ENTRYPOINT} not found" >&2
   exit 1
fi

# gbs:capacity:steps:chunk,chunk,...
#   gbs      samples per optimizer step  -> sets M together with capacity
#   capacity max-tokens-per-gpu (bin capacity)
#   steps    optimizer steps to run; the run script does ONE step per rollout,
#            so --num-rollout IS the step count. 1 is useless for perf, use >= 5.
#   chunks   chunk sizes; 0 = baseline arm, always run first
#
# Capacity stays at SEQ_LENGTH (32768). M is always read from telemetry; the
# values below deliberately avoid assuming an M from GBS alone.
DATA_PROFILE="${DATA_PROFILE:-tail}"
case "${DATA_PROFILE}" in
   tail)
      # 16K-32K controlled samples: expose the maximum scheduling signal first.
      DEFAULT_POINTS="1:32768:35:0,512,1024,2048,4096 4:32768:20:0,1024 16:32768:10:0,1024"
      ;;
   longtail)
      # Realistic 0-32K mix: validate whether the gain survives ordinary data.
      DEFAULT_POINTS="1:32768:35:0,1024 32:32768:20:0,1024 100:32768:10:0,1024"
      ;;
   smoke)
      DEFAULT_POINTS="1:32768:3:0,1024"
      ;;
   *)
      echo "ERROR: DATA_PROFILE must be tail, longtail, or smoke; got '${DATA_PROFILE}'" >&2
      exit 2
      ;;
esac
POINTS="${POINTS:-${DEFAULT_POINTS}}"

DRY_RUN="${DRY_RUN:-}"

# Every arm must finish before the next one starts. ray-job.sh performs
# pre-flight cleanup, so an asynchronous sweep would stop its preceding arm.
if [ -z "${DRY_RUN}" ] && [ -n "${RAY_NO_WAIT:-}" ]; then
   echo "ERROR: unset RAY_NO_WAIT for a sweep; A/B arms must run sequentially" >&2
   exit 2
fi

run_one() {
   local gbs="$1" capacity="$2" steps="$3" chunk="$4"
   local mode exp_name
   if [ "${chunk}" -gt 0 ]; then
      mode=chunkflow
   else
      mode=baseline
   fi
   exp_name="cfbenefit-gbs${gbs}-cap${capacity}-chunk${chunk}-s${steps}"

   echo "==================================================================="
   echo " gbs=${gbs} capacity=${capacity} steps=${steps} chunk=${chunk} mode=${mode}"
   echo " EXP_NAME=${exp_name}"
   echo "==================================================================="

   if [ -n "${DRY_RUN}" ]; then
      echo "  (dry run) EXP_NAME=${exp_name} GBS=${gbs} MAX_TOKENS_PER_GPU=${capacity} \\"
      echo "            NUM_ROLLOUT=${steps} CHUNK_SIZE=${chunk} \\"
      echo "            bash ${ENTRYPOINT} ${RUN_SCRIPT} ${mode}"
      return 0
   fi

   EXP_NAME="${exp_name}" \
   GBS="${gbs}" \
   MAX_TOKENS_PER_GPU="${capacity}" \
   NUM_ROLLOUT="${steps}" \
   CHUNK_SIZE="${chunk}" \
      bash "${ENTRYPOINT}" "${RUN_SCRIPT}" "${mode}"
}

failed=0
for point in ${POINTS}; do
   IFS=: read -r gbs capacity steps chunks <<<"${point}"
   if [ -z "${gbs:-}" ] || [ -z "${capacity:-}" ] || [ -z "${steps:-}" ] || [ -z "${chunks:-}" ]; then
      echo "ERROR: bad POINTS entry '${point}' (want gbs:capacity:steps:chunk,chunk,...)" >&2
      exit 2
   fi

   # Baseline first so both arms of a point run back to back on the same machine
   # state, then each chunkflow arm.
   for chunk in ${chunks//,/ }; do
      if ! run_one "${gbs}" "${capacity}" "${steps}" "${chunk}"; then
         echo "WARN: run failed for gbs=${gbs} capacity=${capacity} chunk=${chunk}" >&2
         failed=$((failed + 1))
      fi
   done
done

if [ "${failed}" -gt 0 ]; then
   echo "sweep finished with ${failed} failed run(s)" >&2
   exit 1
fi
echo "sweep finished cleanly"
