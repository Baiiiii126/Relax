#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_DIR}"
if [ "$#" -ne 0 ] || [ "${RAY_NO_WAIT:-0}" != 0 ]; then
   echo "ERROR: this comparison runs baseline then ChunkFlow sequentially; no arguments or RAY_NO_WAIT=1" >&2
   exit 2
fi
export CHUNKFLOW_PROFILE=1
export PROFILE_START="${PROFILE_START:-34}"
export PROFILE_END="${PROFILE_END:-36}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export TRAIN_SEED="${TRAIN_SEED:-1235}"
PROFILE_RUN_ID="${PROFILE_RUN_ID:-qwen36-profile-$(date +%Y%m%d-%H%M%S)-$$}"
REPORT_DIR="${REPORT_DIR:-analysis/${PROFILE_RUN_ID}}"
mkdir -p "${REPORT_DIR}"
REPORT_DIR="$(cd -- "${REPORT_DIR}" &>/dev/null && pwd)"
cat > "${REPORT_DIR}/profile.md" <<EOF
# Qwen3.6 baseline / ChunkFlow profiler

Commit: $(git rev-parse HEAD)
Seed: ${TRAIN_SEED}; steps: ${NUM_ROLLOUT}; profile rollouts: ${PROFILE_START}..${PROFILE_END} (0-based).
Both arms use the same training script and environment; only ChunkFlow is toggled.
These runs include profiler overhead and are for attribution, not speedup claims.

Shared outputs (all 8 ranks):
- ${REPORT_DIR}/baseline/train_trace/
- ${REPORT_DIR}/chunkflow/train_trace/

Each directory contains *.pt.trace.json.gz and segments_rank*_dp*_tp*_pp*.jsonl.
Segment times are inclusive CPU wall times, include waits, and overlap: do not sum them.
Use the trace's GPU kernels, NCCL and D2H/H2D copies for device-time attribution.
GDN compute includes selective recomputation; cpu_step includes its H2D post-hook.
Forward is the model callback; loss includes the deferred SFT logits/loss on the last PP stage.
EOF
for arm in baseline chunkflow; do
   RUN_NAME="${PROFILE_RUN_ID}-${arm}" LOG_FILE="${REPORT_DIR}/${arm}.log" \
      RELAX_TRAIN_PROFILE_DIR="${REPORT_DIR}/${arm}/train_trace" \
      bash scripts/entrypoint/ray-job.sh \
      scripts/training/sft/run-qwen36-35B-A3B-8xgpu-chunkflow.sh "${arm}"
   echo "${arm}: complete" >> "${REPORT_DIR}/profile.md"
done
echo "Profile comparison complete: ${REPORT_DIR}/profile.md"
