#!/usr/bin/env bash
# Submit the float Linformer training. Reads site values from polaris/env.sh so the
# PBS headers stay portable.
#
#   bash polaris/submit_linformer.sh              # full 200-epoch run
#   SMOKE=1 bash polaris/submit_linformer.sh      # 10-minute gate on the debug queue
#
# NEVER put a trailing comment on a #PBS line -- qsub parses the rest of the line as args.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/polaris/env.sh"

if [ "${SMOKE:-0}" = "1" ]; then
  QUEUE="${QUEUE_DEBUG:-debug}"; WALL="00:30:00"; NAME="clic-linformer-smoke"
  EXTRA="-v REPO_DIR=${REPO_DIR},SMOKE=1"
else
  QUEUE="${QUEUE_PROD:-preemptable}"; WALL="${WALLTIME:-48:00:00}"; NAME="clic-linformer"
  EXTRA="-v REPO_DIR=${REPO_DIR}${RESUME_CKPT:+,RESUME_CKPT=${RESUME_CKPT}}"
fi

mkdir -p "${REPO_DIR}/polaris/logs"
set -x
qsub -A "${PBS_PROJECT}" \
     -q "${QUEUE}" \
     -l "filesystems=${PBS_FILESYSTEMS}" \
     -l "walltime=${WALL}" \
     -N "${NAME}" \
     -o "${REPO_DIR}/polaris/logs/${NAME}.log" \
     ${EXTRA} \
     "${REPO_DIR}/polaris/linformer_train.pbs"
