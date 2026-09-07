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
# CFG selects the arm; the job name follows it so the two runs do not share a log.
CFG="${CFG:-src/hepattn/experiments/clic/configs/linformer_polaris.yaml}"
ARM="$(basename "${CFG}" _polaris.yaml)"
source "${REPO_DIR}/polaris/env.sh"

if [ "${SMOKE:-0}" = "1" ]; then
  QUEUE="${QUEUE_DEBUG:-debug}"; WALL="00:30:00"; NAME="clic-${ARM}-smoke"
  EXTRA="-v REPO_DIR=${REPO_DIR},CFG=${CFG},SMOKE=1"
else
  QUEUE="${QUEUE_PROD:-preemptable}"; WALL="${WALLTIME:-48:00:00}"; NAME="clic-${ARM}"
  EXTRA="-v REPO_DIR=${REPO_DIR},CFG=${CFG}${RESUME_CKPT:+,RESUME_CKPT=${RESUME_CKPT}}${EPOCHS:+,EPOCHS=${EPOCHS}}"
fi

mkdir -p "${REPO_DIR}/polaris/logs"
# Truncate: PBS appends, so successive runs would otherwise blend into one file.
: > "${REPO_DIR}/polaris/logs/${NAME}.log"
set -x
qsub -A "${PBS_PROJECT}" \
     -q "${QUEUE}" \
     -l "filesystems=${PBS_FILESYSTEMS}" \
     -l "walltime=${WALL}" \
     -N "${NAME}" \
     -o "${REPO_DIR}/polaris/logs/${NAME}.log" \
     ${EXTRA} \
     "${REPO_DIR}/polaris/linformer_train.pbs"
