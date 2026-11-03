#!/usr/bin/env bash
# Submit one run: ARM=<ref|sec3|sec3ev>_s<42|43> bash submit.sh   (6 runs, see make_configs.py)
#   SMOKE=1     debug queue, 30 min, 200 train batches (step time and GPU memory)
#   RESUME_CKPT=<path>  continue a preempted run
# Site values come from ENV_SH, a file that exports PBS_PROJECT, PBS_FILESYSTEMS, DATA_ROOT (train/val ROOT files) and
# WORK_ROOT, and optionally QUEUE_DEBUG / QUEUE_PROD. VENV_DIR is a venv with this branch's torch / lightning /
# masked-linformer pins. Never put a comment on a #PBS line.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${HERE}/../../../../../../.." && pwd)"
ENV_SH="${ENV_SH:?set ENV_SH}"
VENV_DIR="${VENV_DIR:?set VENV_DIR}"
source "${ENV_SH}"
ARM="${ARM:?set ARM}"
CFG="studies/sector_decoder/configs/${ARM}.yaml"
[ -f "${REPO_DIR}/src/hepattn/experiments/clic/${CFG}" ] || { echo "no ${CFG}"; exit 1; }
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/sector/logs}"
if [ "${SMOKE:-0}" = "1" ]; then
  QUEUE="${QUEUE_DEBUG:-debug}"; WALL="00:30:00"; NAME="sector-${ARM}-smoke"
else
  QUEUE="${QUEUE:-${QUEUE_PROD:-preemptable}}"; WALL="${WALLTIME:-72:00:00}"; NAME="sector-${ARM}"
fi
mkdir -p "${HERE}/logs"
qsub -A "${PBS_PROJECT}" -q "${QUEUE}" -l "filesystems=${PBS_FILESYSTEMS}" -l "walltime=${WALL}" -r y \
  -N "${NAME}" -o "${HERE}/logs/${NAME}.log" \
  -v "REPO_DIR=${REPO_DIR},ENV_SH=${ENV_SH},VENV_DIR=${VENV_DIR},CFG=${CFG},SMOKE=${SMOKE:-0},LOG_ROOT=${LOG_ROOT}${RESUME_CKPT:+,RESUME_CKPT=${RESUME_CKPT}}" \
  "${HERE}/train.pbs"
