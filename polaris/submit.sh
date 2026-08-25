#!/usr/bin/env bash
# Submit a job script with -A / -l filesystems= / -q filled in from polaris/env.sh,
# so those site-specific values live in exactly one place.
#
#   bash polaris/submit.sh 04_smoke.pbs
#   bash polaris/submit.sh 05_sweep.pbs
#   bash polaris/submit.sh 05_sweep.pbs --depend afterany:1234567.polaris
#
# Prints the job id. Use `qstat -u $USER` to watch, `qdel <id>` to cancel.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] || { echo "ERROR: create $HERE/env.sh from env.sh.example first"; exit 1; }
source "$HERE/env.sh"

# Required. Checked explicitly rather than relying on `set -u`, which would abort with a
# bare "unbound variable" and no hint about which file to edit.
for required in PBS_PROJECT PBS_FILESYSTEMS; do
  [ -n "${!required:-}" ] || { echo "ERROR: set $required in $HERE/env.sh (see 00_discover.sh)"; exit 1; }
done
[ "$PBS_PROJECT" = "CHANGEME" ] && { echo "ERROR: set PBS_PROJECT in polaris/env.sh (see 00_discover.sh)"; exit 1; }

SCRIPT="${1:?usage: submit.sh <script.pbs> [--depend <dep>]}"
[ -f "$HERE/$SCRIPT" ] || [ -f "$SCRIPT" ] || { echo "ERROR: no such script: $SCRIPT"; exit 1; }
[ -f "$HERE/$SCRIPT" ] && SCRIPT="$HERE/$SCRIPT"
shift || true

DEPEND=""
QUEUE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --depend) DEPEND="$2"; shift 2 ;;
    --queue)  QUEUE="$2";  shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# Queue and walltime come from env.sh, picked by script class. Until this existed the
# QUEUE_*/WALLTIME_* variables in env.sh were dead: -q was only ever passed when
# --queue was given explicitly, so the hardcoded #PBS -q in each header always won and
# editing env.sh had no effect. Command-line options override in-script directives, so
# the headers remain as standalone-qsub fallbacks.
# Defaulted, NOT required: an env.sh written before these knobs existed must keep working.
# Under `set -u` a bare reference would abort the submission outright.
BASE="$(basename "$SCRIPT")"
case "$BASE" in
  05_sweep.pbs) DEF_QUEUE="${QUEUE_PROD:-preemptable}"; DEF_WALL="${WALLTIME_PROD:-72:00:00}"  ;;
  *)            DEF_QUEUE="${QUEUE_DEBUG:-debug}";      DEF_WALL="${WALLTIME_SMOKE:-00:20:00}" ;;
esac
QUEUE="${QUEUE:-$DEF_QUEUE}"

# -o must be ABSOLUTE. A relative path in a #PBS directive is resolved against $HOME,
# not the submission directory, and an unwritable -o makes PBS fail the job, retry it,
# and eventually auto-hold it. (`~` does not expand in directives at all.)
LOGDIR="$(dirname "$HERE")/logs"
mkdir -p "$LOGDIR"

ARGS=(-A "$PBS_PROJECT" -l "filesystems=$PBS_FILESYSTEMS" -q "$QUEUE"
      -l "walltime=$DEF_WALL" -o "$LOGDIR/${BASE%.pbs}.log")
[ -n "$DEPEND" ] && ARGS+=(-W "depend=$DEPEND")

echo "qsub ${ARGS[*]} $SCRIPT"
qsub "${ARGS[@]}" "$SCRIPT"
