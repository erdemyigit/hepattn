# Sourced (after polaris/env.sh) by 01_setup_env.sh and every .pbs: the branch the work-space clone tracks, and the
# clone's directory. keras-hgq2 keeps the original $WORK_ROOT/hepattn; any other branch gets its own clone, so that
# setting up a new branch never checks out new code under jobs that run from the keras-hgq2 clone.
#   HEPATTN_BRANCH  branch of https://github.com/erdemyigit/hepattn to clone   (default keras-hgq2)
#   HEPATTN_DIR     clone directory              (default $WORK_ROOT/hepattn, or $WORK_ROOT/hepattn-<branch, / -> ->)
HEPATTN_BRANCH="${HEPATTN_BRANCH:-keras-hgq2}"
if [ -z "${HEPATTN_DIR:-}" ]; then
  if [ "$HEPATTN_BRANCH" = "keras-hgq2" ]; then
    HEPATTN_DIR="$WORK_ROOT/hepattn"
  else
    HEPATTN_DIR="$WORK_ROOT/hepattn-${HEPATTN_BRANCH//\//-}"
  fi
fi
export HEPATTN_BRANCH HEPATTN_DIR
