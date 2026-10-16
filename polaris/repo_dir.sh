# Sourced (after polaris/env.sh) by 01_setup_env.sh and every .pbs: the branch the work-space clone tracks, and the
# clone's directory. keras-hgq2 keeps the original $WORK_ROOT/hepattn; any other branch gets its own clone, so that
# setting up a new branch never checks out new code under jobs that run from the keras-hgq2 clone.
#   HEPATTN_BRANCH  branch of https://github.com/erdemyigit/hepattn to clone. Default: the branch of the clone this
#                   file is in. So a job runs the clone it was submitted from (qsub passes no environment and jobs
#                   read only polaris/env.sh), and setup run from a keras-jax checkout never pulls or rebuilds the
#                   keras-hgq2 clone. keras-hgq2 when that is unknown (not a git clone, or a detached HEAD).
#   HEPATTN_DIR     clone directory              (default $WORK_ROOT/hepattn, or $WORK_ROOT/hepattn-<branch, / -> ->)
if [ -z "${HEPATTN_BRANCH:-}" ]; then
  _repo_dir_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [ -e "$_repo_dir_root/.git" ]; then  # a directory in a clone, a file in a git worktree
    HEPATTN_BRANCH="$(git -C "$_repo_dir_root" symbolic-ref --short -q HEAD 2>/dev/null || true)"
  fi
  unset _repo_dir_root
fi
HEPATTN_BRANCH="${HEPATTN_BRANCH:-keras-hgq2}"
if [ -z "${HEPATTN_DIR:-}" ]; then
  if [ "$HEPATTN_BRANCH" = "keras-hgq2" ]; then
    HEPATTN_DIR="$WORK_ROOT/hepattn"
  else
    HEPATTN_DIR="$WORK_ROOT/hepattn-${HEPATTN_BRANCH//\//-}"
  fi
fi
export HEPATTN_BRANCH HEPATTN_DIR
