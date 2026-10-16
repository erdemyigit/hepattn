#!/usr/bin/env bash
# Build the HGQ2 environment on a Polaris LOGIN node (compute nodes have no outbound
# network, so all downloads must happen here).
#
#   bash polaris/01_setup_env.sh
#   HEPATTN_BRANCH=keras-jax bash polaris/01_setup_env.sh   # another branch, in its own clone
#
# The branch and the clone directory come from polaris/repo_dir.sh: HEPATTN_BRANCH (default keras-hgq2, in
# $WORK_ROOT/hepattn as before) and HEPATTN_DIR (default $WORK_ROOT/hepattn-<branch> for any other branch). Jobs
# read the same file, so set HEPATTN_BRANCH in env.sh (or export it) for both setup and submission.
#
# Design notes:
#  * Uses a plain venv, not conda/pixi. A venv disables user-site by default, which
#    structurally avoids the "~/.local/lib/python3.12/site-packages shadows the env's
#    torch -> CUDA version mismatch" trap. PYTHONNOUSERSITE=1 is exported anyway.
#  * uv manages its own CPython 3.12 (the repo pins requires-python == 3.12), so this
#    does not depend on which python the module system happens to provide.
#  * flash-attn is deliberately NOT installed: the Keras/HGQ2 path never imports it
#    (attn_type flash/flash-varlen are coerced to torch SDPA, and our config uses the
#    linformer implementation in this repo). That removes the hardest build step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] || { echo "ERROR: create $HERE/env.sh from env.sh.example first"; exit 1; }
source "$HERE/env.sh"
source "$HERE/repo_dir.sh"
export PYTHONNOUSERSITE=1

echo "==> work root: $WORK_ROOT"
mkdir -p "$WORK_ROOT"

# ---------------------------------------------------------------- uv
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv: $(uv --version)"

# ---------------------------------------------------------------- repo
REPO="$HEPATTN_DIR"
if [ -d "$REPO/.git" ]; then
  # Never switch the branch of an existing clone: jobs may be running from it.
  CURRENT="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  if [ "$CURRENT" != "$HEPATTN_BRANCH" ]; then
    echo "ERROR: $REPO is on branch $CURRENT, not $HEPATTN_BRANCH. Set HEPATTN_DIR to a new directory."
    exit 1
  fi
  echo "==> repo exists, updating (branch $HEPATTN_BRANCH)"
  git -C "$REPO" fetch -q origin && git -C "$REPO" pull -q --ff-only
else
  echo "==> cloning hepattn (branch $HEPATTN_BRANCH) into $REPO"
  git clone -q --branch "$HEPATTN_BRANCH" https://github.com/erdemyigit/hepattn.git "$REPO"
fi
cd "$REPO"
echo "==> at commit $(git log --oneline -1)"

# Carry env.sh into the work-space clone so there is exactly one configured copy and
# every later command can run from $REPO (where the 10 TB project space lives).
if [ "$HERE/env.sh" -ef "$REPO/polaris/env.sh" ]; then
  echo "==> env.sh already in the work-space clone"
else
  cp "$HERE/env.sh" "$REPO/polaris/env.sh"
  echo "==> copied env.sh into $REPO/polaris/"
fi

# ---------------------------------------------------------------- venv + deps
# --no-install-project: the optional lap1015 matcher extension is not needed
# (scipy is the default solver) and its build is fragile on HPC toolchains.
# pyproject pins requires-python = "== 3.12", which under PEP 440 means EXACTLY
# 3.12.0 — not 3.12.x. A system interpreter like 3.12.13 is rejected by uv. Have uv
# fetch a managed 3.12.0 so this does not depend on what the host happens to ship.
PYVER="3.12.0"
echo "==> ensuring CPython $PYVER is available (uv-managed)"
uv python install "$PYVER"
echo "==> creating venv (python $PYVER)"
rm -rf .venv
uv venv --python "$PYVER" .venv
echo "==> installing base + hgq dependency group (this takes a few minutes)"
# keras-hgq2 keeps its old behaviour. Other branches install exactly their uv.lock (--locked fails loudly if the
# lock is stale instead of re-solving on the login node); since the org/main merge that lock includes
# masked-linformer, which is fetched from GitHub, so the login node needs outbound HTTPS.
LOCKED=""
[ "$HEPATTN_BRANCH" = "keras-hgq2" ] || LOCKED="--locked"
# shellcheck disable=SC2086  # LOCKED is one flag or nothing
uv sync --group hgq --no-install-project $LOCKED --python .venv/bin/python

PY="$REPO/.venv/bin/python"

# ---------------------------------------------------------------- verify
echo
echo "==================== VERIFICATION ===================="
PYTHONNOUSERSITE=1 KERAS_BACKEND=torch PYTHONPATH="$REPO/src" "$PY" - <<'EOF'
import os, sys
print(f"python      {sys.version.split()[0]}  ({sys.executable})")
print(f"user-site   enabled={__import__('site').ENABLE_USER_SITE}  (must be False)")
import torch
print(f"torch       {torch.__version__}  built for CUDA {torch.version.cuda}")
print(f"cuda avail  {torch.cuda.is_available()}  (False on a login node is EXPECTED)")
import keras, hgq
print(f"keras       {keras.__version__}  backend={keras.backend.backend()}  (must be torch)")
from hgq.layers import QDense
print("hgq2        import OK")
import hepattn
from hepattn.keras.maskformer import KerasMaskFormer
print("hepattn     keras/HGQ2 path imports OK")
try:  # org/main attention (branches with the org/main merge) imports it
    import masked_linformer
    print(f"masked-lin  {masked_linformer.__file__}")
except ImportError:
    if os.environ["HEPATTN_BRANCH"] != "keras-hgq2":
        raise
assert keras.backend.backend() == "torch", "KERAS_BACKEND must be torch"
print("\nALL CHECKS PASSED")
EOF

cat <<EOF

==================== NEXT ====================
  RUN EVERYTHING FROM THE WORK-SPACE CLONE (project space, 10 TB):

      cd $REPO

  Data:   bash polaris/02_stage_data.sh    # SKIP if DATA_ROOT already has the files
  Smoke:  bash polaris/submit.sh 04_smoke.pbs     # ~10 min, gates everything
  Train:  bash polaris/submit.sh 05_sweep.pbs     # every GPU, one beta each

  Repo:   $REPO
  Venv:   $REPO/.venv
  Data:   $DATA_ROOT
EOF
