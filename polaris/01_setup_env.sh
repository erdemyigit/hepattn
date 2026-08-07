#!/usr/bin/env bash
# Build the HGQ2 environment on a Polaris LOGIN node (compute nodes have no outbound
# network, so all downloads must happen here).
#
#   bash polaris/01_setup_env.sh
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
REPO="$WORK_ROOT/hepattn"
if [ -d "$REPO/.git" ]; then
  echo "==> repo exists, updating"
  git -C "$REPO" fetch -q origin && git -C "$REPO" checkout -q keras-hgq2 && git -C "$REPO" pull -q --ff-only
else
  echo "==> cloning hepattn (branch keras-hgq2)"
  git clone -q --branch keras-hgq2 https://github.com/erdemyigit/hepattn.git "$REPO"
fi
cd "$REPO"
echo "==> at commit $(git log --oneline -1)"

# ---------------------------------------------------------------- venv + deps
# --no-install-project: the optional lap1015 matcher extension is not needed
# (scipy is the default solver) and its build is fragile on HPC toolchains.
echo "==> creating venv (python 3.12)"
uv venv --python 3.12 .venv
echo "==> installing base + hgq dependency group (this takes a few minutes)"
uv sync --group hgq --no-install-project

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
assert keras.backend.backend() == "torch", "KERAS_BACKEND must be torch"
print("\nALL CHECKS PASSED")
EOF

cat <<EOF

==================== NEXT ====================
  Data:   bash polaris/02_stage_data.sh
  Smoke:  qsub polaris/04_smoke.pbs        # ~10 min on 1 GPU, gates everything
  Train:  qsub polaris/05_sweep.pbs        # 4 GPUs, 4 beta values in parallel

  Repo:   $REPO
  Venv:   $REPO/.venv
EOF
