#!/usr/bin/env bash
# Local HGQ2 QAT training on Apple Silicon (MPS). Run from the repo root.
# Data setup (once): copy the CLIC files into data/clic/ — from the CMU cluster:
#   scp falcon:~/data/clic/{train_clic_fix.root,val_clic_fix.root,test_clic_common_infer.root} data/clic/
set -euo pipefail

for f in data/clic/train_clic_fix.root data/clic/val_clic_fix.root; do
  [ -f "$f" ] || { echo "missing $f — see the data setup note at the top of this script"; exit 1; }
done

# Env setup (once): uv sync --group hgq --no-install-project
# (--no-install-project: the lap1015 C++ extension does not build with Apple clang;
#  it is an optional matcher backend and scipy is the default solver)
[ -d .venv ] || { echo "no .venv — run: uv sync --group hgq --no-install-project"; exit 1; }

# MPS fallback: a few ops still route to CPU on some torch versions; eager dynamo:
# the compiled loss registries are unsupported on this platform
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

exec uv run --no-sync python -m hepattn.experiments.clic.main_hgq "${1:-fit}" \
  --config src/hepattn/experiments/clic/configs/pflow_hgq_local.yaml "${@:2}"
