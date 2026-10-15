# hepattn.keras_native: setup and tests

`src/hepattn/keras_native/` is the pure `keras.ops` + HGQ2 port of the CLIC MaskFormer for training on
JAX (plan: [KERAS_JAX_PLAN.md](KERAS_JAX_PLAN.md); stage logs: [port_log/](port_log/README.md)).
No module in it imports torch. `hepattn/keras/` (the hybrid, torch backend) stays the parity oracle.

## Environment (macOS arm64, CPU)

`.venv-jax` (Python 3.12.0) at the worktree root runs the native code and, in the same process, the
pure-torch hepattn reference models for float parity. Before S1 it held the backend-spike set
(keras 3.15.0, hgq2 0.1.9, quantizers 1.2.2, numpy 2.3.5, jax/jaxlib 0.11.2, psutil, scipy 1.18.1, h5py;
from `uv pip list`). To recreate it (reconstructed command, CPU jax wheel; plan section 5.1 has the CUDA one):

```bash
uv venv --python 3.12.0 .venv-jax
uv pip install --python .venv-jax/bin/python "keras==3.15.0" "hgq2==0.1.9" "quantizers==1.2.2" "numpy==2.3.5" "jax==0.11.2" psutil
```

S1 added the torch reference stack and pytest (versions match the hybrid env `hepattn/.venv` where it has the package):

```bash
uv pip install --python .venv-jax/bin/python "torch==2.10.0" "lightning==2.5.2" "pytest==8.4.2" "pyyaml==6.0.3" \
  "uproot==5.7.1" "awkward==2.9.0" "atlas-ftag-tools==0.2.18" "einops==0.8.2" "lapx==0.9.4" \
  "numpy==2.3.5" "keras==3.15.0" "jax==0.11.2" "jaxlib==0.11.2" "hgq2==0.1.9" "quantizers==1.2.2"
uv pip install --python .venv-jax/bin/python "torchmetrics==1.8.2" "pytorch-lightning==2.6.1"
# S1 fix: the rest of the pyproject deps that hepattn.models.wrapper, hepattn.utils.cli / visualise_pes and
# hepattn.experiments.clic.lightning_module import (versions as in hepattn/.venv).
uv pip install --python .venv-jax/bin/python "lion-pytorch==0.2.3" "torchjd==0.7.0" \
  "jsonargparse[signatures]==4.40.1" "matplotlib==3.10.8" "torch==2.10.0" "numpy==2.3.5"
```

With these, every module of `hepattn.models` and `hepattn.utils`, plus `hepattn.experiments.clic.lightning_module`
and `pflow_data`, imports under `KERAS_BACKEND=jax` (41 modules, 0 failures, measured 2026-10-14).

The full resolved set is in [keras_native_env.txt](keras_native_env.txt) (`uv pip freeze`). The canonical way to
recreate the venv is `uv venv --python 3.12.0 .venv-jax && uv pip sync --python .venv-jax/bin/python
docs/hgq/keras_native_env.txt`. `uv.lock` and the pixi files are not changed. hepattn itself is **not** installed: always put the worktree's `src` first on `PYTHONPATH`.

## Running the tests

The keras backend is fixed per process, so the two suites run in separate pytest invocations.

```bash
# Native suite (JAX backend). conftest.py selects jax, and raises if KERAS_BACKEND is set to anything
# else or if hepattn resolves outside this worktree.
env -u KERAS_BACKEND PYTHONPATH=$PWD/src .venv-jax/bin/python -m pytest tests/keras_native -q

# Hybrid suite (torch backend, parity oracle), in the full hybrid env of the main checkout.
KERAS_BACKEND=torch PYTHONPATH=$PWD/src ../hepattn/.venv/bin/python -m pytest tests/keras -q
```

Lane worktrees use the same two interpreters by absolute path, with `PYTHONPATH=<lane>/src`.

A whole-repo run (`pytest`, or the CI job `pixi run -e ci test-unit-ci`) is one process, and `tests/keras`
puts keras on torch. `tests/conftest.py` therefore **ignores** `tests/keras_native` unless the process is on
the jax backend, and the terminal summary prints `tests/keras_native ignored: ...`. Naming `tests/keras_native`
(or a file in it) on the command line under another backend still raises: pytest loads that conftest at startup.
So the native suite never runs in the CI unit job; run it with the command above.

## Rules for code in keras_native

- No `import torch` in any module. `tests/keras_native/test_import_hygiene.py` (a) imports every module of the
  package in a fresh JAX-backend interpreter and fails if `torch` is in `sys.modules`, and (b) parses every
  `.py` file and fails on any import, at any depth (also inside functions, also `importlib.import_module`),
  of torch, torchmetrics, lightning, pytorch_lightning, lion_pytorch, torchjd, or a hepattn module outside
  `hepattn.keras_native`. New modules are covered automatically.
- Consequence: code that needs torch (the S7 torch-ckpt converter, an S6 adapter around the torch
  `DataLoader`) lives outside `keras_native`, e.g. in a sibling module or a script.
- `keras_native/__init__.py` imports no submodule (lanes add modules without editing it). It also must not set
  or check the backend: `factory.py` is re-exported to the torch hybrid through `hepattn/keras/factory.py`.
- Static shapes, deterministic layer names as in the hybrid, and unsupported configs raise.
