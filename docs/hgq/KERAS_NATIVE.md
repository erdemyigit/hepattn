# hepattn.keras_native: setup and tests

`src/hepattn/keras_native/` is the pure `keras.ops` + HGQ2 port of the CLIC MaskFormer for training on
JAX (plan: [KERAS_JAX_PLAN.md](KERAS_JAX_PLAN.md); stage logs: [port_log/](port_log/README.md)).
No module in it imports torch. `hepattn/keras/` (the hybrid, torch backend) stays the parity oracle.

## Modules (state after S6b)

| Stage | Modules | Tests (`tests/keras_native/`) |
|---|---|---|
| S1 | `factory.py` (moved from `hepattn/keras`, re-exported there), `norm.py`, `scaling.py` | `test_import_hygiene.py`, `test_helpers.py`, `test_collection_guard.py` |
| S2 | `dense.py`, `attention.py`, `linformer.py`, `encoder.py`, `input.py` | `test_float_parity.py`, `test_masked_query.py`, `test_quantized_golden.py`, `test_jit.py`, `test_variables.py`, `test_init.py` |
| S3 | `tasks.py` (4 task forwards), `decoder.py`, `maskformer.py` (`MaskFormer`, `from_config`) | `test_maskformer.py`, `test_mf_golden.py`, `test_mf_jit.py` |
| S4 | `losses.py`, `matching.py` (host scipy LSAP via `jax.pure_callback`), `objective.py` (`matched_loss`) | `test_objective_parity.py`, `test_matching.py`, `test_objective_e2e.py` |
| merge | | `test_lane_integration.py` (S3 model through the S4 loss, one jitted step) |
| S5 | `train_step.py` (`TrainStep`), `optim.py`, `ebops.py`, `beta_control.py` | `test_train_step.py`, `test_optim.py`, `test_ebops.py`, `test_beta_control.py` |
| S6a | `config.py`, `checkpoint.py`, `data.py`, `metrics.py`, `predict_writer.py` (+ `experiments/clic/native_data.py`, the torch data half) | `test_config.py`, `test_checkpoint.py`, `test_data.py`, `test_metrics.py`, `test_predict_writer.py`, `test_resume_cross_lane.py` |
| S6b | `loop.py` (`Trainer`), `cli.py` (+ `experiments/clic/main_jax.py`, the entry point) | `test_loop.py`, `test_cli.py` |

`matching.py` imports `jax` directly (`jax.pure_callback`, `jax.lax.stop_gradient`), so the matched loss
(`objective.match_layers` / `matched_loss`) runs only on the JAX backend. Every other module uses `keras.ops` only.
Not started: S7 converter/export, S8 Polaris validation (plan section 4).

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

The retarget to org/main `ac0da8f` (stage R0, `docs/hgq/RETARGET_ORG_MAIN.md` section 2) added masked-linformer
`dce2e15`, which org `src/hepattn/models/attention.py` imports. It is installed in `.venv-jax` and in the hybrid env
`hepattn/.venv`. `--no-deps` keeps torch 2.10.0 (masked-linformer only requires `torch>=2.0`):

```bash
for venv in .venv-jax ../hepattn/.venv; do
  uv pip install --python $venv/bin/python --no-deps \
    "masked-linformer @ https://github.com/compressed-maskformer-reco/masked-linformer/archive/dce2e151e9d25cefd85454b3dffd5c83ec217097.tar.gz"
done
```

The two local venvs were installed from the equivalent `git+https://github.com/compressed-maskformer-reco/masked-linformer@dce2e151…`
URL, so `keras_native_env.txt` records that form; the command above freezes as the archive URL. Both give the same
`masked_linformer.py` (`cmp` against the `dce2e15` checkout, R0 review).

On Polaris (one time; the login node needs outbound HTTPS to GitHub). `origin/keras-jax` is at `826bc0a` (after R5):
it has the org merge and masked-linformer in `uv.lock`, but not S5 / S6 (the driver and the `16_jax_smoke` /
`17_g3_step` scripts; nothing after `826bc0a` is pushed). The Polaris jobs of S6b need a pushed branch that holds them;
set `KJ_BRANCH` to it.

```bash
# Hybrid env: a separate clone, because running jobs use $WORK_ROOT/hepattn on keras-hgq2 (01_setup_env.sh:38-41).
# uv.lock pins masked-linformer; --locked fails loudly if uv.lock is stale.
KJ_BRANCH=${KJ_BRANCH:-keras-jax}
git clone --branch $KJ_BRANCH https://github.com/erdemyigit/hepattn.git $WORK_ROOT/hepattn-kj && cd $WORK_ROOT/hepattn-kj && \
  uv venv --python 3.12.0 .venv && uv sync --group hgq --no-install-project --locked --python .venv/bin/python
# .venv-jax: the KERAS_JAX_PLAN.md section 5.1 line, then masked-linformer (needed only for the torch parity tests):
uv pip install --python .venv-jax/bin/python --no-deps \
  "masked-linformer @ https://github.com/compressed-maskformer-reco/masked-linformer/archive/dce2e151e9d25cefd85454b3dffd5c83ec217097.tar.gz"
```

`--no-install-project` leaves `lap1015` unbuilt: `import lap1015` gives an empty namespace package, and the org
default solver `lap1015_late` fails at the first match (`AttributeError: ... lap_late`). Use `default_solver: scipy`
until lane R4 makes the generated Polaris config do so.

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

## Training (the S6b driver)

`python -m hepattn.experiments.clic.main_jax` runs one training on one device (JAX picks the GPU when the CUDA jax is
installed; there is no DDP). It reads the same YAML files as the hybrid (`config.load_config`: one file, several merged
as LightningCLI merges them, the generated `polaris_hgq.yaml`, or a saved run `config.yaml`) and runs the hybrid's fit
loop with the jitted steps (module docstring of `keras_native/loop.py` for the hook order and the logged names).

```bash
# Polaris: generate the config in fp32 on one device (the driver refuses bf16-mixed: fp32 first, user decision)
HGQ_PRECISION=32 HGQ_DEVICES=1 DATA_DIR=$DATA_ROOT .venv-jax/bin/python polaris/03_make_config.py
export JAX_COMPILATION_CACHE_DIR=$WORK_ROOT/jax_cache        # ~2 min per program on an A100 without it
PYTHONPATH=$PWD/src .venv-jax/bin/python -m hepattn.experiments.clic.main_jax fit \
  --config polaris_hgq.yaml --run-dir $WORK_ROOT/runs/r1
# resume the same run (a walltime chain): the newest checkpoint of the run directory
PYTHONPATH=$PWD/src .venv-jax/bin/python -m hepattn.experiments.clic.main_jax fit \
  --config polaris_hgq.yaml --run-dir $WORK_ROOT/runs/r1 --resume last
# prediction h5 (the PflowPredictionWriter layout) of a checkpoint
PYTHONPATH=$PWD/src .venv-jax/bin/python -m hepattn.experiments.clic.main_jax test \
  --config polaris_hgq.yaml --ckpt $WORK_ROOT/runs/r1/ckpts/epoch-004-step-....npz

# local check on dummy events at test sizes (CPU, ~1 min)
env -u KERAS_BACKEND PYTHONPATH=$PWD/src .venv-jax/bin/python -m hepattn.experiments.clic.main_jax fit \
  --config tests/keras_native/s6b_fixtures/tiny_polaris.yaml --run-dir /tmp/r_tiny --dummy-data
```

`--help` lists the overrides (`--max-epochs`, `--max-steps`, `--limit-train-batches`, `--limit-val-batches`,
`--batch-size`, `--num-workers`, `--num-train`, `--num-val`, `--log-every-n-steps`, `--ckpt-every-n-steps`,
`--ckpt-save-top-k`, `--seed`, `--matmul-precision`, `--no-donate`, `--compile-cache`, `--metrics-per-layer`).
A run directory holds: `config.yaml` (the merged config with the command-line data, trainer, seed and step-checkpoint
options written in: `fit --config <run-dir>/config.yaml` repeats the run; `--matmul-precision`, `--no-donate` and
`--metrics-per-layer` are only in `run.json`), `run.json` (options, versions, model hash, `total_steps`),
`ckpts/` (`step-*.npz` every `every_n_train_steps` of the YAML's `ModelCheckpoint`, `epoch-*-step-*.npz` after each
validation as hepattn's `Checkpoint`, pointers `last.json` / `best.json` on `val/loss`), `steps.csv` (one row per logged
optimizer step), `metrics.csv` (one row per validation epoch, the hybrid `MetricsCsvWriter` layout; a row with new
keys adds columns) and `summary.json` (counters, compile counts, persistent-cache hits of the latest process). Use one
run directory per run and its resumes; the YAML's `CKPT_DIR` and `MetricsCsvWriter` paths are not used. A resume in
the same directory writes `config-r<n>.yaml` / `run-r<n>.json` (the first run's files stay), drops the CSV rows written
after its checkpoint and removes checkpoint temporaries of a killed write. A resume needs the same config, epoch
length, total steps (the LR schedule) and seed (the data order): the driver refuses another. On CPU a resume continues
bit for bit. `--resume last` needs a step checkpoint callback (the YAML's `ModelCheckpoint` or `--ckpt-every-n-steps`):
hepattn's epoch `Checkpoint` writes no `last`, as in the hybrid; pass an epoch checkpoint path instead.

Polaris checks of the driver (one A100 each, 1 h debug queue):

```bash
bash polaris/submit.sh 16_jax_smoke.pbs                                     # real-data smoke + resume
bash polaris/submit.sh 17_g3_step.pbs                                       # gate G3: jax-prod, hybrid, jax-match
G3_JOBS=jax-g1:32,jax-precision:32,hybrid-tf32:16 bash polaris/submit.sh 17_g3_step.pbs   # G1 gap, TF32 (2nd job)
```

Both G3 arms run full fp32 matmuls: jax-prod at "highest", the hybrid with `--matmul_precision highest` (the hybrid
CLI's default "high" is TF32 on an A100; `hybrid-tf32` measures that production setting next to it). `qsub` passes no
environment to a job, so `submit.sh` writes the set knobs (`G3_JOBS`, `G3_ARGS`, `G3_DATA_FILE`, `SMOKE_ARGS`,
`SMOKE_XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR`) to `logs/<script>.<time>.<pid>.env` and gives the job only that path
(`-v SUBMIT_ENV_FILE=...`); a bare `qsub` runs the defaults.

The DataLoader workers (`data.num_workers`, 8 in the Polaris config) are spawned; keep `main_jax` as the entry point
(its `__main__` guard is what a spawned worker re-imports). What differs from the Lightning run is listed in
[port_log/S6b.md](port_log/S6b.md) (deviations).

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
