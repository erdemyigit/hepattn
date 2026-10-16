# Usage: training and running the Keras/HGQ2 CLIC model

## Environment

The keras/HGQ2 stack lives in the `hgq` dependency group (keras ≥3.8, hgq2, hls4ml
≥1.2, da4ml). There are two supported paths, and **they resolve `torch` differently**.
That difference is the most common setup trap, so choose deliberately.

### uv — recommended; macOS and linux, GPU works with no extra flags

```bash
uv sync --group hgq
```

`torch` is **not** a direct dependency of hepattn (it arrives transitively via
lightning / torchjd / lion-pytorch) and the `hgq` group does not pin it, so uv installs
the default PyPI wheel — which on linux-x86_64 bundles CUDA. Nothing further is needed
for GPU training. This is the path the falcon and Polaris runs actually used.

### pixi — linux only; the `hgq` environment is CPU-only BY DESIGN

pixi overrides torch per feature (`cpu` → `.../whl/cpu`, `gpu` → `.../whl/cu128`), and
`pyproject.toml` defines the environment as:

```toml
hgq = { features = ["cpu", "hgq"] }
```

So `pixi install -e hgq` yields a CPU-only torch, and `torch.cuda.is_available()` is
False **even on a GPU node**. That is intentional: this environment serves CI and
hls4ml C-simulation, neither of which needs CUDA.

For GPU training under pixi, add a *separate* environment rather than flipping this
one (the CPU build is still wanted):

```toml
hgq-gpu = { features = ["gpu", "hgq"] }
```

then regenerate the lock **on a linux host** (`pixi lock`) — cross-locking from macOS
fails, because the editable hepattn build and some group sdists cannot be built for
foreign platforms. `[tool.pixi.system-requirements] cuda` is checked at install time,
so install from a compute node, or export `CONDA_OVERRIDE_CUDA=12.8` on a login node
without a driver.

### ALCF Polaris

Do not set this up by hand — `polaris/01_setup_env.sh` is turnkey. It builds a uv venv
on a uv-managed CPython **3.12.0** (`requires-python = "== 3.12"` means *exactly*
3.12.0 under PEP 440, so a system 3.12.x is rejected), exports `PYTHONNOUSERSITE=1`,
skips flash-attn, and verifies the result. See `polaris/README.md`.

The branch is `HEPATTN_BRANCH` (`polaris/repo_dir.sh`). Unset, it is the branch of the checkout the script runs
from, and `keras-hgq2` when that is unknown; `HEPATTN_BRANCH=keras-jax bash polaris/01_setup_env.sh` overrides it.
`keras-hgq2` keeps its clone at `$WORK_ROOT/hepattn`; any other branch gets its own clone, `$WORK_ROOT/hepattn-<branch>`
(or `HEPATTN_DIR`), so jobs running from the `keras-hgq2` clone are never touched. The setup refuses to switch the
branch of an existing clone, and for branches other than `keras-hgq2` it installs with `uv sync --locked`
(masked-linformer comes from GitHub, so the login node needs outbound HTTPS). `qsub` passes no environment to a job:
every `.pbs` reads only `polaris/env.sh` and `repo_dir.sh` in the clone it is submitted from, so submit from the
clone the setup made. The setup checks that these files resolve to that clone and stops if they do not (for example a
`HEPATTN_DIR` given only on the command line; put it in `polaris/env.sh`). If you set `HEPATTN_BRANCH` in `env.sh`,
use the `${HEPATTN_BRANCH:-...}` form of `env.sh.example`, so that a value on the command line still wins.

Notes:
- `KERAS_BACKEND` must be `torch` (or unset — `hepattn.keras` pins it on import and
  fails loudly on anything else). Always import keras via `from hepattn.keras import keras`.
- The keras torch backend auto-selects cuda/mps for new tensors independently of
  where your torch code runs; drivers pin it (`set_keras_default_device`). The
  Lightning integration handles this for you.
- `torch.cuda.is_available()` is False on any **login node** regardless of how the
  environment was built — always confirm GPU visibility from a compute node.
- flash-attn is **not** required: the keras/HGQ2 path never imports it, and `flash` and
  `flash-varlen` are coerced to torch SDPA.
- da4ml has no linux-aarch64 wheels (marker-gated; only needed for FPGA conversion).

## Local training on Apple Silicon (M2/M3, MPS)

For development/small studies without the cluster, `run_hgq_mac.sh` wraps the
`pflow_hgq_local.yaml` config (MPS accelerator, batch 8 sized for 16 GB unified
memory, logger disabled, `num_train`-bounded load of the full ROOT file, same
calibrated β as production).

```bash
uv sync --group hgq --no-install-project   # --no-install-project: lap1015 C++ ext
                                           # won't build with Apple clang (scipy
                                           # matcher is the default; lap1015 optional)
# stage the CLIC files once:
scp falcon:~/data/clic/{train_clic_fix,val_clic_fix,test_clic_common_infer}.root data/clic/
./src/hepattn/experiments/clic/run_hgq_mac.sh fit
```

The runner sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (a few ops still route to CPU) and
`TORCHDYNAMO_DISABLE=1` (the compiled loss registries are unsupported on Metal).
Expect ~0.15–0.2 it/s — this path is for iteration and debugging, not production
throughput; use the cluster for full campaigns.

## Cluster training (same CLI shape as the torch experiments)

```bash
# HGQ2 quantization-aware training
python -m hepattn.experiments.clic.main_hgq fit --config src/hepattn/experiments/clic/configs/pflow_hgq.yaml

# float keras reference (same graph, no quantizers)
python -m hepattn.experiments.clic.main_hgq fit --config src/hepattn/experiments/clic/configs/pflow_keras_float.yaml

# evaluation through the Lightning test stage (reloads the best checkpoint)
python -m hepattn.experiments.clic.main_hgq test --config <run_dir>/config.yaml
```

The configs read the exact same ROOT files as `base.yaml` (uproot, tree `EventTree`).
Constraints baked into the generated configs: `devices: 4` with `strategy: ddp`
(validated 2026-09-07, run 7598515; `HGQ_DEVICES` overrides), no `Compile` callback
(measured net-slower on the quantized stack). `precision: bf16-mixed` is validated on CUDA (finite loss,
quantizer bitwidth gradients flow); the quantizers keep fp32 internal state, so
expect higher activation memory than the float twin (dim-256 model: micro-batch
~32 per 40 GB — use gradient accumulation for larger effective batches. Do not rescale
LR or grad-clip for it: Lightning averages accumulated micro-batch gradients).

## The Polaris config: the org/main Linformer model

`polaris/03_make_config.py` writes `polaris_hgq.yaml` from the org/main model (`ac0da8f`, docs/hgq/RETARGET_ORG_MAIN.md):
`configs/base.yaml` with `configs/linformer.yaml` merged over it. Compared with org, it changes only:
- `class_path` is `hepattn.keras.maskformer.KerasMaskFormer`, and the encoder is a plain dict (its `dim` is the
  model's), so every keras layer is built inside the HGQ2 config scopes;
- `quant:` (the HGQ2 spec) is added;
- the matcher solver is `scipy` instead of `lap1015_late`: the Polaris env is installed with `--no-install-project`,
  so `import lap1015` is an empty namespace package and the first match fails. scipy finds the same optimum; ties
  may break differently;
- trainer, data paths, batch, precision, optimizer and callbacks as listed in the script (`HGQ_*` variables).

So the model is: SwiGLU in the input net and every FFN, FastLayerNorm, `dense_norm_placement: legacy`,
`ca_sa_dense` with `self_attn_norm_kv`, constituents sorted by phi (`input_sort_field: phi`, padding last; the
encoder sorts and unsorts, the decoder query cross-attention gets the sort index), masked-Linformer attention with
k = 64 (seq_len 168 in the encoder, 160 in the decoder), and the PR #1/#3/a4263c1 tasks, matcher and losses.

`HGQ_ATTN=torch` writes `base.yaml` alone with dense attention and no sorting (the org clic-paper tag model).

Status (lanes R4a and R4b, port_log/R4a.md and R4b.md): the full org Linformer model matches org torch in float
(`tests/keras/test_org_model_parity.py`), and the default config runs a forward pass and the matched loss in float
and quantized mode (`tests/keras/test_polaris_config.py`). The attention core is the keras_native masked-Linformer
code (`hepattn.keras_native.masked_linformer`) run under the torch backend, gate H0 in
`tests/keras/test_linformer_parity.py`. Its cores build at the first forward pass in float mode too, as the quantized
layers already did; `MPflowHGQ.setup` runs that forward before the optimizer exists. Outside Lightning, call
`model.materialize(inputs)` (one eval-mode forward) before `state_dict()` or `load_state_dict()`: on a linformer model
that has not run a forward both raise, because the state would lack (or leave random) every proj_k/proj_v. Before a
QAT sweep, re-derive the EBOPs setpoints of `03_make_config.py` on this model (RETARGET_ORG_MAIN.md section 7, risk 1).

Quantized SwiGLU follows one leaf rule in the hybrid and in keras_native: the packed `Linear(in, 2h)` becomes two
dense halves, `<name>_hidden<i>_x1` (linear) and `<name>_hidden<i>_x2` (`activation="silu"` inside the layer), and
the product is an einsum `bnc,bnc->bnc` (`<name>_hidden<i>_glu`; `QEinsum` when quantized). Float mode uses the same
graph, so a SwiGLU net's float state_dict keys changed in R4a (no earlier run used SwiGLU).

## The `quant:` section

```yaml
model:
  model:
    class_path: hepattn.keras.maskformer.KerasMaskFormer
    init_args:
      ...
      quant:              # null => float reference model
        weight:           # QuantizerConfigScope(place="weight", ...)
          default_q_type: kbi   # NB: q_type/place are SELECTORS in HGQ2 scopes;
          b0: 8                 # the type is chosen via default_q_type
          i0: 2
        datalane:         # activations
          default_q_type: kif
          i0: 4
          f0: 8
        table:            # QSoftmax exp/inv LUT outputs — bounds softmax accuracy
          default_q_type: kif   # INDEPENDENTLY of weight/datalane bitwidths
          i0: 2
          f0: 10
        ebops:
          beta0: 1.0e-12
```

**Calibrate beta.** The EBOPs of a freshly initialized model are enormous (~1e17 for
the dim-32 test model); `beta * EBOPs` must be comparable to the task loss or it
either dominates the optimization (and can push all bitwidths to zero) or — numerically
worse — the summed loss is so large that task-loss progress vanishes below fp32
resolution. Monitor `train/quant_ebops_loss` against `train/loss` and use the
`BetaScheduler` callback (`beta_start/beta_end/warmup_steps`) to ramp the pressure in
after the task loss settles.

## Porting a trained torch checkpoint

```bash
python -m hepattn.experiments.clic.port_to_keras \
    --config <torch_run_dir>/config.yaml \
    --ckpt <torch_run_dir>/ckpts/epoch=...ckpt \
    --out keras_state.pt
```

The resulting state loads into a `KerasMaskFormer` built from the same config
(float, or quantized for a QAT warm start — kernels/biases share one layout; use
`hepattn.keras.porting.port_keras_to_keras` for float→quant keras-side ports).

Not yet for the org Linformer model (R4b open issue): `port_to_keras` fails at `change_attn_backends(model, "torch")`,
which org refuses for linformer, and once that is fixed it must call `keras_model.materialize(batch)` after
`port_maskformer` and before `state_dict()` (the save raises otherwise). `evaluate_hgq --float-state` must materialize
the float model before `load_state_dict` in the same way. `build_keras_maskformer_from_config` (used by `port_to_keras`,
`eval_keras` and `evaluate_hgq`) keeps `input_sort_field` of a torch config since the retarget critique
(`tests/keras/test_port_to_keras_config.py`); before, it built an unsorted hybrid for the org model.

## Evaluating a trained model

`evaluate_hgq.py` produces a consolidated report on a checkpoint: physics metrics on
the test split (each task's own `metrics()` — the functions the validation stage
logs), total + per-region EBOPs (the FPGA-cost side), and optionally an hls4ml
resource estimate for the deployable head.

```bash
python -m hepattn.experiments.clic.evaluate_hgq \
    --config <run_dir>/config.yaml \
    --state  <run_dir>/ckpts/epoch=NNN-....ckpt \
    --num-events 2000 --out eval_report.md \
    [--float-state <float_state.pt>]   # side-by-side quantized-vs-float columns
    [--hls]                            # + hls4ml conversion/resource report on the classifier head
```

Notes:
- Point `--config`'s `test_path` at a **train-format** file (`val_clic_fix.root` or
  `train_clic_fix.root`): it has the full targets the metrics need. The
  `test_clic_common_infer.root` file is the inference-format file for the offline
  PredictionWriter path and is not loadable by the metrics harness.
- Ratio metrics (efficiency/fake-rate/precision) are batch-averaged — treat as
  estimates that tighten with more events.
- Total EBOPs on a real dim-256 checkpoint ≈ 1.7e15, consistent with the standalone
  measurement. **The encoder/decoder split is strongly config-dependent** — measure it,
  don't quote it:

  | config | encoder | decoder |
  |---|---|---|
  | `pflow_hgq_local.yaml` (torch attention) | 8.94e14 (~47%) | 1.01e15 (~53%) |
  | `polaris_hgq.yaml` (linformer both stages) | 8.95e10 (~0.005%) | 1.68e15 (**>99.99%**) |

  Most of the reported total comes from `QSoftmax` layers, so anything that changes the
  attention implementation moves the split by orders of magnitude.
- `quant_losses()` is affine in beta — `floor + beta * E_eff`. Calibrate beta with
  `evaluate.effective_ebops` (or `polaris/09_measure_beta.py`). Since R4b, beta reaches
  every HGQ2 leaf, also those nested in QSoftmax and in the Linformer cores, and
  `E_eff = total_ebops + the task heads' EBOPs` (`quant_losses()` counts the heads twice;
  10.8% of the generated Linformer model at initial bitwidths). Before R4b beta reached
  only the top-level layers: on the old Polaris config `E_eff = 2.88e11` against a total of
  1.68e15, and four decades of beta moved the mean learned bitwidth by 0.01%.

## Standalone inference (no Lightning)

```bash
python -m hepattn.experiments.clic.eval_keras \
    --config <run_dir>/config.yaml \
    --state <keras_state.pt | lightning.ckpt> \
    --out metrics.json
```

Runs the test split eagerly (same ROOT files), reporting per-loss means and
events/s throughput.

## Tests

```bash
PYTHONPATH=src uv run pytest tests/keras tests/polaris tests/experiments/clic/test_clic_hgq.py
```

- The org-model gates are `test_org_model_parity.py` (full-model float parity and the state_dict key/shape map
  against org torch), `test_polaris_config.py` (the generated config), `test_sorting.py`, `test_swiglu_quantized.py`
  and `test_polaris_scripts.py`; gate H0 and the masked-Linformer attention are `test_linformer_parity.py`. The
  backend-consistency golden `tests/keras/golden/linformer_backend.npz` is written under JAX by
  `tests/keras/make_linformer_backend_golden.py` (run it in the native environment).

- Parity numbers are recorded to a ledger; regenerate the report with
  `python tests/keras/gen_parity_report.py` after a full run (see PARITY.md).
- The hls4ml C-simulation gates skip on macOS (toolchain); run them on linux.
- The torch.compile'd loss registries run eagerly inside these tests (an Inductor
  path bug with spaces in directory names; unrelated to the keras path).
