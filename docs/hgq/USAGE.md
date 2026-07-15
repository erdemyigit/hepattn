# Usage: training and running the Keras/HGQ2 CLIC model

## Environment

The keras/HGQ2 stack lives in the `hgq` dependency group (keras ≥3.8, hgq2, hls4ml
≥1.2, da4ml). With uv (works on macOS and linux):

```bash
uv sync --group hgq
```

Notes:
- `KERAS_BACKEND` must be `torch` (or unset — `hepattn.keras` pins it on import and
  fails loudly on anything else). Always import keras via `from hepattn.keras import keras`.
- The keras torch backend auto-selects cuda/mps for new tensors independently of
  where your torch code runs; drivers pin it (`set_keras_default_device`). The
  Lightning integration handles this for you.
- pixi: the `hgq` group is not yet wired into a pixi environment — regenerating
  pixi.lock requires a linux host (cross-locking fails from macOS). To add it, on a
  linux machine append an `hgq = { features = ["cpu", "hgq"] }` environment in
  pyproject.toml and run `pixi lock`.
- da4ml has no linux-aarch64 wheels (marker-gated; only needed for FPGA conversion).

## Training (same CLI shape as the torch experiments)

```bash
# HGQ2 quantization-aware training
python -m hepattn.experiments.clic.main_hgq fit --config src/hepattn/experiments/clic/configs/pflow_hgq.yaml

# float keras reference (same graph, no quantizers)
python -m hepattn.experiments.clic.main_hgq fit --config src/hepattn/experiments/clic/configs/pflow_keras_float.yaml

# evaluation through the Lightning test stage (reloads the best checkpoint)
python -m hepattn.experiments.clic.main_hgq test --config <run_dir>/config.yaml
```

The configs read the exact same ROOT files as `base.yaml` (uproot, tree `EventTree`).
Constraints baked into the generated configs: `precision: 32` (bf16-autocast ×
quantizers is unvalidated), `devices: 1` (DDP over keras-torch modules not yet
validated), no `Compile` callback.

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
PYTHONPATH=src uv run pytest tests/keras tests/experiments/clic/test_clic_hgq.py
```

- Parity numbers are recorded to a ledger; regenerate the report with
  `python tests/keras/gen_parity_report.py` after a full run (see PARITY.md).
- The hls4ml C-simulation gates skip on macOS (toolchain); run them on linux.
- The torch.compile'd loss registries run eagerly inside these tests (an Inductor
  path bug with spaces in directory names; unrelated to the keras path).
