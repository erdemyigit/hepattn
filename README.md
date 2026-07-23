# hepattn

We present a general end-to-end ML approach for particle physics reconstruction by adapting cutting-edge object detection techniques.
Our work demonstrates that a single encoder-decoder transformer can solve many different reconstruction problems that traditionally required specialised, task-specific approaches.

Our method has been successfully applied to various reconstruction tasks and detector setups:

- **Pixel cluster splitting** - ATLAS [[PUB][tide]]
- **Hit filtering** - TrackML [[arXiv][trackml]], ITk [WIP]
- **Tracking** - TrackML [[arXiv][trackml]], ATLAS [[PUB][tide]]
- **Primary vertexing** - *Interested in working on this? Get in touch!*
- **Secondary vertexing** - Delphes [[EPJC][vertexing]]
- **Particle flow** - CLIC [[arXiv][glow]]
- **End-to-end reconstruction** - CLD [[ML4Jets][ml4jets]]
- **Muon Tracking** - ATLAS [[ConnectingTheDots][ctd]]

[tide]: https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2025-045/
[trackml]: https://arxiv.org/abs/2411.07149
[vertexing]: https://link.springer.com/article/10.1140/epjc/s10052-024-13374-5
[glow]: https://arxiv.org/abs/2508.20092
[ml4jets]: https://indico.cern.ch/event/1526677/contributions/6530938/
[ctd]: https://indico.cern.ch/event/1499357/contributions/6621917/

## ✨ Key Features

- **🏗️ Modular architecture**: Encoder, decoder, and task modules for flexible experimentation
- **⚡ Efficient attention**: Seamlessly switch between torch SDPA, FlashAttention, and FlexAttention
- **🔬 Cutting-edge transformers**: HybridNorm, LayerScale, value residuals, register tokens, local attention
- **🚀 Performance optimised**: Full `torch.compile` and nested tensor support
- **🧪 Thoroughly tested**: Comprehensive tests across multiple reconstruction tasks
- **📦 Easy deployment**: Packaged with Pixi for reproducible environments
- **🔩 FPGA-oriented quantization-aware training**: HGQ2/Keras-3 twin of the MaskFormer with hls4ml conversion — see [docs/hgq](docs/hgq/README.md)


## 🛠️ Setup

First clone the repository:

```shell
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn
```

We recommend using a container to set up and run the code.
This is necessary if your system's `libc` version is `<2.28`
due to requirements of recent `torch` versions.
We use `pixi`'s CUDA image, which you can access with:

```shell
apptainer pull pixi.sif docker://ghcr.io/prefix-dev/pixi:0.54.1-jammy-cuda-12.8.1
apptainer shell --nv pixi.sif
```

**📝 Note**: If you are not using the `pixi` container, you will need to make sure
`pixi` is installed according to https://pixi.sh/latest/installation/.

You can then install the project with locked dependencies:

```shell
pixi install --locked
```

**📝 Note**: The `default` environment targets GPU machines and installs FA2.
See the [pyproject.toml](pyproject.toml) or [setup/isambard.md](setup/isambard.md)
for more information.

## 🌟 Activating the Environment

To run the installed environment, use:

```shell
pixi shell
```

Multiple environments are configured in `pyproject.toml` for different hardware setups and experiments (`default`, `cpu`, `isambard`, `clic`, `tide`, `ci`). Use `-e <env>` to specify a specific environment.

You can close the environment with `exit`.
See the [`pixi shell` docs](https://pixi.sh/latest/reference/cli/pixi/shell/) for more information.

## 🧪 Running Tests

Once inside the environment, if a GPU and relevant external data are available, just run:

```shell
pytest
```

To test parts of the code that don't require a GPU, run:

```shell
pytest -m 'not gpu'
```

To test parts of the code that don't require external input data, run:

```shell
pytest -m 'not requiresdata'
```

The current CI only tests the parts of the code that don't require a GPU or external input data:

```shell
pytest -m 'not gpu and not requiresdata'
```

**📝 Note**: If you encounter import errors for missing modules like `numba` when running tests in the `default` environment, switch to the appropriate experiment environment or use the `ci` environment which includes all required dependencies for tests (e.g. `pixi run -e ci pytest -m 'not gpu and not requiresdata'`).


## 🏃 Run Experiments

See experiment directories for instructions on how to run experiments.

- [TrackML Tracking](src/hepattn/experiments/trackml/)
- [CLIC Particle Flow](src/hepattn/experiments/clic/)

## 🔩 Quantization-Aware Training with HGQ2

The `keras-hgq2` branch adds an FPGA-oriented quantization-aware training (QAT) path for the CLIC MaskFormer, built on [HGQ2](https://github.com/calad0i/HGQ2) (Keras 3) running on the **torch backend** — Keras-3 layers under this backend are `torch.nn.Module`s, so the quantized model plugs into the existing Lightning training loop, matcher, losses, and data pipeline unchanged. HGQ2 learns per-weight and per-activation bitwidths during training and reports EBOPs (effective bit-operations ≈ LUT + 55·DSP), a differentiable estimate of post-synthesis cost for [hls4ml](https://github.com/fastmachinelearning/hls4ml) deployment. Full documentation lives in [docs/hgq](docs/hgq/README.md): design rationale ([README](docs/hgq/README.md)), commands ([USAGE](docs/hgq/USAGE.md)), float-parity numbers ([PARITY](docs/hgq/PARITY.md)), and conversion status ([FPGA_STATUS](docs/hgq/FPGA_STATUS.md)).

### Setup

The stack lives in the `hgq` dependency group (`keras>=3.8`, `hgq2`, `hls4ml>=1.2`, `da4ml`):

```shell
uv sync --group hgq
```

**📝 Note**: `KERAS_BACKEND` must be `torch` or unset — `hepattn.keras` pins it on import and fails loudly on anything else. Always import keras via `from hepattn.keras import keras`, never `import keras` directly.

**📝 Note**: On Apple Silicon, add `--no-install-project` (the optional `lap1015` C++ matcher does not build with Apple clang; the default scipy solver is used instead).

### Implementation layout

- `src/hepattn/keras/` — the Keras/HGQ2 twins: `maskformer.py` (`KerasMaskFormer`), `encoder.py`, `decoder.py`, `attention.py`, `dense.py`, `factory.py` (the float/quant layer factory), `porting.py` (torch→keras weight ports), `callbacks.py` (`EBOPsMonitor`, `BetaScheduler`), `export.py` (hls4ml).
- Configs: [`pflow_hgq.yaml`](src/hepattn/experiments/clic/configs/pflow_hgq.yaml) (QAT) and [`pflow_keras_float.yaml`](src/hepattn/experiments/clic/configs/pflow_keras_float.yaml) (float reference). The model is selected by `class_path: hepattn.keras.maskformer.KerasMaskFormer`; the `quant:` dict switches modes — `quant: null` builds the float twin on the **identical graph** (same weight layout, so float weights warm-start QAT).
- Tests: `tests/keras/` and `tests/experiments/clic/test_clic_hgq.py` (`PYTHONPATH=src uv run pytest tests/keras tests/experiments/clic/test_clic_hgq.py`).

### Running

```shell
# QAT training (CLIC particle flow)
python -m hepattn.experiments.clic.main_hgq fit --config src/hepattn/experiments/clic/configs/pflow_hgq.yaml

# float keras reference (same graph, no quantizers)
python -m hepattn.experiments.clic.main_hgq fit --config src/hepattn/experiments/clic/configs/pflow_keras_float.yaml

# local development on Apple Silicon (MPS, batch sized for 16 GB)
./src/hepattn/experiments/clic/run_hgq_mac.sh fit

# consolidated evaluation: physics metrics + EBOPs + optional hls4ml resource report
python -m hepattn.experiments.clic.evaluate_hgq --config <run_dir>/config.yaml --state <ckpt> --num-events 2000 --out eval_report.md
```

**📝 Note**: Calibrate `beta` before long runs — a fresh model's EBOPs are enormous, so `beta * EBOPs` must be comparable to the task loss or it dominates (or numerically drowns) the optimization. Ramp it with the `BetaScheduler` callback and monitor `train/quant_ebops_loss` against `train/loss`. See [USAGE](docs/hgq/USAGE.md) for the full `quant:` reference.

### 🤖 Invariants and pitfalls (for humans and AI agents)

Hard constraints that are easy to violate and expensive to rediscover:

1. **Backend**: `KERAS_BACKEND=torch` always; `hepattn.keras` enforces this at import.
2. **Device pinning**: the keras torch backend picks cuda/mps for new tensors independently of your torch code — the Lightning module (`MPflowHGQ`) re-pins it on every fit/validate/test/predict start; standalone scripts must call `hepattn.keras.set_keras_default_device(...)` themselves.
3. **Lazy build**: quantized layers build at first forward (they need static shapes) — run one forward before porting weights, saving state dicts, or constructing optimizers manually.
4. **Deterministic names**: `KerasMaskFormer` calls `keras.utils.clear_session()` at init so quantizer layer names (which appear in `state_dict` keys) depend only on construction order — do not construct other keras layers before it in the same process.
5. **Precision**: `bf16-mixed` on CUDA is validated (loss finite, quantizer bitwidth gradients flow); HGQ2 quantizers keep fp32 internal state, so activation memory is higher than a float model — expect a smaller usable batch size (dim-256 model: micro-batch ~32 per 40 GB).
6. **On MPS, `TORCHDYNAMO_DISABLE=1` is required** (the run script sets it): the `torch.compile`-wrapped loss registries produce inf/NaN `mask_bce` through inductor-on-Metal; stock eager losses are clean.
7. **Do not add the `Compile` callback**: regional `torch.compile` of the quantized encoder/decoder is measured net-slower (keras-torch dispatch inserts graph breaks at every layer).
8. **Checkpointing for long runs**: `hepattn.callbacks.Checkpoint` monitors `val/loss`, which never fires mid-epoch — for walltime-limited clusters add a vanilla `ModelCheckpoint` with `every_n_train_steps` + `save_last: true` or mid-epoch progress is lost on resume.
9. **Evaluation data format**: point `test_path` at a train-format file (`val_clic_fix.root`); the `test_clic_common_infer.root` inference-format file lacks the targets the metrics harness needs.

## 📖 Terminology

To ensure clarity and consistency throughout this project, we use the following definitions:

- **constituent** - input entities that go into the encoder/decoder, e.g. inner detector hits
- **object** - reconstructed outputs from the decoder, e.g. reconstructed charged particle tracks
- **input** - (also `input_object`) generic term for any input to a module (could be constituents, objects, etc)
- **output** - generic term for any output from a module (could be objects, predictions, or intermediates)

## 🤝 Contributing

If you would like to contribute, please lint and format code with

```shell
ruff check --fix .
ruff format .
```

You can also set up pre-commit hooks to automatically run these checks before committing:

```shell
pre-commit install
```

## 📄 Citing

If you use this software in your research, please cite it using the citation information available in the GitHub repository sidebar (generated from [`CITATION.cff`](CITATION.cff)).
Please also cite [our papers](#hepattn) if they are relevant to your work.
