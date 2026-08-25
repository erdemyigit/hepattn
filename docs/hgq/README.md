# hepattn × HGQ2: quantization-aware MaskFormer training through Keras 3

This package (`hepattn.keras`) lets the hepattn MaskFormer train quantization-aware
with [HGQ2](https://github.com/calad0i/HGQ2) (High-Granularity Quantization v2) and
convert its deployable subgraphs to FPGA firmware with
[hls4ml](https://github.com/fastmachinelearning/hls4ml) — while reusing the existing
PyTorch-Lightning training driver, YAML configs, ROOT data pipeline, losses, and
Hungarian matcher unchanged.

- **Usage / commands:** [USAGE.md](USAGE.md)
- **Measured numerical parity:** [PARITY.md](PARITY.md) (generated from the test ledger)
- **FPGA conversion status:** [FPGA_STATUS.md](FPGA_STATUS.md)

## The central design decision

HGQ2 requires a Keras 3 frontend; hepattn is PyTorch Lightning. Keras 3 is
multi-backend, and under its **torch backend every keras layer is a genuine
`torch.nn.Module`**: its weights are torch parameters, its outputs are torch tensors,
and autograd flows through it. That single fact dissolves the frontend conflict —
instead of porting the training framework, only the *network graph* is rebuilt from
keras/HGQ2 layers, and it slots into the existing Lightning driver:

| Reused unchanged (torch) | Rebuilt as keras/HGQ2 layers |
|---|---|
| `MaskFormer` forward/loss/predict orchestration | `Dense` MLPs → `KerasDense` (QDense stacks) |
| Decoder mask-attention (threshold/scatter/detach) | `Attention` → `KerasAttention` (QDense projections + QEinsum/QSoftmax core) |
| Norms, `Residual`, hybrid-norm schedule (float boundary) | mask-logit / incidence einsums + incidence softmax (quantized forwards) |
| Hungarian matcher, `loss.py` registries | |
| `ModelWrapper`/CLI/Comet, `CLICDataset`, collation | |
| Task loss/cost/predict/attn_mask logic | |

`KerasMaskFormer` subclasses `MaskFormer`; `KerasMaskFormerDecoder` subclasses the
torch decoder and replaces only layer construction, so the delicate mask-attention
semantics are inherited, never re-derived. Task modules and input nets are the very
objects the YAML config describes, with their `Dense` sub-nets swapped in place
(`kerasify_module`).

## Why the ROOT files need no conversion

The target experiment (CLIC pflow) reads its ROOT files directly at train time with
uproot (`CLICDataset`, tree `EventTree`). Because the torch backend keeps the whole
torch `DataLoader` pipeline intact, the keras path consumes the exact same dataset
objects the YAML config names — no format conversion, no second data path, no
re-validation of a new reader. This satisfies the read-the-same-ROOT-files
requirement *by construction* rather than by writing new I/O code.

## Float / quantized dual mode

Every parameterized compute layer is created through a `LayerFactory`
(`hepattn.keras.factory`): with `quant: null` it emits plain keras layers — the
**float parity reference** that is validated elementwise against the torch model
(see PARITY.md) — and with a quant spec it emits HGQ2 Q-layers on the *identical
graph with the identical weight layout*, so float-trained weights warm-start QAT
(`port_keras_to_keras`). HGQ2 configuration scopes are applied inside the model
constructor, making YAML instantiation order irrelevant.

The EBOPs (effective bit operations, ≈ FPGA resource usage) regularization that
drives HGQ2's gradient-based bitwidth optimization is collected from the keras
layers' `add_loss` terms and added to the Lightning loss (`MPflowHGQ`); `beta` is
ramped by a scheduler callback and must be calibrated per model (see USAGE.md).

## Known boundaries (deliberate, documented)

- **Norms stay float.** HGQ2 has no quantized LayerNorm; the CLIC model's hybrid-norm
  schedule interleaves LayerNorms everywhere (including q/k/v norms inside attention).
  The torch norm modules are reused as-is — a conversion boundary for hls4ml, not a
  training limitation.
- **torch-SDPA attention semantics only.** flash/flex/varlen backends, sliding
  windows, score mods and input sorting are torch-side performance features with
  identical math on the dense-mask path; `KerasEncoder` coerces `flash-varlen`
  configs to `torch` with a warning.
- **Fully-masked attention rows produce zeros**, matching the fused SDPA kernels the
  torch reference runs on (a textbook softmax would produce NaN and poison valid
  outputs through `0 × NaN`).
- **Quantized layers build lazily** at the first forward: HGQ2 sizes per-element
  bitwidth variables and the EBOPs parallelism from the true static shapes, which the
  padded CLIC pipeline provides (fixed `max_nodes`/`num_queries`). The Lightning
  driver runs one forward in `setup()` before creating the optimizer.
- **The matcher is training-only** (host-side scipy on CPU) and is not part of any
  deployed graph.

## Package layout

```
src/hepattn/keras/
  __init__.py     backend guard (torch only) + device helpers
  factory.py      QuantSpec + LayerFactory (float / HGQ2 leaves), softmax dispatch
  dense.py        KerasDense (+ torch<->keras Dense porting)
  attention.py    KerasAttention (composed: projections + einsum/softmax core)
  encoder.py      KerasEncoder(Layer) — dense-mask path
  decoder.py      KerasMaskFormerDecoder(Layer) — construction only, forward inherited
  tasks.py        kerasify_module net-swap + quantized task forwards
  maskformer.py   KerasMaskFormer + EBOPs collection
  porting.py      torch->keras and keras->keras weight porting
  callbacks.py    EBOPsMonitor, BetaScheduler (Lightning)
  export.py       functional assembly + hls4ml conversion
src/hepattn/experiments/clic/
  lightning_module_hgq.py  MPflowHGQ
  main_hgq.py              CLI entrypoint
  port_to_keras.py         checkpoint porting CLI
  eval_keras.py            standalone inference/eval
  configs/pflow_hgq.yaml, pflow_keras_float.yaml
```
