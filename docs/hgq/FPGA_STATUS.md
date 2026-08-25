# FPGA conversion status (hls4ml 1.3.0, HGQ2 0.1.9)

Measured on this repository's models and pinned by `tests/keras/test_export.py`
(the characterization tests fail if an upstream release changes any row — update
this document when they do).

## Support matrix

| Construct | Convert + HLS codegen | C-sim bit-exactness |
|---|---|---|
| `QDense` (linear / relu), rank-2 per-token | ✅ | ✅ verified (linux¹) |
| `QDense` pointwise, rank-3 `(150, dim)` (CLIC head shape) | ✅ | ✅ verified (linux¹) |
| Attention core: `QDense`×4 + `QEinsum`×2 + `QSoftmax`, fixed shapes | ✅ | ✅ verified (linux¹) |
| silu **inline** in `QDense(activation="silu")` | ❌ (converter TypeError) | — |
| silu as `QUnaryFunctionLUT(allow_heterogeneous_table=False)`, rank-2 | ✅ ² | codegen ✅ (add a csim gate next) |
| `QUnaryFunctionLUT` on rank-3 (pointwise) inputs | ❌ (HGQ2 table-shape bug) | — |

¹ `hls_model.compile()` (C simulation) cannot build on macOS: hls4ml's bundled
`ap_types` headers are incompatible with Apple's libc++ (`reference to 'complex' is
ambiguous`). The gates skip there; they PASS on linux (verified in GitHub CI and on
the falcon cluster, 2026-07-16): hls4ml C simulation of both the pointwise MLP and
the full attention core is EXACTLY equal to keras — zero mismatches over 256/64
random inputs.

² With a converter warning about `result_t` being propagated twice
("bit-exactness may be compromised") — benign for the gated constructs above per
the csim verdicts.

Conversion must run under `torch.no_grad()` on the torch backend (the converter
materializes LUT tables by evaluating activations and calling `.numpy()`);
`hepattn.keras.export.convert_to_hls` handles this.

## What this means for deploying the CLIC model

The training model uses inline silu activations (mirroring the torch model
one-to-one for parity). A deployment-oriented model/config should:

1. express nonlinear activations as explicit `QUnaryFunctionLUT` layers with
   homogeneous tables (a `LayerFactory` option — natural follow-up work), and
2. export per-query heads either at full pointwise rank with relu/linear, or
   per-token (rank-2) when LUT activations are involved.

## Out of FPGA scope (training-side machinery, by design)

- **Float LayerNorms** — including the hybrid-norm schedule's q/k/v norms inside
  every attention block. hls4ml has only experimental keras-v2 LayerNorm support and
  none in the HGQ2 converter path. Long-term options: DyT (tanh) norms, norm-free
  retraining, or an hls4ml extension.
- **Value-residual mixing** — couples every encoder layer to layer-0 values.
- **Per-layer mask-attention thresholding/scatter** — decoder control flow that
  re-derives attention masks from task outputs between layers.
- **Register-token concatenation and padding-mask plumbing.**
- **Hungarian matcher and all loss machinery** — training-only, never deployed.
- **`IncidenceBasedRegressionTask` proxy-feature construction** — float torch glue.

The deployable subgraphs demonstrated here — task-head MLPs and the norm-free
attention core — are the building blocks; assembling a full fixed-latency
reconstruction pipeline around them (and closing the norm gap) is the next research
step, tracked outside this repository's scope.
