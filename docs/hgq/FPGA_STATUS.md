# FPGA conversion status (hls4ml 1.3.0, HGQ2 0.1.9)

Measured on this repository's models and pinned by `tests/keras/test_export.py` (hybrid, torch backend) and
`tests/keras_native/test_export.py` (native, JAX backend). The characterization tests fail if an upstream release
changes a row: update this document when they do.

## Support matrix

| Construct | Convert + HLS codegen | C-sim bit-exactness |
|---|---|---|
| `QDense` (linear / relu), rank-2 per-token | ✅ | ✅ verified (linux¹) |
| `QDense` pointwise, rank-3 `(150, dim)` (CLIC head shape) | ✅ | ✅ verified (linux¹) |
| Attention core: `QDense`×4 + `QEinsum`×2 + `QSoftmax`, fixed shapes | ✅ | ✅ verified (linux¹) |
| silu **inline** in `QDense(activation="silu")` | ❌ rank 3: `TypeError` in hls4ml `bit_exact.py:602` (`np.ones(k)` on an array); rank 2 followed by a layer: shape `AssertionError` (same cause); rank 2 as the last layer: writes a call to `nnet::silu`, which no hls4ml 1.3.0 template defines³ | — |
| sigmoid **inline** in `QDense` (the encoder value-residual mix) | ✅ | ❌ hls4ml's own sigmoid table: all 2304 outputs differ, max 9.8e-4³ |
| silu / sigmoid as `QDense` (linear) + rank-2 `QUnaryFunctionLUT` (the S7 `activation_export_form`) | ✅ ² | ✅ verified (linux³); approximate to the trained layer: within half a lane step while the table input does not saturate (\|pre-activation\| < 2⁴ - 2⁻⁶ for the (4, 6) lane), unbounded past it (measure the range on real events before deployment) |
| `QUnaryFunctionLUT` on rank-3 (pointwise) inputs | ❌ (HGQ2 table-shape bug) | — |
| renorm `kept`: `QEinsum` of a mask with a computed batch-1 (1, n, k) table | ❌ "Batch dim mismatch for the inputs and output" | — |
| renorm `kept` as `QEinsumDense` with the frozen table as kernel (S7 `kept_export_form`, equal to the leaf) | ✅ | ✅ verified (linux³) |
| every other HGQ2 leaf kind of the org model (`QEinsumDense` E/F projections, the 8 `QEinsum` equations, `QSoftmax`, rsqrt `QUnaryFunctionLUT`) | ✅ | ✅ verified (linux³)⁴, rsqrt on its domain with one characterized point⁵ |

¹ `hls_model.compile()` (C simulation) cannot build on macOS: hls4ml's bundled
`ap_types` headers are incompatible with Apple's libc++ (`reference to 'complex' is
ambiguous`). The gates skip there; they PASS on linux (verified in GitHub CI and on
the falcon cluster, 2026-07-16): hls4ml C simulation of both the pointwise MLP and
the full attention core is EXACTLY equal to keras — zero mismatches over 256/64
random inputs.

² With a converter warning about `result_t` being propagated twice
("bit-exactness may be compromised") — benign for the gated constructs above per
the csim verdicts.

³ Native model under JAX (S7, `docs/hgq/port_log/S7_csim.py`), on falcon (Ubuntu 24.04, g++ 13.3.0), 2026-10-20: one
model per HGQ2 leaf kind of the org QAT model converted from a hybrid Lightning checkpoint (19 kinds, the kept and
silu / sigmoid kinds in export form), the toy MLP and attention core, and the classification head in its LUT form:
22 of 22 bit-exact (16 random events each); control rows: the inline sigmoid as trained differs, and one input
perturbed by 3 LSB changes 6 outputs. Keras on Linux equals keras on macOS on every row
(`docs/hgq/port_log/S7_evidence/csim.json`). Inputs are in each leaf's domain (0/1 masks; the rsqrt table's input is
the kept energy fraction in [0, 1)); rerun after review round 1 without a training call before the export (the
converted checkpoint's state): again 22 of 22, `S7_evidence/csim_r1.json`.

⁵ The rsqrt `QUnaryFunctionLUT` (renorm, input lane unsigned (1, 18), output lane (10, 14)) is bit-exact on its domain
except at the floor, input 0 (an input below 0 saturates to 0 too, but the kept energy fraction is never negative):
keras gives 2¹⁰, one LSB past the output lane, hls4ml the lane maximum 2¹⁰ - 2⁻¹⁴. Cause: HGQ2's SAT quantizer clips
to 2¹⁰ - 2⁻¹⁴ and then rounds `floor(v * 2¹⁴ + 0.5)` in float32, where (2²⁴ - 1) + 0.5 rounds to 2²⁴. The floor is
reached on a mask row that selects no key; there the table output multiplies exact zeros (attend) or is selected out,
so it does not reach the core output (`tests/keras_native/test_export.py::test_rsqrt_floor_does_not_reach_the_core_outputs`:
any floor value gives bitwise the same outputs). Measured on falcon: 16 floor points, all as characterized, the other
112 outputs exact. Fed standard-normal inputs (half of them negative) the leaf differs in 268 of 512 outputs (review
round 1); on a grid over [-1, 2.5] every difference is at an input in [-1, 0].

⁴ A leaf without an input quantizer of its own (`QSoftmax`) is gated behind a fixed (8, 12) input quantizer: in the
full graph the previous layer's result type sets its input precision; fed a float input directly, hls4ml reads it at
its default precision and C simulation differs (measured: 16856 of 18432 outputs, max 2.9e-3, before the fix).

Under the torch backend the conversion must run under `torch.no_grad()` (the converter materializes LUT tables by
evaluating activations and calling `.numpy()`); `hepattn.keras.export.convert_to_hls` handles this. Under JAX no
wrapper is needed (`hepattn.keras_native.export.convert_to_hls`).

## What this means for deploying the CLIC model

The training model uses inline silu activations (mirroring the torch model
one-to-one for parity). A deployment-oriented model/config should:

1. express nonlinear activations as explicit `QUnaryFunctionLUT` layers with
   homogeneous tables (a `LayerFactory` option — natural follow-up work; the S7
   `activation_export_form` does this after training, which is approximate), and
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
