# PyTorch → Keras numerical parity report

This report is GENERATED from measurements recorded while running `tests/keras`
(see `tests/keras/gen_parity_report.py`); do not edit the tables by hand.

## Methodology

Every Keras component is weight-ported from its torch twin and compared elementwise
in fp32 against the torch reference (attention backend `torch`, i.e.
`scaled_dot_product_attention` semantics — the portable reference obtained via
`change_attn_backends(model, "torch")`). The acceptance criterion is
`|keras - torch| <= atol + rtol * |torch|` evaluated on valid (non-padded) slots;
each row below reports the measured maxima. Boolean outputs (thresholded predictions
and the per-decoder-layer mask-attention masks) are asserted to be EXACTLY equal —
for the self-referential mask-attention graph this is a stronger and more meaningful
criterion than any float tolerance.

## Known, deliberate numerical semantics

- **Bitwise equality across frameworks is not achievable**: a single keras `Dense`
  (matmul+add) vs torch `Linear` (fused addmm) already differs by up to ~3.6e-07 abs
  (measured in `tests/keras/test_env_smoke.py`). All further deviations compound from
  this kernel-level floor.
- **Fully-masked attention rows produce zeros**, matching the torch fused SDPA
  kernels the reference runs on. A textbook softmax would yield NaN there and poison
  valid outputs downstream through `0 * NaN` in the values contraction (padded
  decoder keys in the bidirectional cross-attention hit exactly this case).
- **Large max-relative errors on near-zero elements are expected** (the rel column is
  dominated by elements where the reference is ~0); the combined atol+rtol criterion
  is the pass/fail bar, and the abs column is the physically meaningful one there.
- **`mask_bce` can be `inf` on randomly-initialized models** (the mask task pads
  invalid-key logits to `finfo.min`, which saturates BCE): the torch reference itself
  is `inf` and the keras twin reproduces it exactly. Trained models do not sit in
  this regime.
- Regression outputs in `mode=scale` are exponentially scaled, so their max-abs error
  scales with the output magnitude; the relative error (~1e-5, i.e. fp32 precision)
  is the meaningful metric.

## Measured float parity (torch fp32 reference vs weight-ported float Keras twin)

| Component / tag | Configuration | Max abs err | Max rel err | atol | rtol |
|---|---|---|---|---|---|
| `attention.cross_masked` | `dim64h8+qkvnorm` | 2.38e-07 | 2.30e-04 | 1.00e-06 | 1.00e-05 |
| `attention.k_proj` | `dim64h8` | 0.00e+00 | 0.00e+00 | 1.00e-06 | 1.00e-05 |
| `attention.q_proj` | `dim64h8` | 0.00e+00 | 0.00e+00 | 1.00e-06 | 1.00e-05 |
| `attention.self_masked` | `dim64h8+qkvnorm` | 2.98e-07 | 4.65e-04 | 1.00e-06 | 1.00e-05 |
| `attention.v_proj` | `dim64h8` | 0.00e+00 | 0.00e+00 | 1.00e-06 | 1.00e-05 |
| `attention.value_residual_l1` | `dim64h8` | 1.49e-07 | 8.41e-04 | 1.00e-06 | 1.00e-05 |
| `attention.value_residual_l2` | `dim64h8` | 1.04e-07 | 3.70e-05 | 1.00e-06 | 1.00e-05 |
| `decoder_layer.kv` | `dim32h16_bidir` | 4.77e-07 | 7.55e-04 | 1.00e-06 | 1.00e-05 |
| `decoder_layer.q` | `dim32h16_bidir` | 7.15e-07 | 8.92e-04 | 1.00e-06 | 1.00e-05 |
| `dense` | `clic_classification_head` | 2.98e-08 | 1.24e-06 | 1.00e-06 | 1.00e-05 |
| `dense` | `clic_encoder_ffn_256` | 5.36e-07 | 3.84e-03 | 1.00e-06 | 1.00e-05 |
| `dense` | `clic_input_net_27_to_256` | 2.98e-07 | 9.81e-03 | 1.00e-06 | 1.00e-05 |
| `dense` | `clic_regression_head_518` | 2.98e-08 | 7.18e-07 | 1.00e-06 | 1.00e-05 |
| `dense` | `final_activation_sigmoid` | 5.96e-08 | 1.44e-07 | 1.00e-06 | 1.00e-05 |
| `dense` | `no_bias` | 0.00e+00 | 0.00e+00 | 1.00e-06 | 1.00e-05 |
| `dense` | `swiglu_ffn` | 2.38e-07 | 7.21e-03 | 1.00e-06 | 1.00e-05 |
| `dense` | `with_dropout_eval` | 2.98e-07 | 1.28e-03 | 1.00e-06 | 1.00e-05 |
| `encoder.clic_shaped` | `6L_dim256_h16_hybrid_vres_8reg` | 1.55e-06 | 2.07e-02 | 2.00e-06 | 2.00e-05 |
| `encoder.hybrid_norm` | `layer0` | 4.77e-07 | 4.46e-04 | 1.00e-06 | 1.00e-05 |
| `encoder.hybrid_norm` | `layer1` | 7.15e-07 | 6.74e-05 | 1.00e-06 | 1.00e-05 |
| `encoder.hybrid_norm` | `layer2` | 7.15e-07 | 5.01e-04 | 1.00e-06 | 1.00e-05 |
| `encoder.register_tokens` | `2L+4reg` | 4.77e-07 | 6.90e-05 | 1.00e-06 | 1.00e-05 |
| `maskformer.e2e` | `final.classification.pflow_class_prob` | 2.98e-08 | 2.07e-07 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `final.classification.pflow_logit` | 5.22e-08 | 8.63e-05 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `final.incidence.pflow_incidence` | 2.33e-09 | 3.50e-07 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `final.mask.pflow_node_logit` | 1.14e-05 | 3.36e-03 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `final.regression.pflow_proxy_regr` | 0.00e+00 | 0.00e+00 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `final.regression.pflow_regr` | 1.00e+00 | 1.35e-05 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `layer_0.mask.pflow_node_logit` | 8.11e-06 | 2.87e-03 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `layer_1.classification.pflow_class_prob` | 4.47e-08 | 2.40e-07 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `layer_1.classification.pflow_logit` | 3.73e-08 | 9.12e-05 | 5.00e-05 | 1.00e-04 |
| `maskformer.e2e` | `layer_1.mask.pflow_node_logit` | 1.14e-05 | 3.57e-02 | 5.00e-05 | 1.00e-04 |
| `maskformer.loss` | `final.classification.object_ce` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `final.incidence.kl_div` | 2.38e-07 | 9.55e-08 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `final.mask.mask_dice` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `final.regression.l1` | 1.91e-06 | 2.33e-07 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_0.mask.mask_dice` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_1.classification.object_ce` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_1.mask.mask_dice` | 5.96e-08 | 8.64e-08 | 1.00e-04 | 1.00e-03 |
| `quantized.delta.mask_l0` | `high_bitwidth_w20_d24_t18` | 3.01e-01 | — | 5.00e-01 | 0.00e+00 |
| `quantized.delta.mask_l0` | `low_bitwidth_w4_d4_t4` | 3.11e+01 | — | 5.00e-01 | 0.00e+00 |

## Quantized deltas (HGQ2 mode vs float twin)

The rows tagged `quantized.delta.*` above compare the SAME weights run through the
float twin and the HGQ2-quantized twin on layer-0 mask logits (computed before any
mask-attention feedback, so deltas are not amplified by threshold flips). The paired
criterion is what makes this non-vacuous: the high-bitwidth config must stay close to
float (fails if the config scopes silently did not apply) while the low-bitwidth
config must differ by more than 10x the high-bitwidth delta (fails if the quantizers
are inert under the torch backend).

Measured: high-bitwidth (20-bit weights, 24-bit datalane, 18-bit softmax tables)
deviates ~0.30 abs on logits of scale ~15 (~2%) — dominated by the quantized-softmax
exp/inv lookup tables accumulated across the attention pipeline; the `table`
quantizer place bounds softmax accuracy INDEPENDENTLY of weight/datalane bitwidths
(without configuring it the delta is ~2.7 regardless of the other bitwidths).
Low-bitwidth (4-bit) deviates ~31 — a factor ~100 separation.
