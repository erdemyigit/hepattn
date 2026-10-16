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
- **Padded mask logits are `finfo.min`** (as torch writes them). Since org `a4263c1`
  (bug 1, per-event mask losses) the padded-node terms are multiplied by the event's
  pad mask, so `mask_bce` is finite at random init and on dummy targets; before, it was
  `inf` in both frameworks.
- **Linformer rows are org/main's masked Linformer** (`masked_linformer` `dce2e15`,
  PR #9), whose core the hybrid runs from `hepattn.keras_native.masked_linformer`
  under the torch backend. Helen's Linformer layer is gone.
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
| `dense` | `clic_swiglu_ffn_256` | 3.87e-07 | 3.32e-03 | 1.00e-06 | 1.00e-05 |
| `dense` | `clic_swiglu_incidence_node_net` | 3.87e-07 | 3.52e-03 | 1.00e-06 | 1.00e-05 |
| `dense` | `clic_swiglu_input_net_27_to_256` | 2.09e-07 | 2.19e-03 | 1.00e-06 | 1.00e-05 |
| `dense` | `final_activation_sigmoid` | 5.96e-08 | 1.44e-07 | 1.00e-06 | 1.00e-05 |
| `dense` | `no_bias` | 0.00e+00 | 0.00e+00 | 1.00e-06 | 1.00e-05 |
| `dense` | `swiglu_ffn` | 2.38e-07 | 7.21e-03 | 1.00e-06 | 1.00e-05 |
| `dense` | `with_dropout_eval` | 2.98e-07 | 1.28e-03 | 1.00e-06 | 1.00e-05 |
| `encoder.clic_shaped` | `6L_dim256_h16_hybrid_vres_8reg` | 1.55e-06 | 2.07e-02 | 2.00e-06 | 2.00e-05 |
| `encoder.hybrid_norm` | `layer0` | 4.77e-07 | 4.46e-04 | 1.00e-06 | 1.00e-05 |
| `encoder.hybrid_norm` | `layer1` | 7.15e-07 | 6.74e-05 | 1.00e-06 | 1.00e-05 |
| `encoder.hybrid_norm` | `layer2` | 7.15e-07 | 5.01e-04 | 1.00e-06 | 1.00e-05 |
| `encoder.register_tokens` | `2L+4reg` | 4.77e-07 | 6.90e-05 | 1.00e-06 | 1.00e-05 |
| `h0.backend` | `high.enc.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.enc.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.enc_short.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.enc_short.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.kv_ca.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.kv_ca.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.q_ca.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.q_ca.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.q_sa.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `high.q_sa.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.enc.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.enc.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.enc_short.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.enc_short.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.kv_ca.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.kv_ca.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.q_ca.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.q_ca.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.q_sa.eval` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.backend` | `prod.q_sa.train` | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `h0.masked_linformer.attend` | `kv_ca` | 6.56e-07 | 1.55e-01 | 1.67e-06 | 0.00e+00 |
| `h0.masked_linformer.attend` | `q_ca` | 8.34e-07 | 3.87e-02 | 1.67e-06 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `kv_ca.E` | 1.91e-05 | 2.78e-03 | 8.49e-05 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `kv_ca.F` | 1.62e-05 | 6.52e-04 | 1.18e-04 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `kv_ca.k` | 3.87e-07 | 4.69e-03 | 6.56e-07 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `kv_ca.q` | 6.85e-07 | 3.66e-02 | 1.67e-06 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `kv_ca.v` | 3.58e-07 | 1.56e-02 | 7.15e-07 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `q_ca.E` | 1.53e-05 | 5.64e-03 | 8.49e-05 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `q_ca.F` | 1.72e-05 | 1.22e-03 | 1.18e-04 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `q_ca.k` | 3.28e-07 | 5.26e-03 | 6.56e-07 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `q_ca.q` | 7.15e-07 | 2.00e-02 | 1.67e-06 | 0.00e+00 |
| `h0.masked_linformer.attend_grad` | `q_ca.v` | 3.58e-07 | 1.14e-01 | 7.15e-07 | 0.00e+00 |
| `h0.masked_linformer.project` | `enc` | 1.01e-06 | 3.41e-02 | 2.03e-06 | 0.00e+00 |
| `h0.masked_linformer.project` | `enc_short` | 5.96e-07 | 7.66e-01 | 1.19e-06 | 0.00e+00 |
| `h0.masked_linformer.project` | `q_sa` | 0.00e+00 | 0.00e+00 | 5.96e-07 | 0.00e+00 |
| `hybrid.linformer` | `enc` | 6.26e-07 | 1.06e-01 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer` | `enc_short` | 4.47e-07 | 2.40e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer` | `kv_ca` | 6.56e-07 | 5.19e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer` | `q_ca` | 6.85e-07 | 1.55e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer` | `q_ca.sorted` | 5.36e-07 | 3.26e-03 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer` | `q_sa` | 6.26e-07 | 3.31e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_b1` | `enc.none` | 3.87e-07 | 1.79e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_b1` | `enc.ones` | 4.32e-07 | 4.15e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_b1` | `enc_short.none` | 3.87e-07 | 3.16e-04 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_b1` | `enc_short.ones` | 3.28e-07 | 1.16e-03 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_distinct_v` | `kv_ca` | 4.47e-07 | 5.21e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_distinct_v` | `q_ca` | 4.77e-07 | 9.75e-02 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_value_residual` | `layer0` | 4.47e-07 | 9.03e-03 | 1.37e-06 | 0.00e+00 |
| `hybrid.linformer_value_residual` | `layer1` | 4.47e-07 | 2.93e-04 | 1.37e-06 | 0.00e+00 |
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
| `maskformer.loss` | `final.incidence.kl_div` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `final.mask.mask_bce` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `final.mask.mask_dice` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `final.regression.l1` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_0.mask.mask_bce` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_0.mask.mask_dice` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_1.classification.object_ce` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_1.mask.mask_bce` | 9.54e-07 | 1.03e-07 | 1.00e-04 | 1.00e-03 |
| `maskformer.loss` | `layer_1.mask.mask_dice` | 0.00e+00 | 0.00e+00 | 1.00e-04 | 1.00e-03 |
| `org_model.dense.losses` | `seed0` | 5.28e-08 | — | 5.50e-07 | 5.50e-07 |
| `org_model.dense.losses` | `seed1` | 6.88e-08 | — | 5.50e-07 | 5.50e-07 |
| `org_model.dense.outputs` | `seed0` | 1.18e-05 | — | 3.70e-05 | 3.70e-05 |
| `org_model.dense.outputs` | `seed1` | 8.26e-06 | — | 3.70e-05 | 3.70e-05 |
| `org_model.linformer.losses` | `seed0` | 4.91e-08 | — | 3.75e-07 | 3.75e-07 |
| `org_model.linformer.losses` | `seed1` | 9.39e-08 | — | 3.75e-07 | 3.75e-07 |
| `org_model.linformer.outputs` | `seed0` | 4.49e-06 | — | 1.02e-05 | 1.02e-05 |
| `org_model.linformer.outputs` | `seed1` | 3.62e-06 | — | 1.02e-05 | 1.02e-05 |
| `quantized.delta.mask_l0` | `high_bitwidth_w20_d24_t18` | 3.01e-01 | — | 5.00e-01 | 0.00e+00 |
| `quantized.delta.mask_l0` | `low_bitwidth_w4_d4_t4` | 3.11e+01 | — | 5.00e-01 | 0.00e+00 |
| `swiglu.quantized.ffn_256` | `high_bitwidth` | 1.14e-02 | — | 2.60e-02 | 0.00e+00 |
| `swiglu.quantized.ffn_256` | `high_bitwidth_zero_bias` | 4.27e-05 | — | 8.60e-05 | 0.00e+00 |
| `swiglu.quantized.ffn_256` | `low_swiglu` | 2.74e-01 | — | 2.60e-02 | 0.00e+00 |
| `swiglu.quantized.incidence_node_net` | `high_bitwidth` | 1.16e-02 | — | 2.60e-02 | 0.00e+00 |
| `swiglu.quantized.incidence_node_net` | `high_bitwidth_zero_bias` | 3.58e-05 | — | 8.60e-05 | 0.00e+00 |
| `swiglu.quantized.incidence_node_net` | `low_swiglu` | 3.51e-01 | — | 2.60e-02 | 0.00e+00 |
| `swiglu.quantized.input_net_27_to_256` | `high_bitwidth` | 1.19e-02 | — | 2.60e-02 | 0.00e+00 |
| `swiglu.quantized.input_net_27_to_256` | `high_bitwidth_zero_bias` | 3.80e-05 | — | 8.60e-05 | 0.00e+00 |
| `swiglu.quantized.input_net_27_to_256` | `low_swiglu` | 7.32e-01 | — | 2.60e-02 | 0.00e+00 |

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
