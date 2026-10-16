# Retarget completeness critique (R0-R5)

Plan: [RETARGET_ORG_MAIN.md](../RETARGET_ORG_MAIN.md). Reviewed `keras-jax` = `rt/integration` = `2435fa8` on 2026-10-16,
macOS arm64, CPU only (torch 2.10.0, keras 3.15.0, HGQ2 0.1.9, JAX 0.11.2, masked-linformer `dce2e15` in both venvs:
the same md5 as the pinned checkout). Probes, runners, mutant scripts and outputs: [retarget/critique/](retarget/critique/)
(scratch: `$S/crit/`). Nothing was pushed; the `hepattn` main working tree was not touched.

## Checks

| # | Check | Result |
|---|---|---|
| 1 | Native `from_config` (raw `base.yaml` + `linformer.yaml` merge) against org torch built by **jsonargparse** from the two files (LightningCLI's merge), **production dims** (dim 256, 16 heads, 6 + 4 layers, k 64), 3 CLIC dummy events (55/78/100 of 160 nodes valid), seeds 7 and 11, biases perturbed, weights ported | Every output: max 6.7e-6 / 7.4e-6 (mask logits, scale 5); `proxy_regr` 3.05e-5 at scale 132. Padding sentinels equal. 4/4 attention masks exactly equal. 17 matched losses (`specs_from_config` of the same YAML, scipy): max 9.5e-7. Control (native without sorting): 2.29, 77-253 mask flips, losses 4.3e-2 (`native_e2e_*.out`) |
| 2 | Hybrid `KerasMaskFormer` (float) on the same config against the same torch model | Max 7.2e-6; masks equal; losses 9.5e-7; the attention runs `AttendCore` / `ProjectCore`. Control without sorting: 2.29, 77-253 flips (`hybrid_e2e_7_*.out`) |
| 3 | Bugs 1 and 2 reverted in native (scratch copies; 67 targets: `test_org_loss_fixes`, `test_objective_parity`, `test_objective_e2e`, `test_org_model`) | Unmutated 67/67 pass. Old cross-event `mask_bce` + `mask_dice`: 18 fail; old `mask_dice` only: 15 fail; class cost on `_class_prob`: 10 fail (B2-a, B2-b, B2-c, the 3 cost rows, the 3 e2e rows, `test_org_model` losses). B1-c (single event, old = new) passes under the mutants, as designed |
| 4 | Every section 5 gate has a passing test (map below); discrimination spot-proved by 12 mutants across R1-R4 (below) plus the R5 mutant set (`retarget/r5/mut`, 7/7 killed) | pass |
| 5 | No xfails; import hygiene; Helen's Linformer gone | Native 543 passed, 0 skipped, 0 xfailed. Hybrid + Polaris 283 passed, 8 skipped (2 hls4ml C-sim, 6 CUDA). Torch 522 passed, 173 skipped. Matching: `ran=172 failures=106 lap1015_attr=106 other=0`, gate exit 0; its 2 xfails are org/main's own (`tests/{matching,models,loss}` equal org/main). Importing all 14 `keras_native` modules under JAX loads 0 torch / lightning / masked_linformer modules. The 3 Helen files are gone; `git grep` finds only comments, the `linformer_proj_dim` refusals and their tests. `uv lock --check` exits 0 |
| 6 | `polaris/03_make_config.py` output, `15_g2_forward.py` | The generated config, built by jsonargparse as the hybrid, equals org torch in float (7.2e-6, masks equal); quantized train-mode forward + loss finite (17 losses). Native `from_config` of it: float 3.05e-5 (scale 132), quantized matched total finite. G2 sweep at tiny dims (JAX b2/b4, hybrid b2) exits 0: phi sort, 416/416 nonzero gradients and penalty 0.1312 in JAX and the hybrid surrogate (`gen_*.out`) |
| 7 | Docs against code | `KERAS_JAX_PLAN.md`, `PARITY.md` (org-model rows present, mask_bce note replaced), `port_log/README.md`, `USAGE.md`, `KERAS_NATIVE.md` match. Stale: the status line of `RETARGET_ORG_MAIN.md` ("Nothing here is implemented yet"): fixed |

Section 5 gate map (all pass in the native run above): 5.0 JIT gate `test_jit`, `test_mf_jit`, the `*_trains_under_jit*`
rows; 5.1 `test_from_config_builds_org_linformer_model`, `test_linformer_projection_shapes_equal_torch`,
`test_variables`, `test_build_attention_rejects_unknown_or_missing_options`; 5.2 `test_fast_layer_norm_*`,
`test_encoder_legacy_placement_*`, `test_placement_mismatch_*`; 5.3 `test_decoder_layer_pr8_options_equal_torch` (controls at
depth 0, no-effect at depth 1), `test_kv_sort_idx_reaches_q_ca_only`; 5.4 `test_swiglu_*` (swap-halves mutant inside);
5.5 `test_sort_index_*`, `test_unsort_round_trip_*`, `test_encoder_sorting_*`; 5.6 `test_masked_linformer.py` (22 tests:
float, exactness, tail, init, renorm bound, convergence, leaf inventory, paired delta, JIT) and hybrid H0
`tests/keras/test_linformer_parity.py`; 5.7 `test_matching.py`, `test_matcher_n_jobs_from_yaml_equals_torch_matcher`,
`test_unsupported_matcher_configs_raise`; 5.8 `test_org_model.py`, `test_mf_golden.py` (`mf_org`, paired delta),
`test_lane_integration.py`, `test_mf_jit.py`; 3.3 `test_org_loss_fixes.py`, `test_objective_e2e.py`.

## Mutants (scratch rsync copies; `retarget/critique/mut/`)

| Mutant | Stage / gate | Failing targets |
|---|---|---|
| none (native targets; hybrid targets) | | 0 of 136; 0 of 94 |
| n1 `host_match` without `nan_to_num` | R1, 5.7 | 2 (`[nan-and-inf]` 100-batch row, sanitising-step test) |
| n2 `proj_*` init U(±1/k) instead of U(±1/√k) | R2, 5.6 init | 2 |
| n7 unknown attention keys accepted | R2, 5.1 | 2 |
| n3 `self_attn_norm_kv` ignored | R3, 5.3 | 3 (decoder layer depth 0, both org-model rows) |
| n4 FastLayerNorm affine | R3, 5.2 | 16 (variable counts, `test_variables` enc_lin x3, ...) |
| n5 `sort_index` without the `+inf` fill | R3, 5.5 | 5 (valid-prefix, position stub, org model, ...) |
| n6 `legacy` placement read as `hybridnorm` | R3, 5.2 | 9 |
| h1 hybrid attention ignores `kv_sort_idx` | R4b, 5.6 / H0 | 6 (`q_ca` rows, paired delta, full-model linformer x2) |
| h2 hybrid encoder does not sort | R4a, 5.5 | 4 (position stub x2, full-model linformer x2) |
| b1, b1dice, b2 | R1, 3.3 | 18, 15, 10 (row 3 above) |

## Fixes (committed)

1. `156bc97` `port_to_keras.build_keras_maskformer_from_config` dropped `input_sort_field` for a torch config, so
   `eval_keras`, `evaluate_hgq` and `port_to_keras` built an **unsorted** hybrid for the org model (R4b open issue,
   unowned file). One line; new `tests/keras/test_port_to_keras_config.py`: sorted build against org torch
   3.6e-6 to 4.5e-6 (seeds 0-2; tolerance 1.02e-5, the `org_model.linformer` row), masks equal; control without
   sorting 1.71 / 9.31 / 21.6 with 4/4 masks different. With the fix reverted the gate test fails (4 masks differ).
2. `97cc129` `test_pr8_layers.py`: the `skipif` on the R2 `Attention` signature (always false since R2) could only
   hide 3 masked-Linformer rows if the signature changed; removed. 37 tests collected before and after; 37 pass.
3. This log, the `RETARGET_ORG_MAIN.md` status line and a README row.

## Open issues (not fixed here)

- Native `from_config` refuses a jsonargparse-normalised config (a saved `config.yaml`): `activation:
  {'class_path': 'hepattn.models.SwiGLU'}` raises `TypeError: activation must be a name`. The raw YAML and the
  generated Polaris config work. Matters for S6 if the native driver reads saved run configs.
- Still open from R4b/R5: `port_to_keras` fails for the org model at `change_attn_backends` and needs `materialize`
  before `state_dict()`; `evaluate_hgq --float-state` needs `materialize`; `polaris/07_profile.py` builds the
  pre-retarget model; EBOPs setpoints and the `quant_losses()` double count; the S7 `kept` leaf export form;
  the Linux/GPU re-measure; G2 on Polaris.
- `polaris/14_backend_spike.py:217-222` docstrings name the deleted `KerasLinformerAttention` (a spike script, history).
