# Completeness critic log (R0-R5), 2026-10-16
KJ=/Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-keras-jax at keras-jax=rt/integration=2435fa8
## Findings
- (1) native from_config(raw base+linformer merge) vs org torch built by jsonargparse (LightningCLI merge), dim 256 prod dims, 3 dummy events (55/78/100 valid of 160), biases perturbed: seed 7 worst 6.7e-6 (mask logits, scale 5), proxy_regr 3.05e-5 at scale 132; 0 mask flips; 17 losses max 9.5e-7. seed 11 worst 7.4e-6. Control nosort: 2.29, 77-253 flips, losses 0.043. logs/native_e2e_*.out
  * finding: from_config rejects a jsonargparse-normalized config (activation {'class_path': 'hepattn.models.SwiGLU'}) -> TypeError. Saved config.yaml form. Minor (raw yaml OK).
- (2) hybrid KerasMaskFormer float, same: worst 7.2e-6, 0 flips, losses 9.5e-7; cores AttendCore/ProjectCore. Control nosort 2.29. logs/hybrid_e2e_7_*.out
- hybrid suite c1: 283 passed / 8 skipped (2 csim, 6 CUDA) EXIT 0 (421 s). matches R5.
- (3) bug reverts in scratch copies (crit/mut/{b1,b1dice,b2}; targets org_loss_fixes+objective_parity+objective_e2e+org_model, 67 tests): m0 67 pass; b1 (bce+dice cross-event) 18 fail; b1dice 15 fail; b2 (_class_prob into cost) 10 fail incl B2-a/b/c.
- (6) 03_make_config -> $S/crit/polaris_hgq.yaml: hybrid via jsonargparse float == org torch (worst 7.2e-6, 0 flips); quant train fwd+loss finite (17 losses, AttendCore/ProjectCore); native from_config float worst 3.05e-5 (proxy_regr scale 132), quant matched total finite.
- native suite c1: 543 passed, 0 skip/xfail, EXIT 0 (517 s). keras_native import of all 14 modules loads 0 torch/lightning/masked_linformer modules.
- G2 tiny local (jax b2,b4; hybrid b2): EXIT 0, 416/416 grads jax+surrogate, penalty 0.1312 equal, sort phi. crit/g2_tiny.json
- fix: test_pr8_layers R2_REQUIRED skipif (always false now, latent silent skip) removed; collection 37 both.
- finding: port_to_keras.build_keras_maskformer_from_config drops input_sort_field for torch configs (eval_keras, evaluate_hgq): unsorted hybrid for the org model. Fixing (one line + test).
- fix port_to_keras input_sort_field + tests/keras/test_port_to_keras_config.py: sorted 4.5e-6/3.6e-6/3.9e-6 (seeds 0-2, tol 1.02e-5), masks equal; unsorted 1.71/9.31/21.6, 4/4 masks differ. Fix reverted -> test fails (4 attn masks differ).
- torch suite c1: 522 passed / 173 skipped EXIT 0; matching: 106 failed (all lap1015 AttributeError) ran=172 other=0, gate EXIT 0; 2 xfails = org/main's own (tests/{matching,models,loss} identical to org/main).
- uv lock --check exit 0.
- mutants (scratch copies): n1 nan_to_num removed -> 2 fail (test_matching); n2 proj init 1/k -> 2 fail; n7 unknown-key check removed -> 2 fail; n3 self_attn_norm_kv ignored -> 3 fail.
- commits 156bc97 (port_to_keras), 97cc129 (test_pr8_layers skip)
- mutants n4 16, n5 5, n6 9, h1 6, h2 4 fail; m0 136/94 pass. R-critique.md written.
