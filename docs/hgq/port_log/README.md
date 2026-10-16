# Keras-native port: stage logs

One log per stage of [KERAS_JAX_PLAN.md](../KERAS_JAX_PLAN.md) section 4: decisions, deviations from the
plan, measured numbers, gate evidence (with the control or mutation that made each gate fail), open issues.
Setup and test commands: [KERAS_NATIVE.md](../KERAS_NATIVE.md).

| Stage | Log | Branch / worktree |
|---|---|---|
| S1 env, skeleton, factory move, import hygiene | [S1.md](S1.md) | `keras-jax` / `hepattn-keras-jax` |
| S2 layers, S3 decoder + MaskFormer (lane A) | [S2.md](S2.md), [S3.md](S3.md) | `kn/lane-a` / `hepattn-kn-a` |
| S4 costs, losses, matching (lane B) | [S4.md](S4.md) | `kn/lane-b` / `hepattn-kn-b` |
| Merge of lanes A and B, cross-lane interface test | [merge.md](merge.md) | `keras-jax` / `hepattn-keras-jax` |
| Completeness critique of S1-S4 after the merge | [critique.md](critique.md) | `keras-jax` / `hepattn-keras-jax` |
| Retarget to org/main `ac0da8f` ([RETARGET_ORG_MAIN.md](../RETARGET_ORG_MAIN.md)): R0 merge of org/main, lock files, strict xfails | [R0.md](R0.md) | `rt/integration` / `hepattn-keras-jax` |
| R1 bugs 1 and 2 (per-event mask losses, class cost on logits), matcher (`nan_to_num`, `lap1015_late` via scipy) | [R1.md](R1.md) (`R1_mutants.py`) | `rt/r1` / `hepattn-rt-r1` |
| R2 quantized masked-Linformer core and native `Attention` branch | [R2.md](R2.md) (`R2_mutants.py`) | `rt/r2` / `hepattn-rt-r2` |
| R3 config, FastLayerNorm, dense-norm placement, decoder order, quantized SwiGLU, sorting | [R3.md](R3.md) (`R3_evidence/`) | `rt/r3` / `hepattn-rt-r3` |
| R4a hybrid: quantized SwiGLU, sorting, generated org-model Polaris config | [R4a.md](R4a.md) | `rt/r4` / `hepattn-rt-r4` |
| Merge of R1-R4a, cross-lane wiring test | [R-merge.md](R-merge.md) | `rt/integration` / `hepattn-keras-jax` |
| R4b hybrid masked-Linformer core (gate H0), beta/EBOPs on nested leaves | [R4b.md](R4b.md) | `rt/integration` / `hepattn-keras-jax` |
| R5 goldens, full-model gates, Helen's Linformer deleted, hls4ml smoke, G2 script on the org model | [R5.md](R5.md) (`R5_hls_smoke.py`) | `rt/integration` / `hepattn-keras-jax` |

## Gates carried to a later stage

Each item must be closed (or explicitly re-deferred) in the named stage's log.

| From | To | Gate | Why deferred |
|---|---|---|---|
| S2 | S3 | ~~Full-model paired delta: layer-0 mask logit, high ~0.30 vs low ~31, >= 10x.~~ **Closed in S3**: native 0.30144 / 31.0916 vs hybrid 0.30141 / 31.0916, 103x ([S3.md](S3.md)). | Needed the S3 decoder and mask task. |
| S2, S3, R4b, R5 | first Linux/GPU run | Re-measure the golden rows whose tolerance is 0 (bit-exact on macOS arm64 CPU): `golden/make_golden.py` rows (R5: all `low` rows of the masked-Linformer and SwiGLU cases), `golden/measure_mf_golden.py` rows (R5: `mf_org`), the H0 backend rows and 16-head bounds of `tests/keras/golden/linformer_backend.npz` (R4b), and every 2x tolerance measured on macOS (R1-R5). | 2x a measured 0 is 0 ([S2.md](S2.md) deviation 7, [S3.md](S3.md), [R4b.md](R4b.md), [R5.md](R5.md)). |
| S2 | S3 | ~~Task-head `Dense` layers start from the hybrid's init (`init="torch"`), with an init test.~~ **Closed in S3** (`test_task_heads_decoder_and_queries_start_from_the_hybrid_init`). S4 builds no Dense heads. | |
| S3 | Polaris | G2: jitted native forward + backward <= 0.5x the hybrid bare-loop step, same batch, fp32, EBOPs every step (`polaris/15_g2_forward.pbs`). Decided at batch 16 fp32 (torch fp32 OOMs at 32); b8 is the second point, JAX b32 is scaling only. **R5:** both arms now build the org/main model from the generated Polaris config (k = 64, sorting, masked attend); run G2 on it, not on the old model (RETARGET_ORG_MAIN.md section 7). | Needs one A100; validated on CPU only ([S3.md](S3.md), "G2 decision rule"; [R5.md](R5.md), tiny dims). |
| S3 | S5 | EBOPs penalty semantics: use `stateless_call(return_losses=True)` (each layer once). The hybrid's `quant_losses()` counts the task heads twice (e.g. prod 804.9605 vs 804.9105), so S5 penalty checks compare against the per-layer sum; eager `model.losses` counts repeated task calls (org model: 52 extra terms, `test_mf_jit.py`). On the org model the task heads are 10.8% of the EBOPs, so fix `quant_losses()` before re-deriving the beta setpoints ([R4b.md](R4b.md)). | [S3.md](S3.md) deviation 9. |
| S3 | S5, S8 | WRAP datalane overflow (HGQ2 default) makes native-vs-hybrid quantized outputs chaotic at production bitwidths; parity at prod is gated before the mask feedback only. Training-level comparisons (G4) must not expect forward parity. | [S3.md](S3.md) deviation 8. |
| S3 | S4, S6 | Regression proxies depend on top-k tie-breaking for quantized incidences (torch CPU, torch CUDA and JAX differ). | [S3.md](S3.md) deviation 7. |
| S4 | S5 | Pass `n_jobs=matcher_n_jobs_from_config(...)` (16 for `pflow_hgq.yaml` and for org `base.yaml`, whose `lap1015_late` native solves with scipy, R1) to `matched_loss`; the default 1 is about 6.5x slower at B=32. Check `jax_default_matmul_precision` (TF32) for the fp32 costs on the A100. | Train step not in S4 ([S4.md](S4.md) open issues). |
| S4 | user | ~~Torch `mask_bce` / `mask_dice` average across events at B > 1; S4 keeps it. Change it in both frameworks or keep it.~~ **Closed by org `a4263c1` (bug 1)**: per-event normalisation in torch, hybrid and native since R1 ([R1.md](R1.md) section 3.3 gates). | User decision ([S4.md](S4.md) finding 1). |
| S4 | S6 | Real CLIC batches: prefix-contiguous `particle_valid`; no target 1 at padded nodes. The latter is no longer fatal: since bug 1 a target 1 at a padded node gives a finite `mask_bce` (R1 B1-d; `test_lane_integration.py::test_dummy_targets_give_a_finite_loss`), but it still changes the dice target sum. | Needs real data ([S4.md](S4.md) open issues). |
| merge | S5 | The `high`-spec HGQ2 penalty (3.2e12) is 99.99% decoder cross-attention softmax exp tables; the penalty gate must be stated per layer, not as one total. **R5:** that term came from QSoftmax's `-1e9` mask fill; the org model's masked attend softmax takes no mask, so it is gone (total 2.509e11 at the production spec, [R4b.md](R4b.md)). Still state the S5 penalty gate per layer, and re-derive `HGQ_TARGET_EBOPS`, `HGQ_INIT_BETA`, `HGQ_PID_I` on this model. | Explained in [critique.md](critique.md); plan risk 1. |
| R0 | R5 / pre-push | ~~`pixi lock` on linux-64, then `pixi lock --check` exits 0.~~ **Closed in R0 fix round 2** (`9a6bde5`, falcon, pixi v0.50.2). Re-check `pixi lock --check` on falcon before the push if any lane edits `pyproject.toml`. | macOS cannot solve the linux platforms ([R0.md](R0.md) deviation 2). |
| R0 | plan owner | ~~12 strict xfails in `test_jit.py`, `test_maskformer.py`, `test_variables.py` are blocked on R5's `golden/{cases,mf_cases}.py`: decide which lane edits what.~~ **Closed in R5**: all 50 strict xfails removed ([R5.md](R5.md)). | File ownership ([R0.md](R0.md) open issues). |
| R0 | R4b | ~~Rewrite the `both` / `legacy` rows of `tests/keras/test_build_attention_linformer_k.py` when `linformer_proj_dim` starts to raise.~~ **Closed in R4b** (`5ab1c1c`). | Plan 5.1. |
| R0 | plan owner | ~~Assign `src/hepattn/experiments/clic/configs/pflow_hgq.yaml` (still `linformer_proj_dim: 256`) before R2 makes unknown attention keys raise.~~ **Closed in R5**: it reads `linformer_k: 256` (masked Linformer at its old sizes; the org model is the generated Polaris config) ([R5.md](R5.md) deviations). | No lane owns it ([R0.md](R0.md) open issues). |
| R0 | R4a | ~~The generated Polaris config builds `proj_k.shape[-1] == 64` in every attention.~~ **Closed in R4a** (`proj_k` shapes {(168, 64), (160, 64)}, [R4a.md](R4a.md) gates). | `linformer_k: 256` equals the hybrid default, so the R0 hunk is unobservable ([R0.md](R0.md) open issues). |
| R0 | R4a, R4b | ~~Keep `tests/keras/test_pr8_layer_options.py` green through the hybrid rewrite.~~ **Closed**: green in R4a, R4b and R5 suites. | Added in R0 fix round 2; the attribute-only suite passed with each option ignored. |
| R5 | S7 | hls4ml: the renorm `kept` leaf as built (a `QEinsum` of the 1-bit mask with the computed (1, n, k) energy table) does not convert ("Batch dim mismatch"); its export form, a `QEinsumDense` with the frozen table as kernel, does. Every other new leaf kind generates code. | Plan section 7 risk 5 ([R5.md](R5.md), `R5_hls_smoke.py`). |
| R4b | before any QAT sweep | Re-derive the EBOPs setpoints (`polaris/03_make_config.py`) on the org model; fix `quant_losses()` double count first. | [R4b.md](R4b.md) open issues. |
