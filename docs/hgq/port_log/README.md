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

## Gates carried to a later stage

Each item must be closed (or explicitly re-deferred) in the named stage's log.

| From | To | Gate | Why deferred |
|---|---|---|---|
| S2 | S3 | ~~Full-model paired delta: layer-0 mask logit, high ~0.30 vs low ~31, >= 10x.~~ **Closed in S3**: native 0.30144 / 31.0916 vs hybrid 0.30141 / 31.0916, 103x ([S3.md](S3.md)). | Needed the S3 decoder and mask task. |
| S2, S3 | first Linux/GPU run | Re-measure the golden rows whose tolerance is 0 (bit-exact on macOS arm64 CPU): S2 `golden/make_golden.py` rows, S3 `golden/measure_mf_golden.py` rows. | 2x a measured 0 is 0 ([S2.md](S2.md) deviation 7, [S3.md](S3.md)). |
| S2 | S3 | ~~Task-head `Dense` layers start from the hybrid's init (`init="torch"`), with an init test.~~ **Closed in S3** (`test_task_heads_decoder_and_queries_start_from_the_hybrid_init`). S4 builds no Dense heads. | |
| S3 | Polaris | G2: jitted native forward + backward <= 0.5x the hybrid bare-loop step, same batch, fp32, EBOPs every step (`polaris/15_g2_forward.pbs`). Decided at batch 16 fp32 (torch fp32 OOMs at 32); b8 is the second point, JAX b32 is scaling only. | Needs one A100; validated on CPU only ([S3.md](S3.md), "G2 decision rule"). |
| S3 | S5 | EBOPs penalty semantics: use `stateless_call(return_losses=True)` (each layer once). The hybrid's `quant_losses()` counts the task heads twice (e.g. prod 804.9605 vs 804.9105), so S5 penalty checks compare against the per-layer sum; eager `model.losses` counts repeated task calls. | [S3.md](S3.md) deviation 9. |
| S3 | S5, S8 | WRAP datalane overflow (HGQ2 default) makes native-vs-hybrid quantized outputs chaotic at production bitwidths; parity at prod is gated before the mask feedback only. Training-level comparisons (G4) must not expect forward parity. | [S3.md](S3.md) deviation 8. |
| S3 | S4, S6 | Regression proxies depend on top-k tie-breaking for quantized incidences (torch CPU, torch CUDA and JAX differ). | [S3.md](S3.md) deviation 7. |
| S4 | S5 | Pass `n_jobs=matcher_n_jobs_from_config(...)` (16 for `pflow_hgq.yaml`) to `matched_loss`; the default 1 is about 6.5x slower at B=32. Check `jax_default_matmul_precision` (TF32) for the fp32 costs on the A100. | Train step not in S4 ([S4.md](S4.md) open issues). |
| S4 | user | Torch `mask_bce` / `mask_dice` average across events at B > 1; S4 keeps it. Change it in both frameworks or keep it. | User decision ([S4.md](S4.md) finding 1). |
| S4 | S6 | Real CLIC batches: no target 1 at padded nodes (else `mask_bce` is inf), prefix-contiguous `particle_valid`. | Needs real data ([S4.md](S4.md) open issues). |
| merge | S5 | The `high`-spec HGQ2 penalty (3.2e12) is 99.99% decoder cross-attention softmax exp tables; the penalty gate must be stated per layer, not as one total. | Explained in [critique.md](critique.md). |
| R0 | R5 / pre-push | ~~`pixi lock` on linux-64, then `pixi lock --check` exits 0.~~ **Closed in R0 fix round 2** (`9a6bde5`, falcon, pixi v0.50.2). Re-check `pixi lock --check` on falcon before the push if any lane edits `pyproject.toml`. | macOS cannot solve the linux platforms ([R0.md](R0.md) deviation 2). |
| R0 | plan owner | 12 strict xfails in `test_jit.py`, `test_maskformer.py`, `test_variables.py` are blocked on R5's `golden/{cases,mf_cases}.py`: decide which lane edits what. | File ownership ([R0.md](R0.md) open issues). |
| R0 | R4b | Rewrite the `both` / `legacy` rows of `tests/keras/test_build_attention_linformer_k.py` when `linformer_proj_dim` starts to raise. | Plan 5.1. |
| R0 | plan owner | Assign `src/hepattn/experiments/clic/configs/pflow_hgq.yaml` (still `linformer_proj_dim: 256`; read by R1, R2, R3 and R5 tests) to R2 or R5 before R2 makes unknown attention keys raise. | No lane owns it ([R0.md](R0.md) open issues). |
| R0 | R4a | The generated Polaris config builds `proj_k.shape[-1] == 64` in every attention (asserted in the generated-config gate). | `linformer_k: 256` equals the hybrid default, so the R0 hunk is unobservable ([R0.md](R0.md) open issues). |
| R0 | R4a, R4b | Keep `tests/keras/test_pr8_layer_options.py` green (PR #8 options against org torch, depth 0 and 1) through the hybrid rewrite. | Added in R0 fix round 2; the attribute-only suite passed with each option ignored. |
