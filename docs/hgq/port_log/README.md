# Keras-native port: stage logs

One log per stage of [KERAS_JAX_PLAN.md](../KERAS_JAX_PLAN.md) section 4: decisions, deviations from the
plan, measured numbers, gate evidence (with the control or mutation that made each gate fail), open issues.
Setup and test commands: [KERAS_NATIVE.md](../KERAS_NATIVE.md).

| Stage | Log | Branch / worktree |
|---|---|---|
| S1 env, skeleton, factory move, import hygiene | [S1.md](S1.md) | `keras-jax` / `hepattn-keras-jax` |
| S2 layers, S3 decoder + MaskFormer (lane A) | [S2.md](S2.md), [S3.md](S3.md) | `kn/lane-a` / `hepattn-kn-a` |
| S4 costs, losses, matching (lane B) | `S4.md` | `kn/lane-b` / `hepattn-kn-b` |

## Gates carried to a later stage

Each item must be closed (or explicitly re-deferred) in the named stage's log.

| From | To | Gate | Why deferred |
|---|---|---|---|
| S2 | S3 | ~~Full-model paired delta: layer-0 mask logit, high ~0.30 vs low ~31, >= 10x.~~ **Closed in S3**: native 0.30144 / 31.0916 vs hybrid 0.30141 / 31.0916, 103x ([S3.md](S3.md)). | Needed the S3 decoder and mask task. |
| S2, S3 | first Linux/GPU run | Re-measure the golden rows whose tolerance is 0 (bit-exact on macOS arm64 CPU): S2 `golden/make_golden.py` rows, S3 `golden/measure_mf_golden.py` rows. | 2x a measured 0 is 0 ([S2.md](S2.md) deviation 7, [S3.md](S3.md)). |
| S2 | S3 | ~~Task-head `Dense` layers start from the hybrid's init (`init="torch"`), with an init test.~~ **Closed in S3** (`test_task_heads_decoder_and_queries_start_from_the_hybrid_init`). S4 builds no Dense heads. | |
| S3 | Polaris | G2: jitted native forward + backward <= 0.5x the hybrid bare-loop step, same batch, fp32, EBOPs every step (`polaris/15_g2_forward.pbs`). | Needs one A100; validated on CPU only ([S3.md](S3.md)). |
| S3 | S5 | EBOPs penalty semantics: use `stateless_call(return_losses=True)` (each layer once). The hybrid's `quant_losses()` counts the task heads twice (e.g. prod 804.9605 vs 804.9105), so S5 penalty checks compare against the per-layer sum; eager `model.losses` counts repeated task calls. | [S3.md](S3.md) deviation 9. |
| S3 | S5, S8 | WRAP datalane overflow (HGQ2 default) makes native-vs-hybrid quantized outputs chaotic at production bitwidths; parity at prod is gated before the mask feedback only. Training-level comparisons (G4) must not expect forward parity. | [S3.md](S3.md) deviation 8. |
| S3 | S4, S6 | Regression proxies depend on top-k tie-breaking for quantized incidences (torch CPU, torch CUDA and JAX differ). | [S3.md](S3.md) deviation 7. |
