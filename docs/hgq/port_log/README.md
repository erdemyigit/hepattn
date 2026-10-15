# Keras-native port: stage logs

One log per stage of [KERAS_JAX_PLAN.md](../KERAS_JAX_PLAN.md) section 4: decisions, deviations from the
plan, measured numbers, gate evidence (with the control or mutation that made each gate fail), open issues.
Setup and test commands: [KERAS_NATIVE.md](../KERAS_NATIVE.md).

| Stage | Log | Branch / worktree |
|---|---|---|
| S1 env, skeleton, factory move, import hygiene | [S1.md](S1.md) | `keras-jax` / `hepattn-keras-jax` |
| S2 layers, S3 decoder + MaskFormer (lane A) | [S2.md](S2.md), `S3.md` | `kn/lane-a` / `hepattn-kn-a` |
| S4 costs, losses, matching (lane B) | `S4.md` | `kn/lane-b` / `hepattn-kn-b` |

## Gates carried to a later stage

Each item must be closed (or explicitly re-deferred) in the named stage's log.

| From | To | Gate | Why deferred |
|---|---|---|---|
| S2 | S3 | Full-model paired delta (plan section 4, S2): layer-0 mask logit, high ~0.30 vs low ~31, separation >= 10x. | Needs the S3 decoder and mask task. S2 applies the criterion per layer ([S2.md](S2.md), deviation 6). |
| S2 | first Linux/GPU run | Re-measure the golden rows whose S2 tolerance is 0 (bit-exact on macOS arm64 CPU). | 2x a measured 0 is 0 ([S2.md](S2.md), deviation 7). |
