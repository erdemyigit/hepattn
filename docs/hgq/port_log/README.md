# Keras-native port: stage logs

One log per stage of [KERAS_JAX_PLAN.md](../KERAS_JAX_PLAN.md) section 4: decisions, deviations from the
plan, measured numbers, gate evidence (with the control or mutation that made each gate fail), open issues.
Setup and test commands: [KERAS_NATIVE.md](../KERAS_NATIVE.md).

| Stage | Log | Branch / worktree |
|---|---|---|
| S1 env, skeleton, factory move, import hygiene | [S1.md](S1.md) | `keras-jax` / `hepattn-keras-jax` |
| S2 layers, S3 decoder + MaskFormer (lane A) | [S2.md](S2.md), `S3.md` | `kn/lane-a` / `hepattn-kn-a` |
| S4 costs, losses, matching (lane B) | `S4.md` | `kn/lane-b` / `hepattn-kn-b` |
