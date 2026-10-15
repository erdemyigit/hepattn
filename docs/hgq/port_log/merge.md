# Merge log: lanes A and B into `keras-jax`

Date 2026-10-15, macOS arm64, CPU only. Both lanes branch from `fb5ac2d` (S1).

## What was merged
- `4581847` merge of `kn/lane-a` at `1c5ddc9` (S2 layers, S3 decoder/tasks/MaskFormer, `polaris/15_g2_forward.*`).
- `47564c1` merge of `kn/lane-b` at `834898c` (S4 costs, losses, host Hungarian matching).
- No textual conflicts: the two lanes touch disjoint files (`comm -12` of the two `git diff --name-only fb5ac2d` lists is empty).
  Each lane's code and tests are unchanged by the merge.
- Lane worktrees `hepattn-kn-a` / `hepattn-kn-b` are kept. Nothing was pushed.

## Interface check across the lanes (new test)
Neither lane tested lane A's model against lane B's objective. `tests/keras_native/test_lane_integration.py`
(commit `aa2a7e7`) does this: the quantized (high) Linformer MaskFormer at CLIC test shapes, on 3 CLIC dummy
batches whose padded-node targets are zeroed (`real_like`), inside `jax.jit(jax.value_and_grad(stateless_call +
matched_loss(clic_specs())))`.
- Measured: 1 compile over 3 batches; 3 host matching calls in 3 steps; task loss 52.88 / 50.27 / 49.12;
  0 non-finite gradients; every one of the non-quantizer weights gets a nonzero gradient. With the task loss alone,
  40-42 of 490 trainable variables (all HGQ2 quantizer `i`/`b` bits) get zero gradient. They get it from the HGQ2
  penalty, which `test_mf_jit.py` checks.
- Second test (control for `real_like`): raw dummy targets make the matched loss `inf` (S4.md finding on `mask_bce`).
- Mutations (each in a copy of the working tree under the session scratchpad, run on this file):
  | Mutation | Result |
  |---|---|
  | X0 none | 2 passed |
  | X1 `stop_gradient` on the incidence output (`tasks.py`) | 1 failed: "10 model weights get no gradient", all `task2_net` |
  | X2 the model emits `{"encoder": {"query_mask": 0}}` | 2 failed: `NotImplementedError` from `objective.py` |
  | X3 `real_like` returns the targets unchanged | 2 failed: losses `[inf, inf, inf]` |

Side observation for S5: in the same probe, `sum(stateless_call losses)` at the `QUANT_SPECS["high"]` test spec was
3216651583488.0 (3.2e12) and the same for all 3 batches. Not investigated here. S5's penalty gate (plan 3.6, uint32 EBOPs
saturation) must explain it before the penalty is put into a training loss.

## Test runs on the merged tree (before `aa2a7e7`)
- Native, `.venv-jax`, `KERAS_BACKEND=jax`, `PYTHONPATH=hepattn-keras-jax/src`, `tests/keras_native`:
  `348 passed, 1 skipped, 703 warnings in 244.84s (0:04:04)`. Skip: `test_init.py:114: SwiGLU is float only`.
  348 = 65 (S1) + 235 (lane A) + 48 (lane B), the sum of the lane counts (lane A 300, lane B 113, each including S1's 65).
- Hybrid, `hepattn/.venv`, `KERAS_BACKEND=torch`, `tests/keras`: `104 passed, 8 skipped, 42535 warnings in 167.78s`,
  the same counts as S1.
- `tests/polaris` (hybrid env): `3 passed, 6 warnings in 3.21s`.
- Native again after `aa2a7e7` (with the 2 new tests): `350 passed, 1 skipped, 715 warnings in 241.68s (0:04:01)`.
