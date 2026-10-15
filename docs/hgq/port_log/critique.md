# Completeness critique of S1-S4 (after the merge)

Date 2026-10-15, macOS arm64, CPU only. Tree: `keras-jax` at `b229530` (merge of lanes A and B + cross-lane test).
Scope: every S1-S4 gate of plan section 4, the plan 3.2/3.3/3.4 component lists, the torch-import rule, G2 run
locally at tiny dims, and the docs (KERAS_NATIVE.md, port_log, plan Status) against the code.

## Suite runs (re-measured, not copied from merge.md)
- Native (`.venv-jax`, JAX), at `b229530`: `350 passed, 1 skipped in 243.49s`; skip `test_init.py:114: SwiGLU is float only`.
  After this critique's test changes (+3 tests): `353 passed, 1 skipped in 240.89s`, rc 0.
- Hybrid `tests/keras` + `tests/polaris` (`hepattn/.venv`, torch): `107 passed, 8 skipped in 163.76s` (104 + 3, as merge.md).
- `src/hepattn/keras` is unchanged since S1 (`git diff fb5ac2d..HEAD -- src/hepattn/keras` is empty); since `197ede2`
  only `factory.py` changed (the planned shim).
- Torch rule: an independent AST scan (own script; `import`, `from`, `importlib.import_module`, `__import__`, any
  depth) over the 15 `keras_native` modules finds 0 imports of torch / lightning / torchmetrics or of a hepattn
  module outside `keras_native`. The only `torch` matches of `grep` are docstrings and comments.

## Gate -> test map, with mutations run in this critique
Each mutation ran in a fresh copy of the tracked files under the session scratchpad (never in the worktree; the
runner asserts that each edit applies exactly once). Controls (no mutation) on the same targets: 24 passed
(hygiene + masked query), 8 passed (hybrid Linformer + attention parity), 9 passed (new tests, `-k "lin or threshold"`).

| Gate (plan section 4) | Test(s) | Mutation in this critique | Result |
|---|---|---|---|
| S1 no torch in `keras_native` | `test_import_hygiene.py` | M1 `import torch` inside `matching.match_one` (never called at import) | 1 failed: `test_keras_native_has_no_forbidden_import_statements` (the runtime scan alone passes) |
| S1 hybrid suite through the factory shim | `tests/keras` (hybrid env) | M2h `SoftmaxOp` keeps NaN rows (drops the `any_valid` zeroing) in `keras_native/factory.py` | 1 failed: hybrid `test_linformer_parity.py::test_fully_masked_query_is_key_independent` |
| S2 float parity | `test_float_parity.py` | M3 Linformer scale `dim**-0.5` instead of `head_dim**-0.5` | 4 failed: Linformer self/cross/distinct-v, CLIC Linformer encoder |
| S2 fully-masked query | `test_masked_query.py` | M2n (as M2h, native suite) | 2 failed: `[linformer-float]`, `[dense-float]` |
| S2 quantized golden + paired delta | `test_quantized_golden.py` | M4 dense attention scales the scores after the einsum (same in float) | 9 failed: golden attn_self/attn_cross/enc_torch high+low, 3 paired deltas. Every float attention/encoder parity test passed (36 passed) |
| S2 JIT (1 compile, nonzero grads, EBOPs) | `test_jit.py` | M11 `stop_gradient(proj_v)` | 3 failed: lin_self, lin_cross, enc_lin |
| S2 JIT, NaN gradient | `test_jit.py` | M12 `where(True, proj_v, sqrt(-abs(proj_v) - 1))`: finite forward, NaN gradient | **before this critique: 4 passed (gap)**; after the fix: 3 failed with `non-finite gradient ... proj_v` |
| S3 `maskformer.e2e` float, masks exactly equal | `test_maskformer_e2e_float_parity` | M6 all-False restore along axis -2 | 4 failed: e2e torch/lin, all-False torch/lin |
| S3 mask threshold `>=` (part of "masks exactly equal") | new `test_mask_attention_threshold_is_inclusive_as_torch` | M5 `>=` -> `>` in `ObjectHitMaskTask.attn_mask` | **before: 58 passed (all of `test_maskformer.py` + `test_mf_golden.py`, gap)**; after: 2 failed |
| S3 all-False restore + off control | `test_all_false_rows_attend_everywhere` | M6 | failed (above) |
| S3 unsupported configs raise | `test_unsupported_configs_raise` | M7 kmeans `raise` replaced by `pass`; M13 sorter dropped from the sorting check | M7: 1 failed (`cross_attn_mode`); M13: 1 failed on the new `sorter` row (no row covered `sorter` before) |
| S3 compile once | `test_forward_compiles_once`, `test_mf_jit.py` | not mutated here (lane mutation J1: 2 failed) | - |
| S4 costs to 1e-5 | `test_task_costs_equal_torch` | M10 dice cost numerator smoothing 1.0 -> 0.5 | 4 failed: 3 batch sizes + dice-pad cost |
| S4 matched total (100 batches) | `test_matched_total_cost_equals_torch_matcher_over_100_batches` | M8 `linear_sum_assignment(..., maximize=True)` | 2 failed: total cost, default indices |
| S4 weighted CE | `test_weighted_ce_equals_torch_and_is_not_the_plain_mean` | M9 class weights ignored | 1 failed |
| S4 loss rows, callback grads, NaN targets, permutation | `test_objective_e2e.py`, `test_matching.py` | not mutated here (lane mutations M1-M15, R1-R18 in S4.md) | - |
| S3 G2 (Polaris) | `polaris/15_g2_forward.py` | deferred (needs an A100) | local sweep below |

## G2 script at tiny dims (CPU)
`15_g2_forward.py --sweep --dim 32 --heads 4 --enc-layers 1 --dec-layers 1 --jax-batches 2 --hybrid-batches 2
--warmup 1 --iters 3`, rc 0. jax 27.3 ms (312/312 gradients nonzero, penalty 104.8, `cache_size` 1), hybrid-07 248.2 ms
(292/312, penalty 104.9: the double count), hybrid-surrogate 257.2 ms (312/312, 104.8). Same numbers as S3.md's local
validation within CPU noise. Each arm now records `hepattn_file`; all 3 point into `hepattn-keras-jax/src`. Not a G2 result.

## Plan 3.2 / 3.3 / 3.4 components
All implemented: Dense/SwiGLU (`dense.py`), Linformer with `RandomUniform(+-1/sqrt(k))` and a padded `keep`
(`linformer.py:80-81,122`), dense attention, Residual/norms/LayerScale/register tokens (`encoder.py`), InputNet +
non-trainable posenc `B` (`input.py`), decoder with all-False restore (`decoder.py:118-121,233`), the 4 tasks
(`tasks.py`; `finfo.min` sentinel at 235, `top_k` + `one_hot` at 64-65, `FeatureScaler` at 352), quantization
losses through `stateless_call` (tested in `test_mf_golden.py`), all 3.3 losses (`losses.py`; `nan_to_num` in
`objective.py:71`), 3.4 option A (`matching.py:65-68`, `take_along_axis` at 76). Deferred with a reason: 3.4 option
C (on-device optax matching) to S9; DropPath raises (needs SeedGenerator state); other norms, attention types and
decoder options raise (S2.md deviation 2, S3.md deviation 4). Nothing silently missing.

## The 3.2e12 HGQ2 penalty at the `high` test spec (merge.md side observation)
Measured (scratch probe; `lin` model, `QUANT_SPECS["high"]`, 2 dummy events): 411 terms, sum 3.21665e12. Per layer
(own losses after one training call): `decoder_l{0,1}_kv_ca_softmax_exp_table` 8.301e11 each,
`decoder_l{0,1}_q_ca_softmax_exp_table` 7.782e11 each; the next term is 2.078e5 (encoder softmax tables). So 99.99%
is the 4 decoder cross-attention exp tables at table f0 = 16, beta 1e-5. They do not depend on the data (merge.md:
the same sum for 3 batches). It is not uint32 saturation: the loss is a float; 9 of 124 stored `ebops` variables are
saturated at 2^32 - 1, which only matters for read-outs of the stored value (plan 3.6). S5's penalty gate must be
stated per layer (or at the production spec, where S3 measured a total of 1678.1 at beta 1e-12).

## Fixed in this critique
- `test_jit.py`: rejects non-finite gradients (M12 above).
- `test_maskformer.py`: `test_mask_attention_threshold_is_inclusive_as_torch` (torch `task.py:571` `>=`; M5 above)
  and a `sorter` row in `test_unsupported_configs_raise` (M13).
- `maskformer.py` docstring: eager `model.losses` is not equivalent to the `stateless_call` losses (S3.md deviation 9).
- `polaris/15_g2_forward.py`: records `hepattn.__file__`.
- Docs: KERAS_NATIVE.md module map (it described S1 only) and the note that `matching.py` imports jax; plan Status
  (S5 sorter item was already satisfied; minor items updated); port_log README carried-gate rows for S4 and merge.

## Open issues (not fixed here)
- (Checked, no action) `test_mf_jit.py:92` and `test_lane_integration.py` already assert finite gradients; only the
  per-layer `test_jit.py` lacked it.
- G2, the 0-tolerance golden rows on Linux/GPU, and the cross-event mask loss decision stay open (plan Status).
- The native suite never runs in the CI unit job (S1 R1); there is no CI job for it.
