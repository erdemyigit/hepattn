# Merge log: lanes R1, R2, R3, R4a into `rt/integration`

Date 2026-10-15, macOS arm64, CPU only. Plan: [RETARGET_ORG_MAIN.md](../RETARGET_ORG_MAIN.md) section 6 (order R1 → R2 → R3 → R4).
`$S` = `/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad`;
merge evidence is in `$S/rmerge/` (`logs/`, `probes/`, `mut/`, runner `run.sh`). `/private/tmp` is not durable.
Nothing was pushed. The `hepattn` main working tree was not touched.

## Commits

| Commit | What |
|---|---|
| `389e081` | merge `rt/r1` at `43167bc` (bugs 1 and 2, matcher) |
| `236a5ae` | merge `rt/r2` at `6375c18` (masked-Linformer core and `Attention` branch) |
| `1478672` | merge `rt/r3` at `ff340b2` (config, norms, decoder order, quantized SwiGLU, sorting) |
| `107ffd6` | R3's `R3_evidence/proposed_test_float_parity.patch` applied to the R2-owned `test_float_parity.py` (`git apply --3way`, clean) |
| `87cb089` | merge `rt/r4` at `78a33ce` (R4a hybrid) |
| `9f65a07` | new wiring test `test_pr8_layers.py::test_org_linformer_model_equals_torch` (R3 file) |
| `cf586bb` | strict R5 xfails on the 22 native builds that still pass `linformer_proj_dim` (R3 and R5 files) |

No textual conflict in any of the four merges. The only file two lanes changed is `tests/keras_native/porting.py`
(R2: `port_attention`, `port_linformer`, `port_any_attention`; R3: `port_dense`, `port_norm`, one import), on
disjoint hunks, as section 6 allows. `pyproject.toml`, `uv.lock` and `pixi.lock` are unchanged since R0
(`git diff --stat 5db27a8 HEAD -- ...` is empty); `uv lock --check` exits 0.

## Ownership slips (no conflict, recorded)

Checked mechanically: each lane's `git diff --name-only 5db27a8...rt/<lane>` against its section-6 row.
- R1 `docs/hgq/port_log/R1_mutants.py`, R2 `docs/hgq/port_log/R2_mutants.py`, R3 `docs/hgq/port_log/R3_evidence/**`
  (25 files) are under `docs/hgq/**`, which R5 owns. All new files beside the stage logs; R5 keeps or moves them.
- R4a: none.
- The merge itself edits files of other lanes, because the task asks the merge driver to wire R3 to R2 and to leave
  only R4b/R5 xfails:
  - `tests/keras_native/test_float_parity.py` (R2): R3's patch, see `107ffd6`.
  - `tests/keras_native/test_pr8_layers.py` (R3): one new test, `9f65a07`.
  - markers only in `test_maskformer.py` (R3) and in `test_mf_golden.py`, `test_quantized_golden.py`, `test_mf_jit.py`,
    `test_lane_integration.py` (R5), `cf586bb`. No assertion was changed.

## R3 against the real R2 interface

R3 coded the R2 dependency two ways; both now run against R2's code:
- `R2_REQUIRED = skipif(not R2_LANDED)`: `R2_LANDED` is true after the merge, so `test_linformer_projection_shapes_equal_torch`
  and `test_pr8_layers_train_under_jit[decoder_layer_linformer]` run and pass (the native suite has 0 skips).
- `RecordingAttention` test doubles: they check that `kv_sort_idx` reaches `q_ca` only, but never the real core. New
  test `test_org_linformer_model_equals_torch` builds org `base.yaml` + `linformer.yaml` at dim 32 (masked-Linformer,
  k=64, phi sort, R3's encoder and decoder with R2's cores) and compares it with org torch with ported weights.
  Measured (`$S/rmerge/probes/probe_lin_model.out`, `probe_wire_controls.out`):

  | Quantity | Value |
  |---|---|
  | max abs error over all float outputs (valid slots) | 2.86e-6 (`layer_3.mask.pflow_node_logit`), tolerance `maskformer.e2e` 5e-5 / 1e-4 (plan 5.8 rows; R5 re-measures) |
  | attention-mask booleans, 4 layers × 48000 | 0 mismatches |
  | smallest margin / error to the 0.1 threshold | 1597x (layer 0), asserted >= 10x |
  | control: no input sorting, final mask logits | 1.127 (asserted > 10x tolerance) |
  | control: `kv_sort_idx` withheld from the decoder only | 0.970 = 3430x the tolerance |

  Mutants on `git archive 9f65a07` copies (`$S/rmerge/mut/summary.txt`), this test only:
  `m00_none` 1 passed; `w1` (attention never gathers by `kv_sort_idx`) 1 failed; `w2` (decoder passes `kv_sort_idx=None`
  to `q_ca`) 1 failed (both at `final.classification.pflow_class_prob`, 1.85e-3); `w3` (`q_sa` on the masked `project`
  core) passes here, as expected: its forward is identical, and R3's shape and JIT gates catch it (R3.md mutant g01).

## Cross-lane failures and how they were handled

After the four merges (`87cb089`), the native suite gave 456 passed, 28 xfailed, 19 failed, 3 errors
(`$S/rmerge/logs/m1.native.{log,xml}`). Every one of the 22 is `TypeError: linformer_proj_dim is not an org/main
Attention option` from native `build_attention` (R2, plan 5.1). Origins (junit tracebacks):
- 19 build `native_kwargs("lin")` from `golden/mf_cases.py` `LIN_KWARGS` (R5): `test_maskformer` 9, `test_mf_golden` 4,
  `test_mf_jit` 4 (1 failure, 3 fixture errors), `test_lane_integration` 2;
- 2 build `enc_lin` from `golden/cases.py` (R5): `test_quantized_golden::test_init_quantizer_state_matches_hybrid[enc_lin-*]`;
- 1 reads `pflow_hgq.yaml` (no owner; R0 open issue): `test_maskformer::test_from_config_reads_pflow_hgq_yaml`.

This is the set R2.md and both R2 reviewers reported, and R2 option (a) (strict xfail until R5 regenerates the
goldens). Changing the case kwargs here would only move the failures to the stale goldens, so `cf586bb` marks them
`xfail(strict=True, raises=TypeError)` with R5 reasons, as R0 did for the torch side. `test_maskformer.py` reuses its R0
marker `MF_CASES_LIN_XFAIL`; the pflow_hgq row has its own `PFLOW_HGQ_XFAIL` (R5 / plan owner).

R1's open issue `test_lane_integration.py::test_dummy_targets_give_the_documented_inf` (stale `inf` assertion) is now
inside this set: after R2 it raises the TypeError before it reaches the assertion. The marker reason tells R5 to apply
R1.md deviation 1 when `LIN_KWARGS` changes. I did not apply that replacement (R5's file; orchestrator decision).

Discrimination of the markers (`$S/rmerge/mut/x_blocker_removed.{log,xml,summary}`): a `git archive cf586bb` copy with
`linformer_proj_dim` → `linformer_k` in `mf_cases.py`, `cases.py` and `pflow_hgq.yaml`, running the 7 files that carry
xfails: 50 failed, 120 passed, 0 xfailed. All 50 markers (28 R0 + 22 merge) fail: 21 `XPASS(strict)`, the rest stale
goldens (`AssertionError`, `KeyError`, `RuntimeError`) or the `inf` assertion. So no marker can outlive its blocker.
Side result for the plan owner: with `linformer_k: 256`, `test_from_config_reads_pflow_hgq_yaml` passes (XPASS).

## Gates (final, at `cf586bb`)

| Gate | Result | Log |
|---|---|---|
| native `tests/keras_native` (`.venv-jax`, `KERAS_BACKEND=jax`) | **457 passed, 50 xfailed, 0 skipped, 0 failed, 0 xpassed** (225 s). All 50 xfails are strict, R5-reserved, `linformer_proj_dim` (28 from R0, 22 from this merge). The hepattn/.venv no-jax hygiene test ran | `$S/rmerge/logs/m2.native.{log,xml}` |
| hybrid `tests/keras tests/polaris` (`hepattn/.venv`, `KERAS_BACKEND=torch`) | **188 passed, 8 skipped, 6 xfailed** (the 6 strict R4b rows of R4a), 0 failed, EXIT 0. Run at `87cb089`; `9f65a07` and `cf586bb` change only `tests/keras_native`, which this run does not collect | `$S/rmerge/logs/m1.hybrid.{log,xml}` |
| torch `tests/models tests/polaris tests/loss -m 'not gpu and not requiresdata'` | **522 passed, 173 skipped**, 0 failed (= R0) | `$S/rmerge/logs/m2.torch.{log,xml}` |
| torch `tests/matching` | 106 failed, 66 passed, 14 skipped, 2 xfailed; `R0_matching_gate.py m2.matching.xml --max-failures 106`: `ran=172 failures=106 lap1015_attr=106 other=0`, exit 0 (lap1015 not built, plan section 2 gate (c)) | `$S/rmerge/logs/m2.matching.{log,xml}` |
| `uv lock --check` | exit 0 | `$S/rmerge/logs/uv_lock_check.log` |
| prek on the changed files | pass (`test_pr8_layers.py`: one `ARG005` noqa on the monkeypatched `call`) | |

Lane counts against this merge: R1 alone had the `inf` test failing (now xfailed through the TypeError); R2 alone 22
failures (now xfailed); R3 alone 3 float-parity failures (fixed by `107ffd6`; 0 here); R4a's hybrid counts are unchanged.

## Open issues

- R5: change `golden/mf_cases.py` `LIN_KWARGS` and `golden/cases.py` `enc_lin` to `linformer_k`, regenerate the goldens,
  and remove all 50 strict xfails (the control above lists them). Rewrite `test_dummy_targets_give_the_documented_inf`
  with R1.md deviation 1, and its stale docstrings (`test_lane_integration.py` lines with "mask_bce is inf").
- Plan owner: `pflow_hgq.yaml` still has no owner. Renaming its two `linformer_proj_dim: 256` to `linformer_k: 256` makes
  the native test pass and keeps the hybrid's k (it reads `linformer_k` first), but it changes what native builds from
  the hybrid's default training config (masked-Linformer instead of Helen's layer), so it is not done here.
- `test_unsupported_task_options_raise` is xfailed as a whole, so its first check (`mask_queries` raises) is also off
  until R5; it does not depend on `LIN_KWARGS`.
- R4b: the 6 strict hybrid xfails; H0. R5: `port_log/README.md` rows for R1-R4a and this log; the `docs/hgq/**`
  helper files listed under ownership slips; the remaining lane open issues (R1.md, R2.md, R3.md, R4a.md) are unchanged
  by this merge.
- Lane worktrees `hepattn-rt-r1..r4` are kept.
