# Merge log: S5 (review 3) and S6a (fix round 3) into `keras-jax`

Date 2026-10-20, macOS arm64 (M2 Pro), CPU only. Merge driver. Nothing was pushed (`origin/keras-jax` stays at `826bc0a`);
no Polaris job; the `hepattn` main working tree was not touched. hepattn resolves to `hepattn-keras-jax/src` in both
envs (`.venv-jax`, `hepattn/.venv`).

## What was merged

Starting point `05122de` (merges `0f76770` of s5/train and `6bbd365` of s6/io, plus `05122de`, S5's patch for the 4
hybrid tests that asserted the `quant_losses()` double count).

| Commit | Content |
|---|---|
| `5876f42` | merge of `s5/train` at `21e29f7`: `b28dfb0` (optimizer guard compares identities, driver-flow beta retrace check), `964afda` (`learning_rate()` vs torch with `end != initial`, literal ramp values), `b4cf4cb` (M36-M39), `c7152db` (S5 log, review 3), `21e29f7` (pause note) |
| `a8b804f` | merge of `s6/io` at `d579d30`: `7e3ef89` (LightningCLI overlay merge), `c9e7ae0` (step-checkpoint retention = Lightning `save_top_k`), `a74d66b` (`ClicMetrics.update(event_valid)`), `b4200b6` (spawned workers re-import the main module), `8fd64bc` (M80-M95), `d579d30` (S6a log, round 3) |
| `fc920b1` | fix: the S6a config tests pinned the pre-S5 Polaris BetaPID setpoints (below) |
| `8d36490` | new cross-lane test `tests/keras_native/test_resume_cross_lane.py` |
| `86e14f3` | `checkpoint.py` docstring names Keras' counter `iteration` (was `iterations`) |
| (next commit) | these logs, `S56_merge_mutations.py`, S5.md resume note, port_log README rows |

Both merges were textually conflict-free (as the trial merges said).

## What the merge broke, and the fix

The full native suite at `a8b804f` had 2 failures, both S6a tests that S5's `polaris/03_make_config.py` change made stale:

```
FAILED tests/keras_native/test_config.py::test_loader_reads_every_form[polaris_generated]
FAILED tests/keras_native/test_config.py::test_raw_and_saved_forms_give_the_same_run_sections[polaris_generated-polaris_normalised]
2 failed, 740 passed, 4 skipped, 4786 warnings in 1381.83s (0:23:01)
```

The first pinned `BETA_PID["polaris"]` = target 1e15, init 2.08e-9, max 8e-7, min 0 (the old setpoints x 8); the
generated config now holds S5's provisional values x `HGQ_EBOPS_EVERY` = 8: target 1.5e11, init 6.96e-10, max 2.64e-7,
min 1.84e-12 (`BETA_SETPOINTS.md`). The second compared the freshly generated config with the saved fixture
`s6_fixtures/polaris_normalised.yaml`, which S6a had generated from the old `03_make_config.py`.

Fix (`fc920b1`): pin the new values (with `min_beta`, now > 0) and regenerate the fixtures with
`s6_fixtures/make_normalised_configs.py` (hybrid env). The regenerated fixtures differ from the committed ones in exactly
the 4 BetaPID lines of `polaris_normalised.yaml`; the other four fixtures are byte-identical. `test_config.py`: 49
passed. Control: the failing run above (old pins against the new generator).

## Cross-lane test: `TrainStep` state through a checkpoint into a fresh model

No test combined S5's `TrainStep` with S6a's checkpoint (`test_checkpoint.py` uses its own hand-written jitted step and
the default Keras optimizer). `tests/keras_native/test_resume_cross_lane.py`:

- Setup: org model (`build_mf("org", QUANT_SPECS["high"])`), `set_beta(1e-9)` with the EBOPs term on, `build_optimizer`
  AdamW with the OneCycle schedule (`skip_scheduler: False`, so the LR reads the counter), `clip_norm` 0.1 and
  accumulation k = 2; 2 CLIC dummy batches (`real_like`).
- Uninterrupted: `s0 -> s1 -> s2` with `TrainStep.train_step` (donated state, host copies kept). The checkpoint is `s1`,
  taken mid-accumulation, with `BetaPIDController.state_dict()` in the metadata.
- Resumed: an independently built model + optimizer + `TrainStep` (its own compile, other auto-generated layer names),
  `load_checkpoint`, one step on batch 1.

| Test | What it shows | Measured |
|---|---|---|
| `test_the_checkpoint_sits_inside_an_accumulation_and_the_next_step_updates` | non-vacuous: `s0 -> s1` leaves `tv` unchanged and fills the accumulators (one per trainable variable); `s1 -> s2` changes `tv`, `ntv` and `ov`; beta 1e-9 in `ntv` | pass |
| `test_resumed_step_in_a_fresh_model_equals_the_uninterrupted_step` | loaded state = `s1` bitwise; the resumed step = `s2` bitwise (every array, dtype, NaN-aware) | pass, 0 differing arrays |
| `test_control_one_perturbed_leaf_breaks_the_resume[weight]` | one kernel element moved by 1 ulp before the step | comparison fails (as required) |
| `test_control_one_perturbed_leaf_breaks_the_resume[accumulator]` | one nonzero accumulator x 1.5 | comparison fails |
| `test_beta_controller_resumes_from_the_checkpoint_metadata` | the PID restored from the JSON metadata gives the next beta exactly (d = 0.2, so `prev_error` matters); a fresh PID gives another beta | pass |

Cost: 4 min on CPU (140 s for the first model's build + compile, 91 s for the second `TrainStep`'s compile). Two
independently built and compiled org models give the same step bit for bit on CPU.

Mutations ([S56_merge_mutations.py](S56_merge_mutations.py), each in a fresh copy of `src` + `tests` under the session
scratchpad; the worktree was never mutated). Verbatim summaries:

```
XL0_none
  5 passed, 14 warnings in 236.75s (0:03:56)
XL1_accumulators_zeroed
  FAILED tests/keras_native/test_resume_cross_lane.py::test_resumed_step_in_a_fresh_model_equals_the_uninterrupted_step
  FAILED tests/keras_native/test_resume_cross_lane.py::test_control_one_perturbed_leaf_breaks_the_resume[accumulator]
  2 failed, 3 passed, 14 warnings in 222.58s (0:03:42)
XL2_beta_zeroed
  FAILED tests/keras_native/test_resume_cross_lane.py::test_resumed_step_in_a_fresh_model_equals_the_uninterrupted_step
  1 failed, 4 passed, 14 warnings in 230.32s (0:03:50)
XL3_iterations_plus_one
  FAILED tests/keras_native/test_resume_cross_lane.py::test_resumed_step_in_a_fresh_model_equals_the_uninterrupted_step
  1 failed, 4 passed, 14 warnings in 210.92s (0:03:30)
XL4_pid_prev_error_dropped
  FAILED tests/keras_native/test_resume_cross_lane.py::test_beta_controller_resumes_from_the_checkpoint_metadata
  1 failed, 4 passed, 14 warnings in 207.90s (0:03:27)
```

4 / 4 killed. XL1-XL3 change what `load_checkpoint` returns (accumulators zeroed, every `ntv` key ending in `beta`
zeroed, the optimizer counter + 1); XL4 drops `prev_error` from `BetaPIDController.load_state_dict`. XL3 first ran with
the key `iterations` and survived: a no-op, because Keras names the counter `iteration` (checked:
`[v.name for v in AdamW(...).variables]` starts with `'iteration'`); re-run with `iteration`, killed (above). The same
wrong name was in the `checkpoint.py` docstring; corrected. The XL0 run is with the final test (PID `d` = 0.2); the first
run of the test (d = 0) also passed (5 passed, 244 s).

## Test runs

| Suite | Env | HEAD | Result |
|---|---|---|---|
| `pytest tests/keras_native` (full native suite) | `.venv-jax`, `KERAS_BACKEND=jax` | `a8b804f` (merges only) | 2 failed, 740 passed, 4 skipped (23 min); the 2 failures above. Collected 746: the new test file did not exist yet |
| `tests/keras_native/test_config.py` | `.venv-jax` | `fc920b1` content | 49 passed (84 s) |
| `tests/keras_native/test_resume_cross_lane.py` | `.venv-jax` | `8d36490` content | 5 passed (237 s, XL0) |
| `pytest tests/keras_native` (full, 751 collected) | `.venv-jax` | `8d36490` | **747 passed, 4 skipped, 0 failed** (22 min 24 s). The later `checkpoint.py` change is docstring-only |
| `pytest -m 'not gpu and not requiresdata' tests/keras tests/polaris` | `hepattn/.venv`, `KERAS_BACKEND=torch` | `a8b804f` | **316 passed, 2 skipped, 6 deselected, 0 failed** (530 s): the 4 tests of S5 open issue 1 pass; no hybrid file changed after `a8b804f` |
| `pytest -m 'not gpu and not requiresdata' tests/models tests/loss` | `hepattn/.venv` | `a8b804f` | 519 passed, 173 skipped, 5 deselected (24 s) (R0-R5 quote 522 passed for `models + polaris + loss`; `tests/polaris` is in the hybrid row here) |
| `pytest -m 'not gpu and not requiresdata' tests/matching` + `R0_matching_gate.py --max-failures 106` | `hepattn/.venv` | `a8b804f` | 106 failed, 66 passed, 14 skipped, 17 deselected, 2 xfailed; gate `ran=172 failures=106 lap1015_attr=106 other=0`, exit 0 (same as R5) |
| `prek run --files` on every changed file | | | all hooks pass |

## Deviations

1. S5's merge requirement (open issue 1: the hybrid double-count patch must land in the same merge as `7f3747c`) was not
   met by the earlier merge: the patch is `05122de`, after `0f76770` and `6bbd365`, so those two commits carry 4 red
   hybrid tests. Recorded in S5.md open issue 1; nothing to rewrite (unpushed history could be squashed, but the
   instruction was to merge, not rewrite).
2. The first full native suite ran on the merge HEAD; the test file and the config fix came after it. The final full run at
   `8d36490` is in the table.

## Hand-offs resolved by the merge

- S6a open issue "S6b, optimizer builder: AdamW epsilon 1e-8": S5's `build_optimizer` uses `adam_epsilon=1e-8` by
  default. `optimizer_from_config` does not read `OptimizerConfig.epsilon` (no YAML key carries it; both are 1e-8).
- S5 open issue 4 (the native driver must checkpoint the PID state): `save_checkpoint` metadata key `beta_controller`
  carries `BetaPIDController.state_dict()`; the round trip is now tested (XL4 kills a lossy `load_state_dict`).
- S6a open issue "S5: `min_beta` 0.0": the generated Polaris config now has `min_beta` 1.84e-12 > 0. A raw config without
  `min_beta` still loads 0.0 (hybrid default), and `ebops_readout` raises for beta <= 0: S6b must refuse or clamp.

## Open issues (for S6b)

1. Two readers of the same YAML keys: `config.load_config` -> `RunConfig.optimizer` (S6a) and
   `optim.optimizer_from_config(model_section, trainer_section, ...)` (S5). The driver should build the optimizer from
   one of them; until then nothing checks that they agree (a test over the 6 config forms would).
2. The resume gate here uses dummy events and one optimizer configuration (AdamW, k = 2, OneCycle). Lion and k = 1 are
   covered only by `test_checkpoint.py`-style key tests, not by a `TrainStep` resume. The data position (`EpochSampler`
   start batch at an optimizer-step boundary) is S6b's, not tested here.
3. Bitwise resume is shown on CPU only; on an A100 XLA may pick other algorithms per compile (S6b / G3 should re-check
   resume equality there, with a tolerance if needed).
4. All open issues of S5.md and S6a.md that name S6b, G3 or S8 stay open (G1 is NOT resolved; G3 has no script yet).
5. (delta fix) `OneCycleLR` holds its value past `total_steps`, where Lightning + torch raise `ValueError` ("Tried to
   step 401 times") at scheduler step `total_steps + 1`. The hold stays (S5 deviation 5), so the driver must keep
   torch's failure signal: assert (or at least log loudly) when `optimizer.iterations` (optimizer steps, not
   micro-steps) exceeds `total_steps`, which catches a `total_steps` that leaves out the accumulation division.
6. (delta fix) On resume the driver passes the checkpoint it loads as `CheckpointManager(..., resume_path=)` /
   `from_callback(..., resume_path=)`, and gives each run its own checkpoint directory (pruning considers every file
   there).

## Delta fix (review of the S5 review-3 and S6a fix-round-3 commits)

Date 2026-10-20, same machine and envs. Every reviewer claim below was re-measured before the fix.

| Severity | Issue | Verified | Fix | Discrimination |
|---|---|---|---|---|
| major | The loader folded keys next to a `class_path` into `init_args`; jsonargparse 4.40.1 rejects that form | LightningCLI `--print_config` on base + linformer + overlay: top-level class, nested `Encoder`, callbacks item all rc 2 ("Key 'class_path' is not expected" / "Not a valid subclass of Callback") | `config._check_entries` raises `ConfigError` for a `class_path` dict with keys other than `init_args` / `dict_kwargs`, in any file. eval.yaml's form (a plain dict at a class an earlier file gave) still merges through `_merge_value`. S6a decision 13 (a), M80 text, the module and `merge_configs` docstrings and `overlay_partial_dicts.yaml`'s comment corrected | `test_short_class_entries_are_rejected_as_lightningcli_rejects_them[x3]` (recorded LightningCLI rc 2 as the control) and `test_explicit_class_entries_are_read_as_lightningcli_reads_them` (the `init_args` form: rc 0 in LightningCLI, loaded with `num_layers` 2 / `every_n_epochs` 2). D1 killed (8 tests) |
| major | An overlay `ModuleList` item without `class_path` replaced the list; LightningCLI merges it into the base item | LightningCLI rc 0; its `modules[0]` keeps `class_path: hepattn.models.InputNet` and the `FourierPositionEncoder` posenc (eta, phi) | `_check_list_overlay` raises when an overlay list replaces a list of class entries and any item is a dict without `class_path` (not emulated) | `test_overlay_list_item_without_class_path_is_rejected` (control: the printed item, with its class_path, is accepted) and a `merge_rejects` case. D2 killed |
| major | The torch decoder's `posenc` (`dict[str, float] \| None`, `src/hepattn/models/decoder.py:32`) was merged as a `FourierPositionEncoder` class | LightningCLI on P1 `{alpha: 1, base: 2}` + P2 `{alpha: 3}`: `{alpha: 3.0}` | `_DICT_TYPED["MaskFormerDecoder"] = {"posenc"}`; `_class_of(key, parent)` decides class-typed per owning class in the merge and in `model_description` (`_canonical` now passes the owning class) | `test_decoder_posenc_is_replaced_whole_as_lightningcli_does` (fixture value; description keeps a dict; P1-only hash differs) and `test_merge_rules` (InputNet posenc still merges). D3 killed (2 tests), D4 (`_canonical` without the owner) killed |
| minor | LR tolerances were 2x the error at the sampled steps only | Every step 0-400: max abs 1.153e-7 x max_lr (step 128, end 1e-6), 1.026e-7 (step 52, end 1e-8); max rel 1.424e-6 (step 374) | `test_learning_rate_maps_the_lrs_config_like_torch` samples every step 0-400; `LR_ATOL` 2.4e-7, `LR_RTOL` 2.9e-6 (2x the all-step maxima); S5.md table row and item 7 corrected | D7 (2e-6 relative at step 128 only): killed by the new test, survives the old one (`D7_..._oldtest`) |
| minor | The optimizer-guard test used auto-named layers, so a path guard passed it | Mutant R7 (`a.path != b.path`) passed the old test | Both models use explicit layer names `hid` / `out`; the test asserts equal paths and shapes as the control | D5 killed; survives the old test (`D5_..._oldtest`), as the reviewer measured |
| minor | Holding past `total_steps` hides a `total_steps` miscount that torch would raise for | re-measured: torch 2.10.0 raises "Tried to step 401 times" at scheduler step 401; the native schedule returns 1.001349e-06 at 400, 401 and 450 (the hold is `ops.minimum(n, total_steps)` in `OneCycleLR.__call__`) | Hand-off: open issue 5 below | docs only |
| minor | Step-checkpoint pruning removes every `step-*` file beyond k, the resume file included | `_step_files` globs the directory; Lightning guards `trainer.ckpt_path` (`_should_remove_checkpoint`) | `CheckpointManager(..., resume_path=)` (also `from_callback`): `_remove` never deletes it (step pruning and stale `last-*` cleanup); the one-run-per-directory rule is in the docstring and S6a decision 10 (e) | `test_pruning_never_removes_the_resume_checkpoint[step\|last x True\|False]` (False is the control: the same saves remove it). D6 killed (2 tests) |
| minor | `keep_last_k` > 1 is not a Lightning value without a monitor | `model_checkpoint.py:469-474` rejects it; the loader already does | Docstring: -1 / 0 / 1 are Lightning's, k > 1 is a native extension; the `[3-False]` case is marked so; S6a decision 10 (d) | docs only |

The LightningCLI probes are recorded by `s6_fixtures/make_normalised_configs.py` (hybrid env) in
`s6_fixtures/lightningcli_probes.yaml`: each probe's overlays, return code, error line and the printed part it is about.
Regenerating it left the 5 existing fixtures byte-identical, and the model hashes of every form are unchanged
(org `b4f17323e5bc`, eval `3723db9efa12`, partial `a6bfe6adb660`, pflow_hgq `e88127de42ca`, polaris normalised
`01526c395845`, measured before and after).

Mutants (`S56_delta_mutations.py`, scratch copies; log `S56_evidence/delta_mutations.log`): D0 (none) 17 passed;
D1-D7 7/7 killed; D5 and D7 run against the old test files (from `8de6720`) survive, which shows the old tests missed them.

Delta-fix test runs: `pytest tests/keras_native` (full, `.venv-jax`, the content of `e1c3476`): **759 passed, 4 skipped,
0 failed** (763 collected = 751 + 12 new; 22 min 55 s). The hybrid and torch suites were not re-run: no file under
`src/hepattn/{keras,models,experiments}` (other than the test fixtures) or `tests/{keras,polaris,models,loss,matching}` changed
since `8de6720`. The fixture generator ran in `hepattn/.venv` (lightning 2.5.2, jsonargparse 4.40.1). `prek run --files`
passes on every changed file.
