# S6a log: config loader, data adapter, checkpoints, metrics, prediction writer

Plan: `docs/hgq/KERAS_JAX_PLAN.md` sections 3.8, 3.9, 3.10 and the S6 gates of section 4 that do not need the train
loop (the loop is S6b). Branch `s6/io` from `keras-jax` `826bc0a`, worktree `hepattn-s6`. Date 2026-10-16, macOS arm64
CPU. Env `hepattn-keras-jax/.venv-jax` (jax 0.11.2, keras 3.15.0, torch 2.10.0 CPU, torchmetrics 1.8.2, ftag 0.2.18);
fixtures generated in the hybrid env `hepattn/.venv` (lightning 2.5.2).

## What landed

| File | Content |
|---|---|
| `src/hepattn/keras_native/config.py` | `load_config(paths)`: one YAML or several merged in order (dicts merge, lists and scalars replace); `normalise_model_section` maps jsonargparse forms (`activation: {class_path: torch.nn.SiLU, init_args: {inplace: false}}` -> `"SiLU"`, decoder `class_path` wrapper removed); relative `scale_dict_path` resolved against cwd, the config's directory and its parents, then the packaged `configs/<name>` (with a warning). Returns `RunConfig`: model section (input of `from_config`, keeps the `mtl` check), task specs, matcher threads, optimizer (AdamW / Lion, LR schedule numbers, clip, accumulation), quant (spec, BetaPID / BetaScheduler args, PeriodicEBOPs period), checkpoint policy, data (CLICDataset kwargs), the other trainer keys, `model_hash`. Raises for another optimizer, value clipping, both beta callbacks, a duplicate callback. |
| `src/hepattn/keras_native/checkpoint.py` | `save_checkpoint` / `load_checkpoint` of `(tv, ntv, ov)` + metadata (step, epoch, beta controller, data position, config hash) as `.npz` + JSON sidecar (sha256 of the `.npz`, per-array shape and dtype), atomic writes; keys independent of auto-generated layer names; `export_weights` (`.weights.h5`); `CheckpointPolicy` and `CheckpointManager` (every N steps, `last`, best on `val/loss`, keep-last-k). |
| `src/hepattn/keras_native/data.py` | Torch-free `HostBatches`: loader batches -> numpy -> int64/float64 to int32/float32 (range-checked) -> optional `jax.device_put`; invalid-event padding with `event_valid`; host keys (`event_number`, int64); one shape signature per iterator. |
| `src/hepattn/experiments/clic/native_data.py` | The torch side: `make_dataset` (`CLICDataset`), `EpochSampler` (permutation from `seed + epoch`, `start_batch` resume, `drop_last`), `make_loader`, `host_batches` (train drop_last, val/test padded). |
| `src/hepattn/keras_native/metrics.py` | `MulticlassAccuracy` (micro/macro), `BinaryRecall`, `BinaryPrecision` (torchmetrics 1.8 reductions in float32), `MaskInference` and task-metric ports, `WeightedMean` (Lightning epoch reduction), `ClicMetrics` (the names `MPflow.log_metrics` logs). |
| `src/hepattn/keras_native/predict_writer.py` | `predict(outputs, model.tasks)` in `keras.ops` (jit-able), `PredictionWriter` (the `PflowPredictionWriter` h5 layout with the same ftag `H5Writer` settings; padding events dropped), `output_path`. |
| `tests/keras_native/test_{config,checkpoint,data,data_real,metrics,predict_writer}.py` | 71 tests + 4 `requiresdata` tests (below). |
| `tests/keras_native/s6_fixtures/` | Three jsonargparse-normalised run configs and their generator `make_normalised_configs.py` (LightningCLI `--print_config`, hybrid env). |
| `docs/hgq/port_log/S6a_mutations.py` | Mutation runner (table below). |

## Decisions

1. **Data adapter split (torch rule).** `CLICDataset` imports torch and lightning (`pflow_data.py:4-9`), so the torch
   side lives in `experiments/clic/native_data.py`, outside `keras_native` and so outside the import-hygiene scope; no
   exemption was needed. `keras_native/data.py` converts with `np.asarray` (torch tensors' `__array__`), as keras's own
   torch-DataLoader adapter does. `test_import_hygiene.py` passes unchanged (static and runtime scans of the 5 new modules).
2. **Checkpoint keys.** `variable.path` is not stable. Measured on the production quantized org model (4223 variables):
   two builds in one process differ in 3134 paths (`.../fixed_point_quantizer_kif/f` vs `.../fixed_point_quantizer_kif_580/f`).
   Key = path with a trailing `_<digits>` removed from each segment; 420 names then repeat (QEinsum `multiple_quantizers/
   quantizer_<n>`), so a repeated name gets `#<n>` in `model.variables` order (construction order, the same per config).
   Keras names optimizer slots `<ref.path with / -> _>_<slot>` (`keras/src/optimizers/base_optimizer.py:311-313`), so a
   slot is keyed `<ref key>::<slot>` by longest-prefix match against the trainable paths. The S7 converter can reuse
   `variable_keys` (plan: "key quantizer variables by leaf name and position").
3. **Checkpoint integrity.** The sidecar is renamed last, so it marks completeness; load checks the sidecar's sha256 of
   the `.npz`, the zip CRCs (`np.load`), keys, shapes, dtypes, sidecar layout and metadata. `last` / `best` are pointer
   files replaced atomically, so a crash never tears them. A config change without a variable change (e.g.
   `input_sort_field`) loads by layout; only `expect_config_hash` catches it (tested).
4. **Padding = invalid events.** Every `*_valid` False, class 5 at `particle_class`, NaN in the per-object float
   targets (as `CLICDataset` writes invalid particles), 0 elsewhere, `event_number` -1. Measured: the native forward and
   the matched loss of an all-invalid event are finite (float and quantized), and the real events of a padded batch are
   bit-identical to an unpadded batch (max diff 0.0, every output, test width).
5. **Metrics per layer, not pooled.** `MPflow` updates one torchmetrics object per metric for every layer key
   (`lightning_module.py:20-24,45-53`), so Lightning logs the pooled value under each layer name. Measured with a
   minimal Lightning 2.5.2 validate run (scratch `lightning_shared_metric.py`): layer recalls 1.0 and 0.0 are both
   logged as 0.5. `ClicMetrics` keeps one object per layer (the intended value); `test_lightning_module_pools_layers_and_native_does_not`
   pins the difference. Comparisons with hybrid runs must use the final layer or pooled values.
6. **Saved-config fixtures** are generated, not hand-written: `--print_config` is what `SaveConfig` writes. The Polaris
   one is `main_hgq fit --config <03_make_config output>`; its absolute paths are rewritten to `/polaris/home/hepattn`, so
   loading it exercises the packaged scale-dict fallback.

## Gates

Test runs (macOS CPU): `test_config` + `test_data` + `test_data_real` + `test_metrics` + `test_predict_writer` +
`test_import_hygiene`: 74 passed, 4 skipped (requiresdata) in 159 s; `test_checkpoint`: 16 passed in 282 s (one jitted
train step compile at test width: 190-200 s of it). Whole native suite after the last commit: 614 passed, 4 skipped
(requiresdata) in 1140 s (`HEPATTN_PARITY_LEDGER=0`).

Re-verified 2026-10-16 in a new session at `a2bab07`: the 7 S6a test files + `test_import_hygiene.py` gave 92 passed,
4 skipped (requiresdata) in 360 s; `prek run` on all 18 changed files passes; `S6a_mutations.py --only
M1,M3,M13,M14,M19,M22,M24,M25` killed 8/8, and the tree was clean afterwards.

| Gate | Result | Evidence and control |
|---|---|---|
| Loader reads every form | pass | `test_loader_reads_every_form` x6: org raw (base + linformer), org saved, `pflow_hgq.yaml`, its saved form, the generated Polaris config (run in the test), its saved form. Each gives `clic_specs()`, 16 matcher threads, the expected optimizer / quant / accumulation / PeriodicEBOPs period. Controls: `test_unsupported_run_options_raise` x5 (each also loads the unedited config first), activation / decoder rejection tests. |
| Same native model from each form | pass | `test_raw_and_saved_forms_build_the_same_model` x3 at production dims: same keys, shapes, dtypes (335 / 3967 / 4223 variables: float org, quantized `pflow_hgq`, quantized Polaris), outputs bit-identical with copied values. `test_generated_polaris_config_is_the_org_model_with_the_quant_spec`: org raw + the Polaris quant block == generated config (4223). Controls: no input sorting -> same keys, forward differs; `linformer_k` 32 -> shapes differ; `from_config` of the saved config without the loader raises `TypeError: activation must be a name` (the R-critique open issue, closed here by the loader). |
| Checkpoint save -> load bitwise | pass | `test_save_load_is_bitwise_into_a_fresh_model`: all `tv`, `ntv`, `ov` arrays equal in value and dtype after loading into an independently built model + optimizer. Non-vacuous: `test_steps_change_every_state_group` (each step changes all three groups). |
| Resume = uninterrupted (plan S6 gate, CPU) | pass | `test_resume_equals_the_uninterrupted_run`: step 2 from the loaded step-1 state equals the uninterrupted step 2 bitwise; the fresh model with the loaded state gives the first model's forward. Control: +1e-3 on one kernel element of the loaded state -> differs. The resumed step uses the first model's compiled step (a second compile costs 145 s, measured); a new process compiles anyway. |
| Keys stable across builds | pass | `test_keys_are_stable_across_independent_builds`; control in the same test: raw paths and optimizer names differ (> 100 each). `test_optimizer_slots_are_keyed_by_their_variable`: 2 slots per trainable variable, all mapped. |
| Corrupted / partial rejected | pass | flipped byte (sha256), truncated file with a rewritten sidecar hash (zip error), flipped byte inside an array member with a rewritten hash (CRC), foreign sidecar, missing sidecar, config-hash mismatch, a float model's keys; interrupted write (savez raises mid-file) leaves the previous checkpoint loadable and no temporary file. |
| Export | pass | `test_export_weights_loads_into_a_fresh_model`: `.weights.h5` loads bit-exactly; control: the fresh model differs before loading. |
| Metrics == torchmetrics | pass | Streaming micro/macro accuracy over 3 batches, BinaryRecall / BinaryPrecision (incl. the zero-division 0), macro with absent classes (0.75), `MaskInference`, and the whole `MPflow.log_metrics` dict for one layer (31 names). Count-based values exactly equal; float32 means up to 3 ulps and macro 1 ulp different (measured over 10 seeds, summation order); tolerance 2x (6 / 2 ulps). Controls: one perturbed class, mask bit, regression value, proxy value each change their metric by > 100x the tolerance, and then match torch again. |
| Prediction h5 == `PflowPredictionWriter` | pass | `test_h5_layout_and_values_equal_the_torch_writer`: same groups, field names, dtypes, shapes, chunks, compression, attributes; values equal (inverse transforms exact, measured 0.0); `load_convert_h5` gives equal arrays. `predict` == torch task `predict` (mask probabilities within 2x the measured 1.19e-7, 0 threshold flips). Controls: a renamed field (`pred_e` -> `pred_E`) changes the layout; a changed class changes exactly `object_class/pflow_class`; padding events are not written. |
| Data adapter | pass | `test_data.py` (9): train drop_last (10 events / batch 4 -> 2), static int32 / float32 / bool device arrays; val padded (`event_valid` [T, T, F, F], invalid-event values, real events unchanged, `event_number` int64 [8, 9, -1, -1]); one val epoch compiles the jitted eval forward + `predict` once. Controls: the torch batch has int64 and a short last batch; `pad_to_batch=False` raises on it; the unpadded short batch compiles a second executable; an int64 above int32 raises; a new shape raises. |
| Real CLIC batches (requiresdata) | **deferred to Polaris/falcon** | `test_data_real.py`: no target at padded nodes, prefix-contiguous `particle_valid` with class 5 exactly at invalid particles, static batches; records the equal-phi rate (RETARGET risk 6). Not runnable here (no ROOT files). Dry run on dummy events: the padded-node check fails on raw dummy events (targets at padded nodes) and passes after cleaning; the prefix check fails with a residual first (3 of 10 events). |
| S6 gate: train `_cache_size() <= 2` over an epoch | **S6b** | Needs the train loop. Eval + predict part done (above). |

## Mutations

`S6a_mutations.py`: 28 mutants of the committed sources, each run against its test file with `-x` and restored from memory
(`git diff --quiet` afterwards). **28/28 killed** (first failing test shown). Two of the first 28 survived and were
fixed: the old M23 (`nan_to_num` on regression targets before `[valid]`) was an equivalent mutant and was replaced;
M24 (`valid = classes <= num_classes`) survived because the untrained test model never predicts the null class, so
`test_predict_with_null_class_queries_equals_torch` was added (`6791e87`).

| Mutant | Change | Killed by |
|---|---|---|
| M1 | saved-config activation dicts not normalised | `test_raw_and_saved_forms_build_the_same_model[org]` |
| M2 | decoder `class_path` not unwrapped | same |
| M3 | YAML merge replaces dicts (overlay drops base keys) | `test_loader_reads_every_form[org_raw]` |
| M4 | no packaged scale-dict fallback | `test_loader_reads_every_form[org_normalised]` |
| M5 | PeriodicEBOPs period read from a wrong key | `test_loader_reads_every_form[polaris_generated]` |
| M6 | checkpoint keys = raw `variable.path` | `test_save_load_is_bitwise_into_a_fresh_model` |
| M7 | no `#n` for repeated names | same (fixture error: colliding npz keys) |
| M8 | sidecar sha256 not checked | `test_corrupted_npz_is_rejected` |
| M9 | optimizer slots keyed by raw name | `test_save_load_is_bitwise_into_a_fresh_model` |
| M10 | `expect_config_hash` ignored | `test_config_hash_mismatch_is_rejected` |
| M11 | `.npz` written in place, not atomically | `test_interrupted_write_keeps_the_previous_checkpoint` |
| M12 | step 0 counts as an every-N step | `test_policy_every_n_steps_and_best` |
| M13 | int64 not cast | `test_int64_out_of_int32_range_raises` (the device test alone misses it: `jax.device_put` casts int64 to int32 with x64 off) |
| M14 | padding class 0 instead of the null class | `test_val_batches_are_padded_with_invalid_events` |
| M15 | padding put in front of the real events | same |
| M16 | `event_valid` all True | same |
| M17 | train split without shuffle / drop_last | `test_train_batches_drop_last_and_are_static_int32` |
| M18 | order ignores the epoch | `test_train_order_depends_on_seed_and_epoch_and_resumes` |
| M19 | macro averages over classes without support | `test_macro_skips_classes_without_support_like_torchmetrics` |
| M20 | recall uses fp | `test_binary_recall_precision_equal_torchmetrics[BinaryRecall]` |
| M21 | mask efficiency divides by the predicted count | `test_mask_inference_equals_torch` |
| M22 | truth validity from `particle_valid` OR True | `test_clic_metrics_equal_the_lightning_module` |
| M23 | `abs_norm_res` without `abs` of the target | same |
| M24 | `valid = classes <= num_classes` | `test_predict_with_null_class_queries_equals_torch` |
| M25 | field `truth_masks` renamed | `test_h5_layout_and_values_equal_the_torch_writer` |
| M26 | regression not inverse-transformed | same |
| M27 | padding events written | `test_padding_events_are_not_written` |
| M28 | mask logits shifted by 0.01 before the sigmoid | `test_predict_equals_torch_task_predict` |

Checkpoint mutants take 200-270 s each (the train-step compile); the others 6-25 s.

## Deviations

1. New files outside the listed ones: `src/hepattn/experiments/clic/native_data.py` (the torch half of the adapter, as
   RULES require), `tests/keras_native/s6_fixtures/` (fixtures + generator), `docs/hgq/port_log/S6a_mutations.py`.
2. Plan 3.9 says "keyed by `variable.path`"; the keys strip auto-generated suffixes (decision 2).
3. `ClicMetrics` reports per-layer values; the Lightning module pools layers (decision 5).
4. The resume gate steps with the first model's compiled function (cost); the fresh model is checked by an eager forward.
5. The `on_test_end` ROOT file of the torch writer is not written (only the h5); `load_convert_h5` reads the native h5.

## Open issues

- **S6b, validation loss with padded events.** The objective has no per-event weights: a padded event adds 150
  null-target queries to the classification CE mean (and zeros elsewhere). Measured on two dummy events with one of them
  replaced by an invalid event: total 49.80 vs 50.21 (float), 72.57 vs 78.18 (quantized). Torch evaluates the short last
  batch unpadded. S6b: run the last batch's loss with per-event terms, or a second executable at its size, and weight
  `val/loss` by real events (Lightning's epoch mean weights by batch size).
- **S6b:** metrics and the writer need `predict` of the *matched* (permuted) outputs, as torch's `validation_step` /
  `test_step` (after `model.loss`); `permute_outputs` with `matched_loss`'s `idx`.
- **S6b:** pass `expect_config_hash=rc.model_hash` to `load_checkpoint`; resume data position at an optimizer-step
  boundary with accumulation (`EpochSampler(start_batch)`); call `CheckpointManager.save` at the end of an epoch /
  before walltime (`last` points at the latest step checkpoint).
- **S6b / S8:** the generated Polaris config defaults to `precision: bf16-mixed`; the user decided fp32 first, so the
  driver must ignore it or the config be generated with `HGQ_PRECISION=32`. `RunConfig.trainer["precision"]` carries it.
- **S5:** the saved Polaris config's BetaPID has `min_beta: 0.0` (the default); plan 3.6's EBOPs read-out needs beta > 0.
- **Polaris:** run `CLIC_DATA_DIR=/eagle/hgcal-maskformer-fpga/clic_data CLIC_DATA_REQUIRED=1 python -m pytest -m requiresdata tests/keras_native/test_data_real.py -s`.
  Risk found by reading: `pflow_data.py:461-462` sets `particle_valid` only for non-residual particles among the first
  `n_particles` (pdgid -999 -> class 5, `pflow_data.py:687`), so a residual before a real particle breaks the prefix
  that both torch's `Matcher` (`matcher.py:317-320`, lengths = valid count) and native `host_match` assume.
- `CLICDataset.__init__` calls `seed_everything(42)` (`pflow_data.py:49`): the driver must seed after building datasets.
- Not owned: add an S6a row to `docs/hgq/port_log/README.md` and close the "S4 -> S6 real CLIC batches" row there when
  the requiresdata run has passed; plan status table.
