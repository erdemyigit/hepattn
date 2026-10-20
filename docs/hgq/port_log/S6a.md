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
| `tests/keras_native/test_{config,checkpoint,data,data_real,metrics,predict_writer}.py` | First round 71 tests, fix round 1: 107, fix round 2: see Gates (plus 4 `requiresdata` tests). |
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
   `input_sort_field`) loads by layout; only `expect_config_hash` catches it (tested). Fix round: the directory is
   fsynced after every rename (a rename is durable only then); `save_checkpoint` refuses to replace an existing
   checkpoint unless `overwrite=True` (between the two renames the new `.npz` sits next to the old sidecar, rejected
   on load); `CheckpointManager` never overwrites: a second save at one step (e.g. a walltime save at the resumed
   step) becomes `step-<N>-r<k>.npz`, and retention orders by (step, revision).
4. **Padding = invalid events.** Every `*_valid` False, class 5 at `particle_class`, NaN in the per-object float
   targets (as `CLICDataset` writes invalid particles), 0 elsewhere, `event_number` -1. Measured: the native forward and
   the matched loss of an all-invalid event are finite (float and quantized), and the real events of a padded batch are
   bit-identical to an unpadded batch (max diff 0.0, every output, test width).
5. **Metrics per layer, not pooled.** `MPflow` updates one torchmetrics object per metric for every layer key
   (`lightning_module.py:20-24,45-53`), so Lightning logs the pooled value under each layer name. Measured with a
   minimal Lightning 2.5.2 validate run (scratch `lightning_shared_metric.py`): layer recalls 1.0 and 0.0 are both
   logged as 0.5. `ClicMetrics` keeps one object per layer (the intended value); `test_lightning_module_pools_layers_and_native_does_not`
   pins the difference. Lightning logs the pooled value under every layer name, `final` included, so the native
   per-layer `final` value is not the hybrid's `val/final_*`: compare with hybrid logs only through
   `ClicMetrics(pool_layers=True)` (fix round 1), which equals a real `Trainer.validate` with two layers. `ClicMetrics("train")` logs only the task metrics, as `MPflow.log_custom_metrics` (returns early for
   train); Lightning logs train metrics per step, so the driver calls `reset` before each logged step.
6. **Saved-config fixtures** are generated, not hand-written: `--print_config` is what `SaveConfig` writes. The Polaris
   one is `main_hgq fit --config <03_make_config output>`; its absolute paths are rewritten to `/polaris/home/hepattn`, so
   loading it exercises the packaged scale-dict fallback.
7. **Model hash = hash of `model_description`** (fix round). The first `model_hash` hashed the normalised `model:`
   section as text. Measured: each raw/saved pair that builds a bit-identical model hashed differently (org
   a86f7b9cc3a2 vs dc6f9c7ad999, pflow_hgq a3b908209665 vs 3b29d1f8062e, Polaris 706dbef2cf7c vs 5feab1feaacd),
   because the saved form spells out defaults, writes `2.0` for `2`, other class-path spellings
   (`hepattn.models.posenc.FourierPositionEncoder` vs `hepattn.models.FourierPositionEncoder`, `torch.nn.SiLU`
   vs `SiLU`, `KerasMaskFormer`), and because the hash covered the resolved absolute scale-dict path, `lrs_config`
   and `optimizer`. `model_description` keeps `model.model` only, reduces class paths to class names (with the
   hybrid aliases), drops every entry equal to the native constructor default (`inspect.signature`; the encoder also
   takes `EncoderLayer` defaults, as it forwards its kwargs), turns numbers into floats, replaces each
   `scale_dict_path` by the sha256 of the file, and drops the matcher and the encoder's repeated `dim`. Native and
   torch defaults agree for every expanded key of the three saved fixtures (otherwise the pair would hash
   differently, as it then builds a different model). All three pairs now hash equal (re-measured: old hashes
   above reproduced; new org b4f17323e5bc, pflow_hgq e88127de42ca, Polaris 01526c395845, each for both forms); a byte-identical scale-dict
   copy elsewhere hashes equal; other scale-dict content, `input_sort_field`, `linformer_k`, an activation, or a
   non-default `mask_attn` change it. This also closes the silent-fallback concern: a saved config that falls back to
   a packaged scale dict with other content now gets another hash.
8. **Beta-controller arguments completed** (fix round). Raw configs omit BetaPID defaults, saved ones spell them out;
   the loader fills `p, i, d, warmup_epochs, max_beta, min_beta, damp_beta_on_target` (and BetaScheduler's three)
   from one table, casts to the hybrid types (`"1.0e15"` as a string -> 1e15, as the hybrid does), and rejects
   unknown or missing required arguments. A test reads the hybrid constructor defaults from
   `hepattn/keras/callbacks.py` with `ast` (importing it needs lightning and the torch backend) and compares.

9. **DataLoader workers are spawned, never forked** (fix round 2). With `num_workers > 0`, `make_loader` passes
   `multiprocessing_context="spawn"` and `persistent_workers=True`, and refuses `fork`. Reason: once JAX has a backend
   the process is multithreaded and JAX's at-fork hook warns that fork "will likely lead to a deadlock"
   (`jax/_src/xla_bridge.py` `_at_fork`, registered at backend init, lines 799-801 of the installed jax); Python 3.12
   forks by default on Linux (Polaris, `num_workers` 8) and spawns on macOS
   (https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods). `SplitBatches` holds one
   loader per split for the run and moves `EpochSampler` with `set_epoch(epoch, start_batch)`; the sampler is iterated
   in the main process, so persistent workers serve every epoch and a resume. Measured locally (dummy `CLICDataset`,
   40 events, 2 workers, after JAX init): first batch 4.04 s in epoch 0 (worker start), 0.01 s in epoch 1; 0.00 s
   without workers. Spawned workers receive the dataset pickled; the real `CLICDataset` holds all events in memory
   (`load_data`, `pflow_data.py:89-100`), so each of 8 workers gets a copy: measure on Polaris (open issue).
   `forkserver` would also be safe; spawn was chosen because it is the same on macOS and Linux.
10. **Checkpoint policy = what the callbacks do** (fix round 2). The first round read `save_last: true` from
   `base.yaml:61` / `pflow_hgq.yaml:62` and modelled "best on `val/loss`" (plan 3.9) as a single kept file. But
   `hepattn.callbacks.Checkpoint` (`src/hepattn/callbacks/checkpoint.py:10,15`) passes `save_top_k=-1` (Lightning
   2.5.2 `model_checkpoint.py:93`: all are saved) and its `setup` sets `save_last = False` and, for fit,
   `dirpath = <log_dir>/ckpts`; with no trigger set Lightning uses `every_n_epochs=1` (`__init_triggers`,
   lines 494-507), saving at each train epoch end, after that epoch's validation (fix round 3: Lightning saves in
   `on_train_epoch_end` because every CLIC config leaves `val_check_interval` at 1.0, `_should_save_on_train_epoch_end`,
   `model_checkpoint.py:324-339,429-447`; `on_validation_end` then skips the save. The two are the same while each epoch
   ends with a validation; an epoch stopped by `max_steps` runs no validation and Lightning saves it with the last
   logged `val/loss`). `CheckpointConfig` now has two `CheckpointCallback`s with the
   effective values: `epoch` (hepattn `Checkpoint`: monitor `val/loss`, min, top-k -1, save_last False, every epoch,
   dirpath None) and `step` (Polaris `ModelCheckpoint`: every 1000 steps, top-k 1 without a monitor, save_last True,
   `CKPT_DIR`). `CheckpointManager.save_epoch` keeps every epoch checkpoint (`epoch-<E>-step-<N>.npz`) and moves the
   `best` pointer without deleting (a NaN epoch is saved, never best; NaN is stored as `"nan"` so the metadata check
   passes). `save_last` writes `last-<N>.npz` at an epoch end unless `last` already holds that step (Lightning's
   `_should_skip_saving_checkpoint`), and stale `last-*` files are removed. `maybe_save_best` (top-1 with a monitor)
   stays for other configs and never removes an epoch checkpoint. The loader rejects `train_time_interval`,
   `save_on_train_epoch_end`, both triggers at once and top-k > 1 without a monitor (Lightning rejects the last two).
   Fix round 3: (a) `keep_last_k` is Lightning's `save_top_k` without a monitor (`model_checkpoint.py:92-93`): -1 keeps
   every step checkpoint (it kept only the newest before: the documented wiring `keep_last_k = save_top_k` would have
   deleted them), 0 writes no step file (`last-<N>.npz` only with `write_last`, as `save_top_k=0, save_last=True`),
   other values below -1 raise; `CheckpointManager.from_callback(dir, rc.checkpoint.step|epoch)` builds the manager
   (step trigger: `maybe_save_step`; epoch trigger: `maybe_save_epoch` with `every_n_epochs`, `(epoch + 1) % n == 0`
   as `model_checkpoint.py:328`) and raises for top-k by a monitor, which it does not implement. (b) A monitor missing
   from the metrics raises `KeyError` in `save_epoch` / `maybe_save_best` (Lightning raises once validation has run,
   `model_checkpoint.py:378-386`, and only warns before the first validation: the driver must not call `save_epoch`
   before one). (c) Deviation kept: Lightning ranks a NaN monitor value as +inf (min mode) and keeps it in
   `best_k_models`, so a run whose validations so far are all NaN has a NaN epoch as `best_model_path`; the native
   `best` pointer is then unset (a NaN is never best). From the first finite value on they agree (strict `<`).
11. **Checkpoint file mode** (fix round 2). `mkstemp` creates 0o600 and the rename keeps it; files are now chmod-ed to
   `0o666 & ~umask` before the rename, as a plain `open(..., "w")`, so collaborators can read checkpoints in the shared
   project area.
12. **Optimizer settings the torch wrapper implies** (fix round 2; `src/hepattn/models/wrapper.py:141-149`): the name
   is matched case-insensitively and stored canonically (`AdamW` / `Lion`); a missing `lrs_config.weight_decay` raises
   `ConfigError` (torch raises `KeyError`; the loader used to default to 0.0); `OptimizerConfig.epsilon = 1e-8` is
   torch `AdamW`'s default, which the S6b builder must pass to keras `AdamW` (default 1e-7,
   `keras/src/optimizers/adamw.py:59`). Lion: lion_pytorch betas (0.9, 0.99) are the keras Lion defaults.

13. **Config merge = LightningCLI** (fix round 3). The first loader merged every dict key by key (as
   `polaris/03_make_config.py:129-137` does). jsonargparse 4.40.1 (hybrid env) differs in two ways, measured with
   probes and with LightningCLI `--print_config`: (a) keys next to a `class_path` are init args (eval.yaml's
   `model: model: encoder: attn_type: torch`); the old loader kept them as a stray key and ignored it, so
   base + linformer + eval loaded the linformer model (hash `b4f17323e5bc`, the base + linformer hash; LightningCLI:
   `attn_type: torch`, hash `3723db9efa12`); (b) a dict-typed argument (`lrs_config`, `attn_kwargs`,
   `decoder_layer_config`, `data.inputs`, the hybrid `KerasMaskFormer`'s `encoder` / `decoder`) is replaced whole,
   while a class-typed one (a `class_path` entry, or the torch `MaskFormer`'s plain-dict `decoder`) merges its init
   args. Also measured: `init_args` next to other keys is rejected, a `class_path` change keeps only the init args the
   new class accepts (needs the class: the loader raises), `null` replaces. `merge_configs` implements this; the
   class-typed plain-dict keys come from `_CLASS_OF_KEY` (encoder, decoder, posenc, net, ...) minus `_DICT_TYPED`
   (`KerasMaskFormer`: encoder, decoder). Pinned by two new LightningCLI fixtures (`org_eval_normalised.yaml`,
   `org_partial_normalised.yaml` from `overlay_partial_dicts.yaml`). Old loader on the two overlays: both hashed
   `b4f17323e5bc` (overlay ignored); new: equal to the printed configs (`3723db9efa12`, `a6bfe6adb660`). The
   base + linformer hash and every other fixture hash are unchanged (linformer.yaml repeats its dicts whole).
14. **`ClicMetrics.update(..., event_valid=...)`** (fix round 3) drops the padding events of a padded val / test
   batch, as `PredictionWriter.write_batch` does. Without it, padding events (null class everywhere) change 3 of 31
   values on `fixed_case(4)` + 2 padding events (micro accuracy 0.636 -> 0.781). The per-layer default
   (`pool_layers=False`, decision 5) is kept: see open issues (user to confirm).

## Gates

Test runs (macOS CPU): `test_config` + `test_data` + `test_data_real` + `test_metrics` + `test_predict_writer` +
`test_import_hygiene`: 74 passed, 4 skipped (requiresdata) in 159 s; `test_checkpoint`: 16 passed in 282 s (one jitted
train step compile at test width: 190-200 s of it). Whole native suite after the last commit: 614 passed, 4 skipped
(requiresdata) in 1140 s (`HEPATTN_PARITY_LEDGER=0`).

Fix round 2026-10-16 (`af0da5f`..`d063c99`): the 7 S6a test files + `test_import_hygiene.py` at `d063c99` gave 111
passed, 4 skipped (requiresdata) in 293 s (hepattn resolved to this worktree's `src`); `prek run` on the 8 changed files
passes; mutants M29-M56 killed 28/28, the earlier checkpoint mutants M6, M8, M10, M11, M12 re-killed 5/5 (M11 still
caught by the interrupted-write test, which now passes `overwrite=True`); tree clean afterwards.

Fix round 2 2026-10-16 (`5db421e`..`5020a6d`): the 7 S6a test files + `test_import_hygiene.py` gave 131 passed,
4 skipped (requiresdata) in 400 s (hepattn resolved to this worktree's `src`), before
`test_hepattn_checkpoint_settings_overridden_by_its_setup_are_ignored` was added; at `5020a6d` (all code of the round):
132 passed, 4 skipped in 399 s. `prek run` passes on all changed files. Mutants M57-M79 killed 23/23; the config
mutants M29-M38 (M30, M31, M32, M36, M37 retargeted to the new code) and M12, M52, M53, M55 re-run: 14/14 killed.
Tree clean after each run (the runner's `git diff --quiet`).

New gate evidence (fix round 2):
- workers: `test_workers_are_never_forked` (spawn, persistent; control: a `fork` context raises);
  `test_spawned_workers_give_the_in_process_batches_without_a_fork_warning` (after `jax.device_put`: epochs 0, 1 and a
  resume at epoch 1 batch 1 through one 2-worker loader equal `num_workers=0`, epochs differ, no warning mentions
  fork); `test_control_forking_after_jax_init_warns` (a fork-context worker after JAX init records JAX's
  "JAX is multithreaded" warning, so the previous test can see one);
- `test_a_new_batch_shape_or_dtype_raises` x3 (shape; same shape with int32; with bool; control: an equal batch passes);
- checkpoint: `test_checkpoint_of_another_optimizer_is_rejected` x2 (AdamW <-> Lion, `ov: keys differ`; control: the
  same class loads); `test_stored_shape_other_than_the_models_is_rejected` (control: the rewrite alone loads);
  `test_checkpoint_files_get_the_mode_of_a_plain_write` x3 umasks (control: `mkstemp` gives 0o600);
  `test_nan_at_the_first_validation_is_not_the_best` (NaN and inf with no best yet, then 1e6 is accepted);
  `test_manager_keeps_every_epoch_checkpoint_and_points_at_the_best` (losses 4, NaN, 3, 3.5: 4 files, best = epoch 2,
  the NaN epoch loads, a reopened manager restores 3.0, a later top-1 best removes no epoch file);
  `test_manager_save_last_follows_validation_ends`;
- config: `test_loader_reads_every_form` x6 now pins `CheckpointConfig(epoch=..., step=...)` with the effective values
  (decision 10); `test_hepattn_checkpoint_settings_overridden_by_its_setup_are_ignored`;
  `test_optimizer_name_is_matched_case_insensitively`; `test_unsupported_run_options_raise` x9 (+ missing
  weight_decay, both triggers, top-k 3 without a monitor, `save_on_train_epoch_end`);
- metrics: `ulps` at a zero reference uses the smallest subnormal as the spacing (`test_ulps_of_an_exact_zero_reference_is_strict`).

Fix round 3 2026-10-19 (`7e3ef89`..`8fd64bc`): the 7 S6a test files + `test_import_hygiene.py` at `8fd64bc` gave
159 passed, 4 skipped (requiresdata) in 352 s (hepattn resolved to this worktree's `src`); `prek run` passes on every
changed file. Mutants M80-M95 and the retargeted M3 killed 17/17, and M12, M13, M14, M21, M55, M64, M75-M79 re-run
11/11 (28/28 in one run on the committed tree; the runner's `git diff --quiet` held). Reviewer issues and evidence:
- **major, config merge** (decision 13): `test_overlays_merge_as_lightningcli[eval|partial]` (raw base + linformer +
  overlay == the LightningCLI-printed config: same `model_description` and hash, which differs from base + linformer;
  pinned values: eval `attn_type` torch, no checkpoint callbacks, `is_inference`; partial `attn_kwargs` {num_heads: 8},
  2 decoder layers, base's `mask_attention` kept, `num_val` 10). Control: the key-by-key merge leaves the overlay's keys
  next to `model.model`'s `init_args`, which the loader now rejects. `test_dict_typed_arguments_are_replaced_whole`
  (the explicit-`init_args` form of the overlay gives the printed hash; control: merging its dicts keeps
  `linformer_k` / `norm` and gives another hash). `test_overlay_with_replaced_dicts_builds_the_printed_model`
  (bit-identical forward). `test_merge_rules`, `test_merge_rejects_...` x3, `test_partial_lrs_config_is_replaced_whole`
  (control: the full `lrs_config` loads).
- **major, step-checkpoint retention** (decision 10 (a)): `test_keep_last_k_is_lightnings_save_top_k` x5 (-1: all 4
  files; 0: none; 0 + `write_last`: `last-000004000.npz` only; 1; 3), every kept file loads;
  `test_keep_last_k_is_validated` x4 (control: -1 is accepted); `test_manager_from_the_polaris_step_callback`,
  `test_manager_from_the_hepattn_epoch_callback` (every_n_epochs 2 saves after epochs 1, 3; missing monitor raises),
  `test_manager_from_a_callback_it_does_not_implement_raises` x2 (control: top-k -1 with a monitor builds).
  Reproduced the reviewer's probe on the old code by reading: `steps[: max(0, len(steps) - keep_last_k)]` with -1
  slices every older file.
- **minors**: abs_norm_res epsilon placement (`test_abs_norm_res_adds_the_epsilon_after_the_division`: |t| 1e-5..3e-5
  at three particles per field; measured <= 4 ulps over 10 seeds, tolerance 8; the other placement is > 800 ulps away);
  `event_number` not in the device targets / inputs; `ClicMetrics.update(event_valid=...)` (decision 14;
  `test_event_valid_drops_padding_events`, control: without it 3 of 31 values change); `MaskInference` tolerance now
  measured (2 ulps worst over 20 seeds x 3 events, tolerance 4, 5 seeds x 3 events tested; was `rel=1e-6`);
  NaN / missing monitor (decision 10 (b), (c), deviation 7); "validation end" wording corrected to the train epoch end
  (decision 10, `config.CheckpointCallback`, `checkpoint` docstrings); the `_cache_size` gate row reworded; spawn guard
  in the `native_data` docstring and open issues. The reviewer's last minor (stale implementer report) needs no repo
  change.

Re-verified 2026-10-16 in a new session at `a2bab07`: the 7 S6a test files + `test_import_hygiene.py` gave 92 passed,
4 skipped (requiresdata) in 360 s; `prek run` on all 18 changed files passes; `S6a_mutations.py --only
M1,M3,M13,M14,M19,M22,M24,M25` killed 8/8, and the tree was clean afterwards.

| Gate | Result | Evidence and control |
|---|---|---|
| Loader reads every form | pass | `test_loader_reads_every_form` x6: org raw (base + linformer), org saved, `pflow_hgq.yaml`, its saved form, the generated Polaris config (run in the test), its saved form. Each gives `clic_specs()`, 16 matcher threads, the expected optimizer / quant / accumulation / PeriodicEBOPs period; fix round: the whole `OptimizerConfig` (org Lion 1e-6 / 8e-5 / 1e-6, pflow_hgq and Polaris AdamW 1e-6 / 5e-5 / 1e-6, pct_start 0.05, wd 1e-4, clip 0.1, accumulation 1 / 1 / 4), the whole `CheckpointConfig` (every_n_train_steps None / None / 1000, `val/loss`, min, save_last, dirpath), batch size / workers / num_* (512 / 128 / 32; 16 / 16 / 8; -1, 5000, -1), seed 42, the completed BetaPID dict; `test_raw_and_saved_forms_give_the_same_run_sections` x3 (raw == saved except the host dirpath and the CLICDataset defaults the saved form adds); `test_beta_callback_defaults_are_the_hybrid_ones` x2. Controls: `test_unsupported_run_options_raise` x5 (each also loads the unedited config first), activation / decoder rejection tests. |
| Same native model from each form | pass | `test_raw_and_saved_forms_build_the_same_model` x3 at production dims: same keys, shapes, dtypes (335 / 3967 / 4223 variables: float org, quantized `pflow_hgq`, quantized Polaris), outputs bit-identical with copied values. `test_generated_polaris_config_is_the_org_model_with_the_quant_spec`: org raw + the Polaris quant block == generated config (4223). Controls: no input sorting -> same keys, forward differs; `linformer_k` 32 -> shapes differ; `from_config` of the saved config without the loader raises `TypeError: activation must be a name` (the R-critique open issue, closed here by the loader). |
| Checkpoint save -> load bitwise | pass | `test_save_load_is_bitwise_into_a_fresh_model`: all `tv`, `ntv`, `ov` arrays equal in value and dtype after loading into an independently built model + optimizer. Non-vacuous: `test_steps_change_every_state_group` (each step changes all three groups). |
| Resume = uninterrupted (plan S6 gate, CPU) | pass | `test_resume_equals_the_uninterrupted_run`: step 2 from the loaded step-1 state equals the uninterrupted step 2 bitwise; the fresh model with the loaded state gives the first model's forward. Control: +1e-3 on one kernel element of the loaded state -> differs. The resumed step uses the first model's compiled step (a second compile costs 145 s, measured); a new process compiles anyway. |
| Model hash form-independent (fix round) | pass | `test_model_hash_is_the_same_for_every_form_of_one_model` x3 (the sections differ as text, the hashes are equal); controls in the same test: `input_sort_field`, `mask_attn: false`, `activation: GELU` in the saved form each change it. `test_model_hash_follows_the_scale_dict_content_not_its_path`: a byte-identical copy elsewhere is equal, one changed number is not. `test_model_hash_follows_the_model_section_only`: data, LR, optimizer and matcher changes keep it, `linformer_k` changes it. |
| Keys stable across builds | pass | `test_keys_are_stable_across_independent_builds`; control in the same test: raw paths and optimizer names differ (> 100 each). `test_optimizer_slots_are_keyed_by_their_variable`: 2 slots per trainable variable, all mapped. Fix round: `test_optimizer_slots_take_the_longest_matching_variable` (variables `a/b` and `a/b_c`, whose slot names are prefixes of each other). |
| Corrupted / partial rejected | pass | flipped byte (sha256), truncated file with a rewritten sidecar hash (zip error), flipped byte inside an array member with a rewritten hash (CRC), foreign sidecar, missing sidecar, config-hash mismatch, a float model's keys; interrupted write (savez raises mid-file) leaves the previous checkpoint loadable and no temporary file. Fix round: a stored float16 array with a consistent sidecar (dtype check); `__meta__` other than the sidecar's with a rewritten hash (control: the rewrite alone loads); an existing checkpoint is not overwritten without `overwrite=True`; each rename is followed by a directory fsync; a second manager save at one step goes to `-r1` and `last` follows it. |
| Export | pass | `test_export_weights_loads_into_a_fresh_model`: `.weights.h5` loads bit-exactly; control: the fresh model differs before loading. |
| Metrics == torchmetrics | pass | Streaming micro/macro accuracy over 3 batches, BinaryRecall / BinaryPrecision (incl. the zero-division 0), macro with absent classes (0.75), `MaskInference`, and the whole `MPflow.log_metrics` dict for one layer (31 names). Count-based values exactly equal; float32 means up to 3 ulps and macro 1 ulp different (measured over 10 seeds, summation order); tolerance 2x (6 / 2 ulps). Controls: one perturbed class, mask bit, regression value, proxy value each change their metric by > 100x the tolerance, and then match torch again. Fix round: `test_epoch_reduction_over_unequal_batches_equals_lightning` x2 runs a real Lightning `Trainer.validate` over batches of 3, 1 and 2 events (one layer; and two layers with `pool_layers=True`): every name equal, counters exact, float means within the per-batch tolerance + 2.4 ulps (WeightedMean sums in float64, Lightning in float32: measured up to 1.17 ulps over 10 cases; 2x). Control: the other weighting (equal weights for a regression mean, event counts for a mask mean) is > 100x the tolerance away. `test_train_stage_logs_only_the_task_metrics` equals `MPflow.log_metrics(..., "train")` (24 names). |
| Prediction h5 == `PflowPredictionWriter` | pass | `test_h5_layout_and_values_equal_the_torch_writer`: same groups, field names, dtypes, shapes, chunks, compression, attributes; values equal (inverse transforms exact, measured 0.0); `load_convert_h5` gives equal arrays. `predict` == torch task `predict` (mask probabilities within 2x the measured 1.19e-7, 0 threshold flips). Controls: a renamed field (`pred_e` -> `pred_E`) changes the layout; a changed class changes exactly `object_class/pflow_class`; padding events are not written. |
| Data adapter | pass | `test_data.py` (10, fix round: `test_float64_is_cast_to_float32_on_the_host`): train drop_last (10 events / batch 4 -> 2), static int32 / float32 / bool device arrays; val padded (`event_valid` [T, T, F, F], invalid-event values, real events unchanged, `event_number` int64 [8, 9, -1, -1]); one val epoch compiles the jitted eval forward + `predict` once. Controls: the torch batch has int64 and a short last batch; `pad_to_batch=False` raises on it; the unpadded short batch compiles a second executable; an int64 above int32 raises; a new shape raises. |
| Real CLIC batches (requiresdata) | **deferred to Polaris/falcon** | `test_data_real.py`: no target at padded nodes, prefix-contiguous `particle_valid` with class 5 exactly at invalid particles, static batches; records the equal-phi rate (RETARGET risk 6). Not runnable here (no ROOT files). Dry run on dummy events: the padded-node check fails on raw dummy events (targets at padded nodes) and passes after cleaning; the prefix check fails with a residual first (3 of 10 events). |
| S6 gate: train `_cache_size() <= 2` over an epoch | **S6b** | Needs the train loop. Only the forward + `predict` part is done (`test_one_val_epoch_compiles_eval_and_predict_once`); the eval executable with `matched_loss` (pure_callback matching, padded-event weighting) is not compiled or counted here: S6b. |

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

Fix round: 28 more mutants (M29-M56; the reviewer's R1-R14 and the new code), **28/28 killed**. The config ones run
only the config tests that build no model (`FAST` in the runner), M52/M53 only the two small checkpoint tests.
The reviewer reported that its mutants R3-R8 and R12-R14 survived `test_config.py` before this round, R1/R2
`test_metrics.py`, R9-R11 `test_checkpoint.py`, and a float64 mutant `test_data.py` (not re-run here on the old
tests); M29-M56 include those changes and are killed by the new tests.

| Mutant | Change | Killed by |
|---|---|---|
| M29 | `lr_max` read from `initial` (R3) | `test_loader_reads_every_form[org_raw]` |
| M30 | `every_n_train_steps = None` (R4) | `test_loader_reads_every_form[polaris_generated]` |
| M31 | `save_last = False` (R5; fix round 2: the step callback's) | `test_loader_reads_every_form[polaris_generated]` |
| M32 | `weight_decay = 0.0` (R6) | same |
| M33 | `seed = None` (R7) | same |
| M34 | `num_val = -1` (R8) | same |
| M35 | `pct_start = 0.3` (R12) | same |
| M36 | `dirpath = None` (R13) | `test_loader_reads_every_form[polaris_generated]` |
| M37 | `mode = "max"` (R14) | `test_loader_reads_every_form[org_raw]` |
| M38 | `batch_size = 32` | same |
| M39 | defaults not removed from the model description | `test_model_hash_is_the_same_for_every_form_of_one_model[org_raw-org_normalised]` |
| M40 | scale dict hashed by path | `test_model_hash_follows_the_scale_dict_content_not_its_path` |
| M41 | `model_hash` of the raw section (the old hash) | `test_model_hash_follows_the_model_section_only` |
| M42 | numbers not canonicalised | `test_model_hash_is_the_same_for_every_form_of_one_model[org_raw-org_normalised]` |
| M43 | BetaPID defaults not filled | `test_loader_reads_every_form[pflow_hgq_raw]` |
| M44 | BetaPID default `p` 2.0 | same (and, run alone, `test_beta_callback_defaults_are_the_hybrid_ones[BetaPID]`) |
| M45 | classification-task means weighted 1 (R1) | `test_epoch_reduction_over_unequal_batches_equals_lightning[False]` |
| M46 | mask means weighted by batch size (R2) | same |
| M47 | regression means weighted 1 | same |
| M48 | train stage logs the custom metrics | `test_train_stage_logs_only_the_task_metrics` |
| M49 | `pool_layers` ignored | `test_epoch_reduction_over_unequal_batches_equals_lightning[True]` |
| M50 | stored dtype not compared (R10) | `test_stored_dtype_other_than_the_models_is_rejected` |
| M51 | metadata not compared with the sidecar (R11) | `test_metadata_other_than_the_sidecars_is_rejected` |
| M52 | slot reference matched shortest first (R9) | `test_optimizer_slots_take_the_longest_matching_variable` |
| M53 | no directory fsync | `test_renames_are_made_durable_by_a_directory_fsync` |
| M54 | overwrite guard removed | `test_save_does_not_overwrite_a_checkpoint` |
| M55 | manager writes `step-<N>.npz` again at one step | `test_manager_keeps_last_k_and_the_best` |
| M56 | float64 not cast | `test_float64_is_cast_to_float32_on_the_host` |

Fix round 2: 23 more mutants (M57-M79; the reviewer's X1, X6, X11, MR1 and the new code), **23/23 killed**, 5-10 s
each (selected tests: `DATA_WORKERS`, `CK_NEW`, `CFG_RUN` in the runner). M30, M31, M32, M36 and M37 were retargeted to
the new `_checkpoint_callback` / `_optimizer` code (their old snippets no longer exist).

| Mutant | Change | Killed by |
|---|---|---|
| M57 | no `multiprocessing_context` (platform default: fork on Linux) | `test_workers_are_never_forked` |
| M58 | `WORKER_START_METHOD = "fork"` | same |
| M59 | fork guard disabled | same |
| M60 | `persistent_workers=False` | same |
| M61 | `SplitBatches.epoch` does not move the sampler | `test_spawned_workers_give_the_in_process_batches_without_a_fork_warning` |
| M62 | no chmod before the rename | `test_checkpoint_files_get_the_mode_of_a_plain_write[18]` (umask 0o022) |
| M63 | stored shape not compared (X1) | `test_stored_shape_other_than_the_models_is_rejected` |
| M64 | NaN guard removed (X11) | `test_nan_at_the_first_validation_is_not_the_best` |
| M65 | batch signature without dtype (X6) | `test_a_new_batch_shape_or_dtype_raises[int32]` |
| M66 | `ov` keys not compared (MR1) | `test_checkpoint_of_another_optimizer_is_rejected[AdamW-Lion]` |
| M67 | hepattn `Checkpoint` save_last read from YAML | `test_loader_reads_every_form[org_raw]` |
| M68 | hepattn `Checkpoint` top-k read from YAML (1) | same |
| M69 | no default `every_n_epochs=1` | same |
| M70 | both triggers accepted | `test_unsupported_run_options_raise[mutually exclusive]` |
| M71 | hepattn `Checkpoint` dirpath read from YAML | `test_hepattn_checkpoint_settings_overridden_by_its_setup_are_ignored` |
| M72 | optimizer name case-sensitive | `test_optimizer_name_is_matched_case_insensitively` |
| M73 | missing weight_decay not rejected | `test_unsupported_run_options_raise[weight_decay]` (KeyError, not ConfigError) |
| M74 | epsilon 1e-7 | `test_optimizer_name_is_matched_case_insensitively` |
| M75 | top-1 best removes the previous best of any kind | `test_manager_keeps_every_epoch_checkpoint_and_points_at_the_best` |
| M76 | `save_last` always writes | `test_manager_save_last_follows_validation_ends` |
| M77 | stale `last-*` files kept | same |
| M78 | NaN monitor value stored as a float | `test_manager_keeps_every_epoch_checkpoint_and_points_at_the_best` |
| M79 | every epoch becomes best | same |

Checkpoint mutants take 200-270 s each (the train-step compile); the others 6-25 s.

Fix round 3: 16 more mutants (M80-M95; the reviewer's N1 = M93 and D3 = M94, and the new code); M3 retargeted to
`_merge_args` (its old `deep_merge` snippet is gone). Selections: `MERGE`, `CK_KEEP` in the runner.

| Mutant | Change | Killed by |
|---|---|---|
| M3 | `_merge_args` replaces every value (overlay drops base keys) | `test_loader_reads_every_form[org_raw]` |
| M80 | keys next to `class_path` not folded into `init_args` | `test_overlays_merge_as_lightningcli[eval]` (the naive-merge control must be rejected) |
| M81 | dict-typed arguments merged key by key | `test_overlays_merge_as_lightningcli[partial]` |
| M82 | `KerasMaskFormer` encoder / decoder merged as classes | `test_merge_rules` |
| M83 | plain-dict class-typed keys (raw decoder) replaced whole | `test_overlays_merge_as_lightningcli[eval]` |
| M84 | `class_path` change accepted | `test_merge_rejects_what_jsonargparse_rejects_or_needs_the_class_for[...changes class_path]` |
| M85 | top-level sections replaced whole | `test_overlays_merge_as_lightningcli[eval]` |
| M86 | `keep_last_k=-1` prunes | `test_keep_last_k_is_lightnings_save_top_k[-1-False-...]` |
| M87 | `keep_last_k=0` writes a step file | `test_keep_last_k_is_lightnings_save_top_k[0-False-...]` |
| M88 | `keep_last_k` not validated | `test_keep_last_k_is_validated[-2]` |
| M89 | `from_callback` keeps 1 whatever `save_top_k` | `test_manager_from_the_polaris_step_callback` |
| M90 | `epoch_due` without the `+ 1` | `test_manager_from_the_hepattn_epoch_callback` |
| M91 | missing monitor not rejected | same |
| M92 | `from_callback` drops `save_last` | `test_manager_from_the_polaris_step_callback` |
| M93 | `abs_err / (|t| + 1e-8)` (reviewer N1) | `test_abs_norm_res_adds_the_epsilon_after_the_division` |
| M94 | `event_number` copied to the device, not popped (reviewer D3) | `test_train_batches_drop_last_and_are_static_int32` |
| M95 | `event_valid` ignored | `test_event_valid_drops_padding_events` |

## Deviations

1. New files outside the listed ones: `src/hepattn/experiments/clic/native_data.py` (the torch half of the adapter, as
   RULES require), `tests/keras_native/s6_fixtures/` (fixtures + generator), `docs/hgq/port_log/S6a_mutations.py`.
2. Plan 3.9 says "keyed by `variable.path`"; the keys strip auto-generated suffixes (decision 2).
3. `ClicMetrics` reports per-layer values by default; the Lightning module pools layers (decision 5;
   `pool_layers=True` reproduces it).
4. The resume gate steps with the first model's compiled function (cost); the fresh model is checked by an eager forward.
5. The `on_test_end` ROOT file of the torch writer is not written (only the h5); `load_convert_h5` reads the native h5.
6. Fix round 2: `CheckpointConfig` changed shape (two `CheckpointCallback`s instead of flat fields); no S6b code used it yet.
   The checkpoint policy follows the hybrid callbacks, not the plan 3.9 text (decision 10).
7. Fix round 3: a run whose validations so far all give NaN has no native `best` pointer; Lightning's
   `best_model_path` names a NaN epoch then (decision 10 (c)).
8. Fix round 3: `merge_configs` raises for an overlay that changes a `class_path` (jsonargparse keeps the init args
   the new class accepts; the loader has no torch classes to ask), and the loader rejects a class entry with
   `init_args` next to other keys (jsonargparse rejects it too).

## Open issues

- **S6b, validation loss with padded events.** The objective has no per-event weights: a padded event adds 150
  null-target queries to the classification CE mean (and zeros elsewhere). Measured on two dummy events with one of them
  replaced by an invalid event: total 49.80 vs 50.21 (float), 72.57 vs 78.18 (quantized). Torch evaluates the short last
  batch unpadded. S6b: run the last batch's loss with per-event terms, or a second executable at its size, and weight
  `val/loss` by real events (Lightning's epoch mean weights by batch size).
- **S6b:** metrics and the writer need `predict` of the *matched* (permuted) outputs, as torch's `validation_step` /
  `test_step` (after `model.loss`); `permute_outputs` with `matched_loss`'s `idx`.
- **S6b:** pass `expect_config_hash=rc.model_hash` to `load_checkpoint` (safe since the fix round: the hash is the same
  for the raw, generated and saved forms, on any host, and for another LR / optimizer / matcher / data setting; a
  checkpoint written before the fix round has an old-style hash and needs `expect_config_hash=None`); resume data position at an optimizer-step
  boundary with accumulation (`SplitBatches.epoch(epoch, start_batch)`).
- **S6b, checkpoint wiring** (decision 10; fix round 3 replaces the hand mapping `keep_last_k = save_top_k`, which
  deleted step checkpoints for -1): one `CheckpointManager.from_callback(directory, callback)` per non-None
  `rc.checkpoint.epoch` / `rc.checkpoint.step`. Epoch manager: `maybe_save_epoch(epoch, step, {"val/loss": ...})` at
  each train epoch end, after that epoch's validation, in the run's `ckpts` directory (the hepattn callback ignores a
  YAML `dirpath`). An epoch cut short by `max_steps` runs no validation; Lightning still saves it in
  `on_train_epoch_end` with the last logged `val/loss` (and only warns if no validation has run yet): the driver
  must pass the last val metrics then, or skip that save and record the deviation. Step manager: `maybe_save_step`
  after each optimizer step, and, when `manager.write_last`, `save_last(step, ...)` at each epoch end and at the
  train end (else `last` can be up to `every_n_steps - 1` steps old), in `callback.dirpath`.
- **S6b, driver entry point under spawn** (decision 9): spawned DataLoader workers re-import the driver's main
  module, so keep the driver under `if __name__ == "__main__":` and run no JAX computation at import (otherwise each
  of the 8 workers can start a JAX GPU backend and preallocate 75% of the GPU memory); optionally set
  `JAX_PLATFORMS=cpu` in the worker environment. Measured: importing `native_data`, `checkpoint`, `metrics` and
  `predict_writer` leaves `jax._src.xla_bridge._backends` empty.
- **S6b:** pass `event_valid=batch.event_valid` to `ClicMetrics.update` and `PredictionWriter.write_batch` for padded
  val / test batches (decision 14).
- **User to confirm, `ClicMetrics` default** (decision 5; reviewer minor): the default is per-layer values
  (`pool_layers=False`); the Lightning module pools every layer into one metric object, so its logged
  `val/final_*` values differ. G4 comparisons with hybrid logs must use `pool_layers=True`. Alternative: make
  `pool_layers=True` the default so native logs equal hybrid logs unless asked otherwise.
- **Not owned, `polaris/03_make_config.py`:** its `merge` merges every dict key by key (decision 13). For org's
  base + linformer this gives the LightningCLI result (linformer.yaml repeats its dicts whole, as its comment says);
  an overlay with a partial dict-typed argument or eval.yaml's short form would not. `tests/keras_native/test_pr8_layers.py`
  and `golden/mf_cases.py` keep their own copy of the old merge, used only on base + linformer (same result).
- **Plan owner:** plan 3.9 says "best on val/loss"; the hybrid keeps every epoch checkpoint (`save_top_k=-1`) and its
  `Checkpoint` never writes `last` (decision 10). Correct the plan text.
- **S6b, optimizer builder:** `keras.optimizers.AdamW(epsilon=rc.optimizer.epsilon)` (1e-8, torch's default; keras'
  default is 1e-7), or record the deviation.
- **S6b:** host hooks of plan 3.10 (`KERAS_JAX_PLAN.md:394`): `MetricsCsvWriter`, `EBOPsMonitor` and `BitwidthMonitor`
  are not part of S6a and must be written in the loop.
- **S8 / G4, data order:** G4 needs the same data order in the hybrid and JAX runs. The hybrid train loader is
  `DataLoader(shuffle=True)` (a `RandomSampler` seeded from the global torch RNG) without `drop_last`
  (`pflow_data.py:817-831`); the native one is `EpochSampler(seed + epoch)` with `drop_last`. Their orders differ, and
  the hybrid has one more (short) batch per epoch unless S8 injects `EpochSampler` (with `drop_last`) into the hybrid
  data module or exports an order file that both read.
- **Polaris, worker start cost** (decision 9): measure the first-batch time and the memory of 8 spawned workers with
  the real train `CLICDataset` (each receives the pickled in-memory arrays). If that is too large, give the workers a
  dataset that opens the ROOT file itself in `worker_init_fn`.
- **S6b / S8:** the generated Polaris config defaults to `precision: bf16-mixed`; the user decided fp32 first, so the
  driver must ignore it or the config be generated with `HGQ_PRECISION=32`. `RunConfig.trainer["precision"]` carries it.
- **S5:** BetaPID's `min_beta` is 0.0 (the hybrid default, now filled in by the loader for raw configs too); plan 3.6's
  EBOPs read-out (sum(losses) / beta) needs beta > 0: S5/S6b must clamp or read EBOPs another way when beta is 0.
- **S6b:** use `ClicMetrics(pool_layers=True)` when comparing with hybrid logs (G4); `ClicMetrics("train")` per
  logged step with `reset`.
- **Polaris, before S8:** run `CLIC_DATA_DIR=/eagle/hgcal-maskformer-fpga/clic_data CLIC_DATA_REQUIRED=1 python -m pytest -m requiresdata tests/keras_native/test_data_real.py -s`
  and record the counts here (the padded-node incidence-0 gate is still open; the test asserts that every event has a
  padded node, so it is not vacuous).
  Risk found by reading: `pflow_data.py:461-462` sets `particle_valid` only for non-residual particles among the first
  `n_particles` (pdgid -999 -> class 5, `pflow_data.py:687`), so a residual before a real particle breaks the prefix
  that both torch's `Matcher` (`matcher.py:317-320`, lengths = valid count) and native `host_match` assume.
- `CLICDataset.__init__` calls `seed_everything(42)` (`pflow_data.py:49`): the driver must seed after building datasets.
- Not owned: add an S6a row to `docs/hgq/port_log/README.md` and close the "S4 -> S6 real CLIC batches" row there when
  the requiresdata run has passed; plan status table.
