# S6b log: training driver, command line, Polaris smoke and G3 scripts

Plan: `docs/hgq/KERAS_JAX_PLAN.md` sections 3.5 and 3.8-3.10 (the loop and its wiring) and the S6 gates of section 4 that
need the loop. Branch `keras-jax` in `hepattn-keras-jax`, from `c0d143e` (S5, S6a and the S56 delta fix merged). Date
2026-10-20, macOS arm64 CPU, `.venv-jax` (jax 0.11.2, keras 3.15.0, hgq2 0.1.9) for the native code, `hepattn/.venv`
(lightning 2.5.2) for the hybrid arm and the hybrid probe. No Polaris access from this session: the two Polaris scripts
ran locally at tiny sizes only.

## What landed

| File | Content |
|---|---|
| `src/hepattn/keras_native/loop.py` | `Trainer`: model (eager build), optimizer from `RunConfig.optimizer` (`build_run_optimizer`), `TrainStep`, epochs over a `DataSource`, BetaPID per epoch / BetaScheduler ramp per optimizer step / PeriodicEBOPs at the optimizer step, train log rows every `log_every_n_steps` optimizer steps, validation (event-weighted, padded batches, `ClicMetrics`), EBOPs / bitwidth monitors, `MetricsCsvWriter`-layout CSV, step / epoch / last checkpoints (`CheckpointManager.from_callback`, `resume_path`), `restore` (exact resume), `test` (prediction h5). Also `LoopSettings`, `CsvLog`, `CompileEvents` (persistent-cache events), `enable_compilation_cache`. Torch-free. |
| `src/hepattn/keras_native/cli.py` | `fit` / `test` commands, the trainer options a Polaris job sets, run directory layout; the data source is injected. |
| `src/hepattn/experiments/clic/main_jax.py` | Entry point (`python -m hepattn.experiments.clic.main_jax`): sets `KERAS_BACKEND=jax`, CLIC data through `native_data` (`ClicSource`), all work under `main` (spawned workers re-import it). |
| `polaris/16_jax_smoke.{py,pbs}` | Real-data smoke on one A100 (run A + resume B in new processes, checks written after each stage). |
| `polaris/17_g3_step.{py,pbs}` | Gate G3: jobs `jax-prod`, `jax-g1`, `jax-precision`, `jax-match`, `hybrid` (each its own process, JAX first, JSON after every job, `verdict()`). |
| `polaris/jax_common.py` | Shared by 16 / 17 / the test fixture: `generate_config`, `shrink_config` (tiny sizes), timing summaries. |
| `polaris/submit.sh` | Cases for 16 and 17 (debug queue, 1 h). |
| `tests/keras_native/test_loop.py`, `test_cli.py`, `s6b_utils.py`, `s6b_fixtures/{make_tiny_config.py,tiny_polaris.yaml}` | S6b gates; the fixture is the generated Polaris config at test sizes (generator in the file's docstring). |
| `src/hepattn/keras_native/optim.py` | One-line fix (S5's file): `optimizer_from_config` rejected an explicit `gradient_clip_algorithm: null` (deviation 10). |
| `docs/hgq/port_log/S6b_val_loss_probe.py`, `S6b_mutations.py`, `S6b_evidence/` | Evidence. |

## Decisions

1. **`val/loss` is the task loss.** The hybrid logs `{stage}/loss` inside `ModelWrapper.aggregate_losses`
   (`src/hepattn/models/wrapper.py:46-62`) before `MPflowHGQ.aggregate_losses` adds `quant_losses()`
   (`src/hepattn/experiments/clic/lightning_module_hgq.py:138-142`), which it logs as `{stage}/quant_ebops_loss` and only
   returns for the backward. Measured with the real method on a stub (`S6b_val_loss_probe.py`, hybrid env): task terms
   2 + 3, quant 7 -> logged `val/loss` 5.0, `val/quant_ebops_loss` 7.0, returned 12.0. **S5.md ("S6 must log
   `out.loss`") is wrong**; the epoch-checkpoint monitor is the task loss. The driver logs `train/loss` / `val/loss` =
   task, `*/quant_ebops_loss` = penalty + regularizers (train) or regularizers (val), and `train/total_loss`.
2. **Padded val batches.** The val program (`Trainer._val`: `training=False` forward, matched loss, regularizers,
   `predict` of the permuted outputs) runs on the padded batch; metrics drop the padding events (`event_valid`). For a
   short batch the task loss is recomputed on its real events by a loss-only program (`_tail`) from the padded forward's
   outputs, so no second model program compiles. Measured: that loss differs from an unpadded forward's by 2 float32
   ulps (task total; each term <= 2 ulps), the padded batch's own loss by 188961 ulps; `val/loss` over the 10 val events
   then differs from the all-unpadded computation by 3.52e-8 relative. `val/*` are means weighted by the real events of
   each batch (Lightning infers `batch_size` from the batch).
3. **One optimizer reader** (S56 open issue 1): the driver builds from `RunConfig.optimizer` (`build_run_optimizer`, AdamW
   epsilon 1e-8 from the config). `test_one_optimizer_reader_for_every_form` checks that `optimizer_from_config` gives
   the same `get_config()` (auto `name` dropped) for 6 forms (tiny generated, org raw, `pflow_hgq.yaml`, the three
   saved fixtures). It found that `optimizer_from_config` raised `NotImplementedError` for the saved org config
   (`s6_fixtures/org_normalised.yaml`, the form `SaveConfig` writes), whose `gradient_clip_algorithm: null` Lightning
   treats as `norm`; fixed (`or "norm"`).
4. **Counters and data position.** `global_step` = optimizer steps, `micro_step` = micro-batches, data position
   `(epoch, batch)` = the next micro-batch. Checkpoints are written only after an optimizer step (no accumulation in
   flight). `(e, micro_per_epoch)` means "epoch e trained, its end hooks (validation, epoch checkpoint) due": a step
   checkpoint at the last step of an epoch resumes into that epoch's end hooks.
5. **Beta.** BetaPID at each epoch start with the EBOPs read-out (`penalty / beta`) of the latest EBOPs-active
   micro-batch with beta > 0, as the hybrid reads the `_ebops` written by the latest active step; the PID holds while no
   read-out exists (its own rule for `ebops <= 0`). BetaScheduler: `ramp_beta(global_step)` before every micro-batch
   (the hybrid's `on_train_batch_start` at `trainer.global_step`). `set_beta` only when the value changes. The host
   keeps the float64 beta it wrote (the read-out divides by it); `restore` takes it from the metadata and checks it
   against the float32 in the state. No refusal for `min_beta <= 0` (S56 open issue 4): a warning, and a beta of 0 just
   has no read-out (logged NaN).
6. **PeriodicEBOPs** at the optimizer step (`global_step % every == 0`), so all micro-batches of an active step carry the
   term (the hybrid reads `trainer.global_step`).
7. **Train log rows** at optimizer steps with `global_step % log_every_n_steps == 0` (Lightning's logger cadence,
   `logger_connector.should_update_logs`: `(_batches_that_stepped + 1) % n`), with the losses of that step's last
   micro-batch, the read-out, beta, the LR of the step, ms per micro-batch since the last row, bitwidth stats and the
   train task metrics of that micro-batch (`ClicMetrics("train")` on the `training=False` forward, deviation 5).
8. **Checkpoints**: one `CheckpointManager.from_callback` per configured callback, both in `<run_dir>/ckpts`, with
   `resume_path` (S56 open issue 6) and `expect_config_hash=rc.model_hash` on load. Metadata: counters, data position,
   beta, last EBOPs read-out, PID state, last val metrics, `total_steps`, `micro_per_epoch`, seed, model hash. `restore`
   refuses another `total_steps` or `micro_per_epoch` (the schedule and the data position would not continue).
   `--ckpt-every-n-steps` / `--ckpt-save-top-k` override (or add) the step callback.
9. **`total_steps` guard** (S56 open issue 5): stepping when `global_step >= total_steps` raises `RuntimeError`, as torch's
   OneCycleLR does at step `total_steps + 1`.
10. **Precision**: a config whose `trainer.precision` is not fp32 is refused (generate with `HGQ_PRECISION=32`); the steps
    run under `jax.default_matmul_precision("highest")` (S5 default; `--matmul-precision`).
11. **Compilation cache**: `--compile-cache` (default `$JAX_COMPILATION_CACHE_DIR` or `~/.cache/hepattn/jax`), JAX's
    default 1 s threshold. Measured (CPU, tiny config): run 1 0 hits / 3 misses, the resumed run 3 hits / 0 misses; the
    first train step took 14.4 s vs 6.9 s (the cache skips the XLA compile, not the tracing), the eager model build
    21.7 s vs 22.1 s (not cached). Programs with the matching `pure_callback` are cached (checked before writing the
    driver with a toy `pure_callback` program, and by the hits above).
12. **Data**: `ClicSource` (one `SplitBatches` per split for the run, spawned workers); datasets are built before
    `keras.utils.set_random_seed(seed)` (`CLICDataset` seeds 42 in its constructor); train order `EpochSampler(seed +
    epoch)`, resumed at the micro-batch.
13. **Run directory**: `config.yaml`, `run.json`, `ckpts/`, `steps.csv`, `metrics.csv`, `summary.json`; the YAML's
    `ModelCheckpoint.dirpath` and `MetricsCsvWriter.path` (Polaris `CKPT_DIR`, shared between runs) are not used,
    because checkpoint pruning needs one run per directory.
14. **G3 design** (`17_g3_step.py`): fp32 in both arms, the generated config's accumulation 4 and EBOPs period 8, one
    GPU, data on the device (the hybrid `StepTimer` also excludes loading), micro-batch wall times synchronised. The
    production arm times EBOPs-on and -off micro-batches separately (EBOPs share = 1 - off / on); `jax-g1` runs S5's
    four configurations minus the production one (in `jax-prod`) on one model and optimizer: `TrainStep(None, donate)`,
    `TrainStep(None, no donation)`, Keras `train_on_batch` with `compute_loss` = the matched task loss + `model.losses`
    (same objective), and traces both with `jax.profiler` when the donating step is slower; `jax-precision` compares
    "highest" and "default" from one state (outputs, prediction flips, 20-step loss trajectories, step time);
    `jax-match` compares the step with host matching against the same step with an identity assignment and no callback
    (costs computed and kept live through `live * 0`, which XLA does not fold for floats). Ratio on medians (as `13`
    reports) and on means; decided at batch 32, at 16 if the hybrid OOMs at 32.

## Gates

| Gate | Result | Evidence and control |
|---|---|---|
| Save, load and step: bitwise equal to the uninterrupted run (CPU) | pass | `test_resume_continues_bit_for_bit` x4: a fresh `Trainer` restored from run A's checkpoint at (0, 6) inside epoch 0, (0, 8) epoch end pending, the epoch-0 checkpoint (1, 0), and (1, 4) after the PID's first step, ends with A's final state in all 2285 arrays, equal logged train losses / read-outs, equal val losses, equal beta and read-out. Controls: `test_control_a_lost_piece_of_resume_state_changes_the_run` (one micro-batch skipped; PID state dropped) give another final state. Mutants K5 (PID state not restored), K6 (data position off by one), K7 (host beta from the float32 state) killed. New process: `test_resume_in_a_new_process_is_bitwise_and_hits_the_compilation_cache`. |
| Trace counts over an epoch of dummy data: <= 2 train, <= 1 eval / predict | pass | `test_trace_counts_over_the_run` over 3 epochs (stronger than one): train 2 (EBOPs on / off), val 1 (eval + predict in one program, also used for the train-metric rows and `test`), tail 1 (loss only), bits 1; `TrainStep.eval_step` / `predict_step` 0 (unused). Control: a batch of another shape adds a val trace (2). Counts are traces, as S5 deviation 5 asks. Same counts in the CLI process (`test_fit_writes_checkpoints_logs_and_summary`). |
| val/loss as the hybrid defines it | pass | `test_val_loss_is_the_event_weighted_task_loss_on_real_events` (= unpadded event-weighted task loss within 2x the measured 3.52e-8; `val/quant_ebops_loss` = regularizers), `test_tail_loss_from_the_padded_forward_equals_the_unpadded_batch` (<= 2x the measured 2 ulps; control: padded 188961 ulps). Mutants K2 (task + regularizers), K3 (no tail recompute), K4 (weights = batch size) killed. Hybrid definition: decision 1 probe. |
| Prediction h5 layout | pass (S6a) + wiring | S6a `test_h5_layout_and_values_equal_the_torch_writer` (in the full-suite run below). Driver: `test_test_command_writes_the_prediction_file` (the `test` command on the best checkpoint: groups, 10 events in order 0..9, no padding rows, mask fields). |
| Metrics = torchmetrics | pass (S6a) + wiring (fix round) | S6a `test_metrics.py` (full-suite run). Driver wiring, added in the fix round (the first round had no test for it): `test_val_metrics_drop_the_padding_events_and_pool_the_layers` compares `Trainer.validate`'s val metrics with `ClicMetrics(pool_layers=True)` on the real events only (val batches 4, 4, 2): equal bitwise in all 42; the torchmetrics counters are pooled (one value for every layer). Controls: counting the padding events changes 26 of 42 values; per-layer values differ. Mutants K13 (`event_valid=None`, reviewer R1 / R6) and K14 (per layer, R11) killed. |
| YAML loader reads every form | pass (S6a) + one reader | S6a `test_config.py`; `test_one_optimizer_reader_for_every_form` x6 (decision 3; control: another accumulation is seen). Mutants K11 (keras epsilon 1e-7 in the driver) and K12 (the old null-algorithm check) killed; the old `optim.py` fails `[org saved]` with `NotImplementedError` (measured before the fix). |
| CLI end to end from a clean shell: checkpoints, metrics CSV, resumable state, loss decreases, quantizers move | pass | `test_cli.py` (4 passed): env = `PATH`, `HOME`, `TMPDIR`, `PYTHONPATH` only; `fit` with 2 spawned workers -> 6 step + 3 epoch checkpoints, `steps.csv` 12 rows, `metrics.csv` 3 rows, `summary.json`; mean train loss of the last 4 rows < first 4, val loss epoch 2 < epoch 0, > 10 trainable quantizer arrays differ between steps 2 and 12, `train/bits_b_mean` changes; resume in a new process bitwise with 3 cache hits (control: run 1 had 0); `test` command. `test_run_a_is_non_vacuous`: every state group changes, beta changes after the warm-up. |
| PeriodicEBOPs / BetaPID / BetaScheduler wiring | pass | `test_periodic_ebops_follows_the_optimizer_step` (active rows = steps 1, 3, 5, ...; a micro-step counter would log 0 on every row; on-step HGQ2 term > 5x off-step), `test_beta_pid_steps_each_epoch_on_the_latest_readout` (an independent controller fed the logged read-outs gives the logged betas; control: without a read-out the PID would hold init_beta), `test_beta_scheduler_ramps_at_the_optimizer_step` (control: the next step's ramp differs). Mutants K1 (micro-step period), K9 (ramp one step late) killed. |
| Checkpoint policy, resume checkpoint kept, `total_steps` guard, refusals | pass | `test_checkpoint_policy`, `test_resume_checkpoint_survives_pruning` (top-k 1 keeps the newest and the resume file; K10 killed), `test_stepping_past_total_steps_raises` (K8 killed), `test_restore_refuses_another_schedule_or_model` (control: equal lengths load), `test_non_fp32_precision_is_refused`. |
| Both Polaris scripts run locally at tiny sizes on dummy data (both G3 arms) | pass | Latest run: fix round 2 (section below; `S6b_evidence/{smoke_tiny,g3_tiny}.json`): `16_jax_smoke.py --tiny` SMOKE-PASS, 8 of 8 checks, resume bitwise; `17_g3_step.py --sweep --tiny --iters 40` all 8 jobs (now with `hybrid-tf32`) with no error, the hybrid at `torch_matmul_precision` "highest" ("high" for `hybrid-tf32`). The tiny CPU numbers are not G3 evidence (CPU ignores the matmul precision, 1 + 1 layers, dim 16, and they move with machine load: matching share 0.137 in fix round 1, see fix round 2). |
| G3 on the A100 (>= 1.8x the hybrid in-run step; matching <= 15%; G1 gap; TF32; EBOPs share at b32) | **deferred to Polaris** | `bash polaris/submit.sh 17_g3_step.pbs`; no Polaris access in this session. G1 stays OPEN. |
| Real-data smoke and resume on the A100 | **deferred to Polaris** | `bash polaris/submit.sh 16_jax_smoke.pbs` (S6a open issue 3: GPU resume within `--rtol` 1e-3, bitwise count reported). |
| Lint | pass (fix round) | At `1aaf041` this row was wrong: ruff 0.13.0 `--preview` (the `.pre-commit-config.yaml` rev and args) reported E501 at `polaris/16_jax_smoke.py:67` and `ruff format --check` would reformat it (reproduced on `git show 1aaf041:polaris/16_jax_smoke.py`). Fixed; `prek run --files` and `uvx ruff@0.13.0 check --preview` / `format --check` pass on every file of the fix round. |

## Fix round (review of `1aaf041`)

Each reviewer issue was checked before it was fixed. Commits `c4d0dca` (driver), `4b4e3ec` (command line),
`d104d44` (Polaris scripts) and the docs commit.

| Issue | Verified | Fix and evidence |
|---|---|---|
| (major) No test of the val-metric wiring (`event_valid`, pooled) | yes: R1/R6/R11 survived the first-round suite | New driver test (gate row "Metrics"); K13, K14 killed. |
| Lint claim false for `16_jax_smoke.py` | yes (E501 at 67, `format --check`) | Wrapped; Lint row corrected. |
| BetaPID / train columns lost after a mid-epoch resume; header fixed at the first row | yes (by reading `meta` / `CsvLog`) | `meta()` stores `epoch_logs` / `train_logs`, `restore` reads them; `CsvLog` adds columns for new keys (deviation 12). `test_resume_continues_bit_for_bit` now compares every non-`time/` column of every `steps.csv` / `metrics.csv` row: K20 (logs not restored) fails 3 of 4 positions (the epoch-checkpoint position needs no logs), K24 (fixed header) killed by `test_csv_log_grows_its_header`. |
| Untested log cadence, logged LR, bit counts over non-trainable variables, `check_val_every_n_epoch`, matmul precision of `_val` / `_tail` | yes: R2/R3/R5/R8/R9 survived | `test_log_cadence_lr_bits_val_interval_and_precision` (log every 3 steps across the 4-step epochs, validation every 2nd epoch): logged steps [3, 6], `train/lr` = schedule at `(step - 1) * k` (control: next update's LR differs), `bits_*_n` = trainable + non-trainable counts (non-trainable quantizer variables exist), one val row (epoch 1), every `dot_general` of the val and tail programs HIGHEST. K15-K19 killed. |
| G3 compile count underestimated; hybrid arms run last | yes: ~16 programs, not ~9 | `--jobs` order kept; default gate set first (`jax-prod:32`, `hybrid:32`, `hybrid:16`, `jax-match:32`, `jax-prod:16`: 6 JAX programs); `17_g3_step.pbs` submits the gate set, `G3_JOBS=jax-g1:32,jax-precision:32` the diagnostics (~10 programs) as a second job. |
| Hybrid peak memory read in the wrong process | yes | The hybrid child (`13_step_time_fit.py` run through `runpy`) prints `torch.cuda.max_memory_allocated()` at exit; parsed into `peak_dev_gb` (None on CPU: the line is `-1`; seen in `g3_tiny.json`). |
| `meta()` (device sync) on every optimizer step | yes | Built only when a step checkpoint is due. |
| Resume with another `--seed`; `run.json` / `config.yaml` overwritten | yes | `restore` refuses another seed (`check_seed=False` for `test`); later processes write `run-r<n>.json` / `config-r<n>.yaml`. K23, K26 killed. |
| `config.yaml` without the CLI overrides | yes (measured by the reviewer; `apply_overrides` leaves `rc.raw`) | `cli.effective_raw` writes the data, trainer, seed and step-checkpoint options in; `test_saved_config_holds_the_command_line_options` reloads it to run 1's data / checkpoint / trainer settings and model hash (control: the fixture differs in `num_workers` and the step callback). K25 killed. |
| Stale `.tmp` after a kill; duplicated CSV rows on a same-directory resume | yes | `Trainer._managers` removes `ckpts/.*.tmp`; `fit` after `restore` keeps CSV rows with step <= the checkpoint's and epochs < its data-position epoch (deviation 13). `test_resume_in_the_same_run_directory` (copy of run A + a stale temporary, resume at step 6): 12 step rows equal to A's, 3 epoch rows equal, temporary gone, state bitwise. K21, K22 killed. CLI: `--resume last` in a copy of run 1 replaces, not repeats, the epoch-2 row. |
| `16_jax_smoke.py` assumes a step file at the final step | yes | Final files from each run's `last.json`; new check `resume_same_final_step`. |
| G3 verdict at b16 only when the hybrid OOMs at b32 | yes | Records `substitution` and reports `per_event_jax_b32_vs_hybrid_b16`. Checked on synthetic results: hybrid b32 OOM -> decided at 16 with both fields; with a hybrid b32 result neither field appears. |
| Matching share measured with EBOPs on every step | yes | `jax-match` uses the config's period: shares of on and off micro-batches and of the production mix (1 on : `every - 1` off); the gate uses the mix. Tiny CPU: on 0.154, off 0.134, mix 0.137 (not A100 evidence). |
| GPU resume may drift past `--rtol` | plausible | `16_jax_smoke.pbs` exports `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (`SMOKE_XLA_FLAGS=` drops it). The flag is in jaxlib 0.11.2 (`libjax_common.dylib`), and the CPU build accepts it, whereas an unknown flag (`--xla_gpu_bogus_flag`) aborts with "Unknown flag in XLA_FLAGS". The pass still needs only `--rtol`; `resume_bitwise` is reported. |
| Evidence JSONs are git-ignored | yes (`.gitignore:8-9`) | `smoke_tiny.json`, `g3_tiny.json`, `val_loss_probe.out` added with `git add -f` (re-run on the fix-round code). |
| Per-term `float()` syncs in `validate` / `_log_step` | yes | One `jax.device_get` per val batch / logged row. |
| `save_last` only on the step manager; "Lightning would write last.ckpt" | **no** (push back) | hepattn's `Checkpoint.setup` sets `self.save_last = False` (`src/hepattn/callbacks/checkpoint.py:13-15`), so the hybrid writes no `last.ckpt` with only that callback either, and `config._checkpoint_callback` already reads it as `save_last=False`. No code change: `--resume last` now says that a step callback is needed and points to epoch checkpoint paths; KERAS_NATIVE.md says the same. |

Tiny local re-runs on the fix-round code (CPU, dummy data): `16_jax_smoke.py --tiny` (with the GPU determinism
flag set) SMOKE-PASS, all 8 checks true, resume bitwise in 2285 of 2285 arrays; `17_g3_step.py --sweep --tiny --iters
40`: all 7 jobs, in the new order, with no error (`S6b_evidence/{smoke_tiny,g3_tiny}.json`).

## Fix round 2 (review of `8380d8a`)

Each issue was checked before the fix. Commits `4276ef2` (driver + tests), `dbbfae9` (CLI tests), `70d6f6a` (G3
precision), `bd6d1ba` (submit.sh) and the docs commit. Mutants K27-K35: `S6b_mutations.py`, log appended to
`S6b_evidence/mutations.log`.

| Issue | Verified | Fix and evidence |
|---|---|---|
| (major) G3 hybrid arm at TF32, labelled fp32 | yes: `hepattn.utils.cli.CLI` defaults `--matmul_precision high` (`src/hepattn/utils/cli.py:54-59`) and calls `torch.set_float32_matmul_precision` (`:113-114`); nothing in `polaris/` set it. Measured: a scratch copy of `17_g3_step.py` without the new option makes the tiny hybrid job report "high" and fail ("used 'high', asked 'highest'"). | `job_hybrid` passes `--matmul_precision highest`; the child prints `torch.get_float32_matmul_precision()` at exit; the job fails on a mismatch and records `torch_matmul_precision`. New diagnostic `hybrid-tf32:16` (CLI default "high", `precision` "tf32") and verdict field `hybrid_tf32_b16`. Tiny CPU: hybrid "highest" 154.8 ms, hybrid-tf32 "high" 149.2 ms at b16 (CPU has no TF32; the values only show the wiring). Earlier hybrid in-run numbers from `13_step_time.pbs` (CLI) were at "high"; G2's hybrid arm (`15_g2_forward.py`, no CLI) ran at torch's default "highest" (torch 2.10.0 `get_float32_matmul_precision()` = "highest", measured). |
| (major) `G3_JOBS=... bash polaris/submit.sh` never reaches the job | yes: `submit.sh` passes no `-V` / `-v`, and the repo's own notes say jobs get no environment (`polaris/repo_dir.sh:5`). | `submit.sh` writes the set knobs (`G3_JOBS`, `G3_ARGS`, `G3_DATA_FILE`, `SMOKE_ARGS`, `SMOKE_XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR`) as `export` lines (`printf %q`) to `logs/<script>.<time>.<pid>.env` and passes only `-v SUBMIT_ENV_FILE=<path>`; 16 / 17 source it after `env.sh` and print it. A file, not a `-v` list: `qsub -v` splits at commas, and a comma value needs nested quotes (OpenPBS test `pbs_passing_environment_variable.py`). `S6b_submit_passthrough.sh` with a fake `qsub` (only `-v` variables reach the job; the real `.pbs` lines run): `G3_JOBS` with commas, `G3_ARGS` with spaces, an empty `SMOKE_XLA_FLAGS` (clears `XLA_FLAGS`) and a value with `=`, `,` and `'` arrive intact; no knobs -> defaults. Control: `submit.sh` of `8380d8a` -> `G3_JOBS` UNSET in the job (`S6b_evidence/submit_passthrough.out`). Not run on PBS itself. |
| (major) No seed test | yes (reviewer mutants; at seed 42 a dropped `set_random_seed` is invisible because `CLICDataset` seeds 42) | `test_seed_sets_the_model_init_and_the_train_order` (seeds 7, 7, 8: equal init and epoch-0 order for one seed, other init and order for another) and `test_command_line_options_reach_the_data_and_the_trainer` (`--seed 7` reaches both sources and the `Trainer` settings; control: no option -> the YAML's 42). K32 (no `set_random_seed`), K33 (`ClicSource` seed fixed at 42), K34 (CLI data seed fixed) killed. |
| Cache assertion load-sensitive | yes (reasoning: JAX persists only programs compiling >= 1 s) | `cache_misses == 0` dropped; asserts hits >= 3 and >= 2 `jit__train_impl` + >= 1 `jit__val_impl` entries in the cache. The S6b suite passed with the G3 sweep and the smoke running beside it. |
| Epoch checkpoint after an epoch without validation | yes (Lightning 2.5.2 `_should_save_on_train_epoch_end` returns False when `check_val_every_n_epoch != 1`) | `_on_epoch_end` saves the epoch file only when this epoch validated or `check_val_every_n_epoch == 1` (deviation 9). The cadence test now runs 3 epochs at val every 2: one epoch file, `epoch-001-step-000000008.npz` (the old code also wrote `epoch-002-step-000000012.npz`; K27 killed). |
| No `max_steps` test | yes | `test_max_steps_stops_inside_an_epoch` (`max_steps` 5, 4 steps per epoch): global step 5, `total_steps` 5, position (1, 2), rows 1-5, LR = the 5-step schedule (control: the 12-step one differs), only the epoch-0 val row, `last` = step 5. K29 (ignored) killed; K30 (no stop inside the epoch) never ends: killed by the 900 s timeout (the unmutated test takes ~90 s). |
| No `limit_val_batches` test | yes | `test_limit_val_batches_validates_the_first_batches_only`: val/loss = the first batch's task loss (4 real events; control: all 3 batches differ); the CLI test checks `--limit-val-batches 1` reaches the settings. K31, K35 killed. |
| Bitwidth min / max never compared | yes | The cadence test recomputes `BitwidthMonitor`'s reduction in numpy (min of mins, max of maxes exact; mean of means within 2x the measured 1.915e-6 relative) for the step-12 train row (final state) and the epoch-1 val row (its checkpoint); some min < max (else vacuous). K28 (min -> max) killed. |
| Stale gate row, old push hash in the docs | yes | Gate row rewritten; plan header and KERAS_NATIVE.md say `origin/keras-jax` = `826bc0a` (after R5: it has the org merge and masked-linformer, not S5 / S6). The historical 2026-10-15 status line keeps `197ede2`. |

Tiny local re-runs (CPU, dummy data) on this code: `16_jax_smoke.py --tiny` with the determinism flag SMOKE-PASS, 8 of
8 checks, resume bitwise; `17_g3_step.py --sweep --tiny --iters 40` all 8 jobs, no error, b32 ratio 8.42, matching
share 0.214 (0.137 in fix round 1, 0.634 with the test suite running beside it: the tiny CPU share follows machine load
and is not G3 evidence).

## Test runs and mutants

- `tests/keras_native/test_loop.py` + `test_cli.py`: 25 + 4 tests, all passed in the full-suite run at `14a941a`.
  Fix round at `d104d44`: `test_loop.py` 29, `test_cli.py` 6, with `test_import_hygiene.py` and `test_config.py`: 113
  passed, 0 failed (8 min 27 s). The rest of the native suite was not re-run (the fix round changed only `loop.py`,
  `cli.py`, their tests and the Polaris scripts).
- Fix round 2: `test_loop.py` 32, `test_cli.py` 7, with `test_import_hygiene.py` and `test_config.py`: **117 passed, 0
  failed** (11 min 0 s, with the G3 sweep and the smoke running beside it). Mutants: K0c (no mutant) 4 passed;
  K27-K35 9 / 9 killed (K30 by the 900 s timeout). The rest of the native suite was not re-run (only `loop.py`'s epoch
  checkpoint condition changed in `src/`).
- Fix-round mutants (`S6b_mutations.py`, scratch copies, appended to `S6b_evidence/mutations.log`): K0b (no mutant)
  9 passed; K13-K26 14 / 14 killed (K13-K24 in `test_loop.py`, K25-K26 in `test_cli.py`).
- Mutants (`S6b_mutations.py`, scratch copies, log `S6b_evidence/mutations.log`): K0 (none) 12 passed; K1-K12 12 / 12
  killed (each by the test named in the gate table).
- Full native suite at `14a941a` (`env -u KERAS_BACKEND PYTHONPATH=$PWD/src .venv-jax/bin/python -m pytest tests/keras_native`):
  **788 passed, 4 skipped (requiresdata), 0 failed** (792 collected = 763 before + 29 new) in 27 min 50 s; includes
  `test_import_hygiene.py` over the new `loop` / `cli` modules and the S6a gate files (h5 layout, metrics, loader).
- Hybrid / torch suites: not re-run. The only non-native file changed is `polaris/submit.sh` (two `case` rows) and the
  new polaris scripts; no file under `src/hepattn/{keras,models}` or `tests/{keras,polaris,models,loss,matching}`
  changed (`git diff --stat c0d143e..HEAD`).


## Deviations

1. `val/loss` is the task loss, against S5.md's reading (decision 1, measured).
2. The short val batch's loss comes from a loss-only program on the padded forward (2 ulps from an unpadded forward),
   not from a second model program (decision 2).
3. An epoch uses `(batches // k) * k` micro-batches and `total_steps = (batches // k) * epochs` (min `max_steps`);
   Lightning steps once more on a partial accumulation and uses `ceil`. Polaris: 31075 micro-batches at k = 4 drop 3.
4. No sanity validation; one device (a config with `devices > 1` warns); fp32 only.
5. Train task metrics come from the `training=False` forward of the logged micro-batch (the hybrid uses the training
   forward's outputs); one extra val-program call per logged step, no extra compile.
6. A resume from a step checkpoint written at the last step of an epoch re-runs that epoch's validation and epoch
   checkpoint (deterministic values; the epoch file gets an `-r1` revision if it exists). With `save_last`, `last` points
   at that step file when both fall on one step (Lightning skips the second save too).
7. BetaPID with `warmup_epochs = 0` has no read-out at epoch 0 and holds `init_beta`; the hybrid reads the build-time
   `_ebops` then.
8. The YAML's checkpoint `dirpath` and `MetricsCsvWriter.path` are ignored (decision 13).
9. Stopping at `max_steps` inside an epoch: no validation (as S6a decision 10 reads Lightning); the epoch checkpoint uses
   the last val metrics, or is skipped with a warning when none exists (Lightning saves without the monitor). (fix
   round 2) With `check_val_every_n_epoch != 1` an epoch checkpoint is written only in an epoch that validated, as
   Lightning's `ModelCheckpoint._should_save_on_train_epoch_end` (lightning 2.5.2) decides.
10. Changed one line of S5's `optim.py` (decision 3) and added files outside the listed ones: `polaris/jax_common.py`,
    `tests/keras_native/s6b_utils.py`, `tests/keras_native/s6b_fixtures/`, the `S6b_*` evidence files.
11. WORKLOG.hgq.md is in the `hepattn` main tree, which must not be modified; not updated.
12. (fix round) `metrics.csv` / `steps.csv` (`CsvLog`): a row with new keys rewrites the file with the new columns
    appended; the hybrid `MetricsCsvWriter` fixes the header at its first row. Lightning's `callback_metrics` holds
    the train values at every validation, the driver only after a logged step, so a fixed header could drop them.
13. (fix round) A process that resumes into its own run directory drops the CSV rows after the checkpoint (steps >
    the checkpoint's step; epochs >= its data-position epoch) and removes `ckpts/.*.tmp`; it writes
    `config-r<n>.yaml` / `run-r<n>.json` next to the first run's files. A resume with another seed is refused.
14. (fix round 2) `polaris/submit.sh` passes the job knobs through a file (`-v SUBMIT_ENV_FILE`), not through
    `qsub -v` lists or `-V`. A bare `qsub` of 16 / 17 runs the defaults.
15. (fix round 2) G3 has one more diagnostic job, `hybrid-tf32:16` (the hybrid at its CLI default "high"), in the
    second submission.

## Open issues

1. **Polaris**: run `bash polaris/submit.sh 16_jax_smoke.pbs`, `bash polaris/submit.sh 17_g3_step.pbs` (the gate) and
   `G3_JOBS=jax-g1:32,jax-precision:32,hybrid-tf32:16 bash polaris/submit.sh 17_g3_step.pbs` (the diagnostics); the
   job log prints the knobs it got. Record G3 (ratio, matching share, EBOPs share at b32, G1 gap with profiles if the
   donating step is slower, TF32 effect on both sides) in this log and the plan; until then G1 is OPEN and
   PeriodicEBOPs stays (S5). Needs a pushed branch with S5 / S6 (`origin/keras-jax` is `826bc0a`; KERAS_NATIVE.md).
2. The compilation cache saves the XLA compile only; tracing (CPU tiny: ~6.9 s of the 14.4 s first step) and the eager
   build (21.7 s) remain per process. Measure both on the A100 (`summary.json` `build_s`, `first_step_s`); if the
   build dominates, cache the eager ops (`jax_persistent_cache_min_compile_time_secs=0`) or build from shapes.
3. The driver resume is tested with AdamW, k = 2, BetaPID (the generated config's choices). Lion, k = 1 and the
   BetaScheduler path through a resume are not (S56 open issue 2, partly closed).
4. `ClicMetrics` default: the driver logs the pooled (hybrid) values; per-layer with `--metrics-per-layer`. User to
   confirm (S6a decision 5).
5. G4 data order (S6a open issue) and the hybrid BetaPID checkpoint state (S5 open issue 4) are unchanged.
6. Not done: `test` writes the h5 but not the ROOT file of `on_test_end` (S6a deviation 5).
