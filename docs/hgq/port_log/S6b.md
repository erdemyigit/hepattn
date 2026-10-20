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
| Metrics = torchmetrics | pass (S6a) + wiring | S6a `test_metrics.py` (full-suite run). The driver uses `ClicMetrics(pool_layers=True)` by default (the hybrid's logged values, S6a decision 5; `--metrics-per-layer` for the per-layer values) with `event_valid`. |
| YAML loader reads every form | pass (S6a) + one reader | S6a `test_config.py`; `test_one_optimizer_reader_for_every_form` x6 (decision 3; control: another accumulation is seen). Mutants K11 (keras epsilon 1e-7 in the driver) and K12 (the old null-algorithm check) killed; the old `optim.py` fails `[org saved]` with `NotImplementedError` (measured before the fix). |
| CLI end to end from a clean shell: checkpoints, metrics CSV, resumable state, loss decreases, quantizers move | pass | `test_cli.py` (4 passed): env = `PATH`, `HOME`, `TMPDIR`, `PYTHONPATH` only; `fit` with 2 spawned workers -> 6 step + 3 epoch checkpoints, `steps.csv` 12 rows, `metrics.csv` 3 rows, `summary.json`; mean train loss of the last 4 rows < first 4, val loss epoch 2 < epoch 0, > 10 trainable quantizer arrays differ between steps 2 and 12, `train/bits_b_mean` changes; resume in a new process bitwise with 3 cache hits (control: run 1 had 0); `test` command. `test_run_a_is_non_vacuous`: every state group changes, beta changes after the warm-up. |
| PeriodicEBOPs / BetaPID / BetaScheduler wiring | pass | `test_periodic_ebops_follows_the_optimizer_step` (active rows = steps 1, 3, 5, ...; a micro-step counter would log 0 on every row; on-step HGQ2 term > 5x off-step), `test_beta_pid_steps_each_epoch_on_the_latest_readout` (an independent controller fed the logged read-outs gives the logged betas; control: without a read-out the PID would hold init_beta), `test_beta_scheduler_ramps_at_the_optimizer_step` (control: the next step's ramp differs). Mutants K1 (micro-step period), K9 (ramp one step late) killed. |
| Checkpoint policy, resume checkpoint kept, `total_steps` guard, refusals | pass | `test_checkpoint_policy`, `test_resume_checkpoint_survives_pruning` (top-k 1 keeps the newest and the resume file; K10 killed), `test_stepping_past_total_steps_raises` (K8 killed), `test_restore_refuses_another_schedule_or_model` (control: equal lengths load), `test_non_fp32_precision_is_refused`. |
| Both Polaris scripts run locally at tiny sizes on dummy data (both G3 arms) | pass | `16_jax_smoke.py --tiny`: SMOKE-PASS, all 7 checks true, resume 2285 / 2285 arrays bitwise, 230 quantizer arrays moved, val loss 22.196 -> 21.912 -> 21.835, resume run 3 cache hits (`S6b_evidence/smoke_tiny.json`). `17_g3_step.py --sweep --tiny --iters 40`: all 7 jobs (jax-prod 32 / 16, jax-g1, jax-precision, jax-match, hybrid 32 / 16) ran with no error and `verdict()` filled every field (`S6b_evidence/g3_tiny.json`; both runs on the committed scripts). The tiny CPU numbers are not G3 evidence (CPU ignores the matmul precision, 1 + 1 layers, dim 16; e.g. its matching share 0.169 and ratio 8.0 say nothing about the A100). |
| G3 on the A100 (>= 1.8x the hybrid in-run step; matching <= 15%; G1 gap; TF32; EBOPs share at b32) | **deferred to Polaris** | `bash polaris/submit.sh 17_g3_step.pbs`; no Polaris access in this session. G1 stays OPEN. |
| Real-data smoke and resume on the A100 | **deferred to Polaris** | `bash polaris/submit.sh 16_jax_smoke.pbs` (S6a open issue 3: GPU resume within `--rtol` 1e-3, bitwise count reported). |
| Lint | pass | `prek run --files` on every changed file. |

## Test runs and mutants

- `tests/keras_native/test_loop.py` + `test_cli.py`: 25 + 4 tests, all passed in the full-suite run at `14a941a`.
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
   the last val metrics, or is skipped with a warning when none exists (Lightning saves without the monitor).
10. Changed one line of S5's `optim.py` (decision 3) and added files outside the listed ones: `polaris/jax_common.py`,
    `tests/keras_native/s6b_utils.py`, `tests/keras_native/s6b_fixtures/`, the `S6b_*` evidence files.
11. WORKLOG.hgq.md is in the `hepattn` main tree, which must not be modified; not updated.

## Open issues

1. **Polaris**: run `16_jax_smoke.pbs` and `17_g3_step.pbs`; record G3 (ratio, matching share, EBOPs share at b32,
   G1 gap with profiles if the donating step is slower, TF32 effect) in this log and the plan; until then G1 is OPEN and
   PeriodicEBOPs stays (S5).
2. The compilation cache saves the XLA compile only; tracing (CPU tiny: ~6.9 s of the 14.4 s first step) and the eager
   build (21.7 s) remain per process. Measure both on the A100 (`summary.json` `build_s`, `first_step_s`); if the
   build dominates, cache the eager ops (`jax_persistent_cache_min_compile_time_secs=0`) or build from shapes.
3. The driver resume is tested with AdamW, k = 2, BetaPID (the generated config's choices). Lion, k = 1 and the
   BetaScheduler path through a resume are not (S56 open issue 2, partly closed).
4. `ClicMetrics` default: the driver logs the pooled (hybrid) values; per-layer with `--metrics-per-layer`. User to
   confirm (S6a decision 5).
5. G4 data order (S6a open issue) and the hybrid BetaPID checkpoint state (S5 open issue 4) are unchanged.
6. Not done: `test` writes the h5 but not the ROOT file of `on_test_end` (S6a deviation 5).
