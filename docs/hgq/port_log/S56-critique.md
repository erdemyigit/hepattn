# Completeness critique of S5 + S6 (2026-10-20)

Scope: plan [KERAS_JAX_PLAN.md](../KERAS_JAX_PLAN.md) section 4, the S5 and S6 gates, after S5, S6a, their merge and S6b
([S5.md](S5.md), [S6a.md](S6a.md), [S56-merge.md](S56-merge.md), [S6b.md](S6b.md)). Tree: `keras-jax` at `2e12012`
plus the commits of this critique. A first critic run stopped on a network error and left uncommitted edits; this run
audited them (section 1) and finished the critique.

Mutation runner: [S56_critique_mutations.py](S56_critique_mutations.py) (each mutant on a fresh scratch copy of `src`,
`tests`, `polaris`; never on the worktree). Evidence: `S56_evidence/critique_*`.

## 1. Audit of the interrupted run's edits

| Edit | Verdict | Evidence |
|---|---|---|
| `train_step.py`, `test_train_step.py` comments, S5.md row: the hybrid LOGS `val/loss` = task loss; `aggregate_losses` RETURNS task + `quant_losses()` | correct, committed (`638a83d`) | `models/wrapper.py:62` logs `{stage}/loss` from the task terms; `experiments/clic/lightning_module_hgq.py:138-142` calls `super().aggregate_losses` first, then logs `quant_ebops_loss` and returns the sum. Probe `S6b_evidence/val_loss_probe.out` (val/loss 5 = task, returned 12). The CLI run of section 4: `val/loss` 21.678410 = `val/final_loss` + `val/layer_0_loss`, `val/quant_ebops_loss` 0.00877 logged apart. |
| KERAS_NATIVE.md CLI option list, timing | correct; extended (see section 5) | `fit --help` from a clean shell lists exactly `--config`, `--run-dir` and the 21 documented overrides. |
| port_log/README.md carried-gate rows closed by S5 | correct, except the R4b row's spread "0.25-0.33": the first run's own scratch re-runs (4 per tree) gave run means 0.25-0.28 and rows 0.225-0.303; replaced by this run's measurement (section 3) | S3 row: `7f3747c` + `tests/keras/test_quant_losses_dedupe.py` (X16 kills it). S4 row: `test_maskformer_loss_from_config_reads_the_tasks_and_the_matcher` (records `n_jobs` 16), `test_steps_request_highest_matmul_precision`. merge row: `test_readout_equals_the_torch_backend_value` checks every leaf (`per_leaf` max abs < 1). New S6a row: `test_data_real.py` is `requiresdata` (4 skipped locally). New S5/S6b row: S6b open issues. |
| `S56_critique_mutations.py` | kept, extended (X18), lint-clean, summary filter fixed | The old filter took warning lines ("was passed an input") as pytest summaries. |
| Worktree mutation check | clean | `git diff` at the start held only the 5 files above (21 insertions, 14 deletions); no executable line changed. |

## 2. Gates of plan section 4: test, mutation, result

Every gate below has a passing test in the full-suite runs of section 6. "Mutant" = a mutation of THIS critique
(`critique_mutations.log`): 18 run, 18 killed, each by the gate's own assertion (failure messages checked). The
unmutated rows: X0_jax 32 passed, X0_loop 6 passed, X0_torch 8 passed.

| Stage | Gate | Test | Mutant (killed by) |
|---|---|---|---|
| S5 | 30-step port: loss falls, quantizers move | `test_train_step::test_training_port_loss_falls_and_quantizers_move` | X1 quantizer gradients zeroed ("no quantizer variable changed") |
| S5 | every trainable variable gets a nonzero gradient | `test_every_trainable_variable_gets_a_finite_nonzero_gradient` | X2 register tokens detached (`register_tokens` listed) |
| S5 | penalty check rel <= 1e-5 | `test_penalty_check_against_an_eager_reference` | X3 penalty x 0.9999 (rel 9.9e-5) |
| S5 | EBOPs read-out = torch value; `_ebops` control | `test_ebops::test_readout_equals_the_torch_backend_value` (+ `test_jax_stored_ebops_saturate_uint32_control`) | X4 read-out x (1 + 2e-6) (diff 5.5e8 > 1.2e-6 x 2.12e14) |
| S5 | beta change: no retrace, penalty scales | `test_beta_change_keeps_the_executable_and_scales_the_penalty` | X5 `set_beta` returns a tuple (trace count 2) |
| S5 | PeriodicEBOPs active vs skipped | `test_ebops::test_periodic_ebops_reproduces_the_hybrid_test`, `test_train_step::test_periodic_ebops_skipped_step_has_no_penalty`; driver: `test_loop::test_periodic_ebops_follows_the_optimizer_step` | X18 switch ignored (skipped penalty 655.6 / 224.2, not 0); X6 period shifted by one step |
| S5 | BetaPID control law | `test_beta_control::test_controller_equals_hgq2_beta_pid` (hybrid: `tests/keras/test_beta_pid.py`) | X7 half damping below target (2 cases, 8 of 10 betas differ) |
| S5 | OneCycle = torch at 20 steps | `test_optim::test_onecycle_equals_torch_at_20_steps` | X8 no floor on micro-step / k (k = 4: rel 0.72) |
| S5 | k micro-batches = one k-times batch | `test_accumulation_averages_k_micro_batches_like_one_k_times_batch` | X9 accumulation off |
| S5 | weight-decay exclusion by `quantizer` path | `test_weight_decay_excludes_exactly_the_quantizer_paths[AdamW, Lion]` | X10 integer bits `/i` decayed |
| S5 | **G3 (Polaris)** | `polaris/17_g3_step.pbs` | not run (no GPU); tiny local sweep in section 4 |
| S6 | save, load, step = uninterrupted run (CPU bitwise) | `test_loop::test_resume_continues_bit_for_bit` (+ `test_resume_cross_lane.py`, `test_cli.py`) | X11 last EBOPs reading not restored (3 of 4 positions, 766 arrays differ) |
| S6 | traces over an epoch: <= 2 train, <= 1 eval / predict | `test_loop::test_trace_counts_over_the_run` | X12 every val batch through the tail program (tail 2) |
| S6 | prediction h5 = `PflowPredictionWriter` | `test_predict_writer::test_h5_layout_and_values_equal_the_torch_writer` | X13 half precision |
| S6 | metrics = torchmetrics / Lightning | `test_metrics::test_epoch_reduction_over_unequal_batches_equals_lightning`, `test_weighted_mean_is_lightnings_epoch_reduction` | X14 epoch mean per batch, not per event |
| S6 | YAML loader reads `pflow_hgq.yaml` and the generated config | `test_config::test_loader_reads_every_form` (6 forms) | X15 `gradient_clip_val` dropped (6 of 6) |
| S5 | hybrid `quant_losses` dedupe | `tests/keras/test_quant_losses_dedupe.py` | X16 walk without dedupe (211 vs 183; 4 of 5 fail) |
| S5 | provisional beta setpoints | `tests/polaris/test_beta_setpoints.py` | X17 `INIT_BETA` 9.6e-11 |

## 3. Provisional beta setpoints: re-measured spread

`S5_make_batches.py` + `S5_grad_ratio.py`, 4 fresh processes per tree (new `2e12012`, old `37ff127`), configs generated
by each tree's `03_make_config.py` (`critique_beta_rerun.json`). Deterministic quantities reproduce S5 exactly (EBOPs at
init 2.509e11 / 2.512e11 / 2.512e11; old `|grad_q penalty|` 8.408e7). The keras init is not seeded, so the task
gradients change per process, and so does the factor:

| Run | s_q(old) / s_q(new) per batch | mean |
|---|---|---|
| 1 | 0.182, 0.214, 0.216 | 0.204 |
| 2 | 0.273, 0.317, 0.332 | 0.307 |
| 3 | 0.258, 0.324, 0.304 | 0.295 |
| 4 | 0.217, 0.263, 0.266 | 0.249 |

Rows 0.182-0.332, mean 0.264: `INIT_BETA` 6.9e-11 (4.7e-11 to 8.6e-11). S5's 0.333 (8.7e-11) is above all 4 run means.
The default is kept (the planned Polaris scan brackets it 4x each way); recorded in [BETA_SETPOINTS.md](../BETA_SETPOINTS.md)
and the README R4b row. The dedupe fix and the setpoints are tested (X16, X17 above) and documented (BETA_SETPOINTS.md,
S5.md, `keras/maskformer.py` `quant_losses` docstring).

## 4. End-to-end runs

- **CLI, clean shell, as KERAS_NATIVE.md says** (`env -i HOME PATH`, then `env -u KERAS_BACKEND PYTHONPATH=$PWD/src
  .venv-jax/bin/python -m hepattn.experiments.clic.main_jax fit --config tests/keras_native/s6b_fixtures/tiny_polaris.yaml
  --run-dir <dir> --dummy-data`): exit 0 in 72 s (build 33 s, fit 32 s; the JAX cache in `~/.cache/hepattn/jax` was warm:
  4 hits). 3 epochs, 12 steps; `train/loss` 23.36 -> 21.30, `val/loss` 22.33 -> 21.73 -> 21.68; `train/bits_b_mean`
  moves; traces train 2, val 1, tail 1, bits 1; checkpoints, `steps.csv`, `metrics.csv`, `summary.json`, `best.json`.
  Then `main_jax test --ckpt .../epoch-002-step-000000012.npz --dummy-data`: exit 0 (49 s), `__test.h5` with groups
  `events`, `incidence`, `object_class`, ..., `test_metrics.json`. Log: `critique_cli.log`.
- **Polaris one-time install lines.** A fresh venv from the ONE-TIME SETUP block of `16_jax_smoke.pbs` (identical in
  `17_g3_step.pbs`), with `jax==0.11.2` in place of `jax[cuda12]==0.11.2` (macOS): resolves and installs 69 packages; the
  pins equal `.venv-jax` (`uv pip list`), and `.venv-jax` equals `keras_native_env.txt` (`uv pip freeze`, no diff). Both
  scripts below ran in that venv, which shows the block is sufficient (pytest is not in it and is not needed).
- **`16_jax_smoke.py --tiny`** (that venv, `XLA_FLAGS=--xla_gpu_deterministic_ops=true` as the .pbs): SMOKE-PASS, 8 of 8
  checks, resume bitwise in 2285 of 2285 arrays (`critique_smoke_tiny.json`).
- **`17_g3_step.py --sweep --tiny --iters 40`** (jax arms in that venv, hybrid arms in `hepattn/.venv`): all 8 jobs, no
  error; hybrid at `torch_matmul_precision` "highest" ("high" for `hybrid-tf32`), jax at "highest"; b32 ratio 7.22, b16
  7.58, `jax-precision` 0 flips (CPU has no TF32) (`critique_g3_tiny.json`). The tiny CPU numbers are not G3 evidence.

## 5. Torch-freedom and documentation

- No module under `src/hepattn/keras_native/` imports torch, lightning or a hepattn module outside the package (grep and
  `test_import_hygiene.py`, which also parses function bodies and `importlib` calls). Torch enters only through
  `experiments/clic/main_jax.py` -> `native_data.py` (`CLICDataset` + `DataLoader`), the documented data-adapter
  exemption (KERAS_NATIVE.md module table, `main_jax.py` docstring).
- Fixed (`5f4dc3d`): KERAS_NATIVE.md listed S2's deleted `linformer.py` and none of the R1-R5 files; its Polaris note
  sent `.venv-jax` to the plan 5.1 line, which has no torch, so `main_jax` would fail at the data source; it now points to
  the .pbs setup block. Plan Status now names the tail and bit-statistics programs next to "2 train + 1 val".
- Port logs vs code, spot checks: every test the S5 / S6a / S6b gate rows of section 2 name exists and passes; 802
  native tests collected (S6b: 792 at `14a941a` + 10 fix-round tests in `test_loop.py` / `test_cli.py`); the S6b trace
  counts and `val/loss` definition match `loop.py` and the CLI run. Not every number of the stage logs was re-derived.

## 6. Suites (once, at the tree of this critique)

Code content of every run: `src`, `tests`, `polaris` as at `638a83d` (= `2e12012` plus the comment-only edits; the later
commits change docs and the runner only). Machine loaded throughout (the mutants, the beta re-runs and the tiny Polaris
runs ran beside them), so the times are not benchmarks.

| Command | Env | Result (verbatim summary) |
|---|---|---|
| `env -u KERAS_BACKEND PYTHONPATH=$PWD/src .venv-jax/bin/python -m pytest tests/keras_native` | `.venv-jax` | `798 passed, 4 skipped, 4789 warnings in 2977.00s (0:49:36)` (802 collected; the 4 skips are `test_data_real.py`, `requiresdata`) |
| `KERAS_BACKEND=torch PYTHONPATH=$PWD/src ../hepattn/.venv/bin/python -m pytest -m 'not gpu and not requiresdata' tests/keras tests/polaris` | `hepattn/.venv` | `318 passed, 2 skipped, 6 deselected, 61744 warnings in 648.92s (0:10:48)` |
| same env, `pytest -m 'not gpu and not requiresdata' tests/models tests/loss` | `hepattn/.venv` | `519 passed, 173 skipped, 5 deselected, 15 warnings in 21.01s` (= S56-merge) |
| same env, `pytest -m 'not gpu and not requiresdata' tests/matching --junitxml=...` + `R0_matching_gate.py <xml> --max-failures 106` | `hepattn/.venv` | `106 failed, 66 passed, 14 skipped, 17 deselected, 2 xfailed, 23 warnings in 18.22s`; gate `ran=172 failures=106 lap1015_attr=106 other=0`, exit 0 (= R5, S56-merge: the unbuilt `lap1015` only) |

## 7. Open issues

1. **G3, the real-data smoke, the G1 gap, TF32 and the b32 EBOPs share are not measured** (no GPU). Needs a pushed
   branch with S5/S6 (nothing after `826bc0a` is pushed) and the two submissions of `17_g3_step.pbs`.
2. **`INIT_BETA` is init-dependent** (section 3): 0.18-0.33 per batch. Seed the keras init (`keras.utils.set_random_seed`)
   in `S5_grad_ratio.py` and average several seeds, or settle it with the Polaris beta scan.
3. **The tiny G3 EBOPs share can be negative** (`jax-prod` b32: -0.038 here; b16 0.182): it is a difference of two noisy
   medians. On the A100 read it with its spread, not as a single number.
4. Carried: real CLIC batches (`test_data_real.py`, `requiresdata`); S7 (converter, export), S8 (G4).
