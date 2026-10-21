# PROF: where the native driver's ~0.3 s per micro-batch goes

Branch `prof/loop` (from `s7/export` 111aede), worktree `hepattn-prof`. Question: the S6b smoke (Polaris job 7763621,
A100, real data, batch 32 x accumulation 4) ran at ~0.49 s (run A) and ~0.465 s (run B) per micro-batch "end to end".
The compiled step alone (G3, job 7763879, fp32 'highest', host matching included) is 161.6 ms at batch 32. So
~0.30 s per micro-batch is outside the step. Neither job's logs are in a local tree. The numbers come from the task
brief and from the main worktree's `WORKLOG.hgq.md` (lines 232-235 and 281-287). The worklog also records: run A
reached 297 optimizer steps, first step 218 s (run A) and 61 s (run B, cache hits), model build ~110 s, and the G3
matching callback 17.2 ms.

## Status (2026-10-20, end of the session; review 2 fix 2026-10-21)

- The profiler is finished and verified. Profiler off is bitwise identical to the S7 baseline. The phases sum to the
  wall time (unattributed < 0.04% in every local run). All 23 mutants are killed (below).
- The CPU measurements at production dims are done (below). The host-side costs that recur at the smoke cadence sum
  to about 0.04-0.07 s per micro-batch. That does not explain 0.30 s, unless the Polaris file system is much slower
  than this Mac's SSD.
- New lead H0: the "end to end" number may include one-time costs. From the worklog: (build ~110 s + first step
  218 s) / 1188 micro-batches = 0.28 s per micro-batch in run A. That is almost all of the gap, IF the 0.46-0.49
  figure is the process wall time divided by the micro-batches. No GPU time is needed to decide this:
  `18_loop_profile.py smoke-timing` on the smoke's JSON (Polaris commands, step 1).
- H7 (determinism flag) and the other device-side hypotheses need the A100 run. `18_loop_profile` runs the H7 pairs
  first, then the perturbation pair. Not submitted. The branch is not pushed.
- Review fix (same day, "Review fix" below): the derived numbers now exclude one-time costs (steady window), the
  sync-mode idle fraction counts every device phase, the perturbation compares steady log windows, the budget counts
  from the job start, and the derived numbers, the variant commands and the budget have tests (26 more mutants
  killed; 49 of 49 in all).
- Review 2 fix (2026-10-21, "Review 2 fix" below): the steady window still held the epoch-end `last` checkpoint
  (40% of the tiny `pure*` steady windows; 1.1 GB per save at production dims), setup and the train end. A loop
  window (`loop_wall_s`, `s_per_micro_batch_loop`, `device_idle_frac_loop`) leaves them out; `overhead_vs_g3_ms` and
  the H1 / H5 / H7 rules use it. The driver's data positions, the trace window and `--profile-step-ref-ms` are now
  tested.

## The profiler

`hepattn.keras_native.profile`. Opt-in: `main_jax fit --profile-phases <csv>`; library:
`Trainer(..., profiler=PhaseProfiler())`.

- Phases are nested context managers. Each phase records its path (`epoch_end/val/forward`), start, duration,
  exclusive time (duration minus child phases) and the data position. `summary()` adds up the exclusive times; the
  rest of the wall time is `unattributed` (Python between phases). A phase that is not opened shows up there.
- Phases in the driver: `data` (`data/loader` = `next()` on the torch loader, `data/numpy` = conversion, casts,
  padding and signature check, `data/device_put`), `train/beta`, `train/step` (dispatch), `train/first_step_wait`,
  `train/log` (`device_wait`, `device_get`, `ebops`, `bits`, `metrics_forward`, `metrics_device`, `metrics_host`,
  `csv`), `train/ckpt` and `epoch_end/ckpt_{epoch,last}` (`device_wait`, `meta`, `host_copy`, `npz_write`, `sha256`,
  `sidecar`), `epoch_end/val` (`data/*`, `forward`, `device`, `tail`, `device_get`, `metrics`, `metrics_compute`,
  `bits`), `epoch_start`, `epoch_end/drain`, `train_end/*`, `setup`, `summary`.
- A jitted call that traced is relabelled `<path>+compile`. `summary()["groups_s"]` puts every compile phase (and
  `train/first_step_wait`) in group `compile`, so the other groups are steady-state costs.
  `s_per_micro_batch_excl_compile` = (wall minus the compile phases) / micro-batches.
- Steady window: `one_time_s` = the compile phases, the first train batch (loader worker start: 10-20 s with 4-8
  workers on real events) and the first validation batch (validation loader start). `steady_wall_s` = wall minus
  these; `steady_micro_batches` = micro-batches minus the first (its device time is in `train/first_step_wait`);
  `s_per_micro_batch_steady` = their ratio (the end-to-end rate at the run's cadence, without one-time costs).
- Loop window (review 2): the steady window without the non-compile exclusive time of the phases outside the train
  loop (`profile.is_outside_loop`: `setup`, `epoch_start`, `epoch_end/*` except `epoch_end/drain`, `train_end/*`,
  `summary`; `outside_loop_s`). It keeps the micro-batches, the log rows and step checkpoints at their cadence, the
  epoch drain (device time of steps already queued) and the Python between phases. `s_per_micro_batch_loop` =
  `loop_wall_s / steady_micro_batches` is the rate to compare with a device step; 18_loop_profile's
  `overhead_vs_g3_ms` uses it (`overhead_steady_vs_g3_ms` uses the steady rate).
- The matching callback runs in another thread. Its wall time per call is a concurrent counter (`match_callback`),
  not a phase. Only the thread that activated the profiler records phases.
- Off: `profile.NULL`. `phase()` returns one shared `nullcontext`; `wait()` / `block()` return their argument. No clock
  read, no added sync.
- `async` mode adds `block_until_ready` only where the next statement reads the value on the host anyway (before each
  checkpoint save, `epoch_end/drain`, `train_end/drain`, before the host metrics of a logged step and of each
  validation batch). `train/step` is the dispatch time; it grows only under back-pressure. On a short validation
  batch the wait comes before the dispatch of its `tail` program (once per validation).
- `sync` mode (`--profile-sync`) also blocks after every train step (`train/device`) and every device transfer
  (`data/device_put_wait`). `device_idle_frac = 1 - busy / steady_wall_s`, with `busy` the exclusive time of every
  non-compile phase from a device dispatch to its result (`profile.DEVICE_PHASES`: train step + `train/device`, the
  log row's `bits`, `metrics_forward`, `metrics_device`, `device_wait`, `device_get`, validation's `forward`,
  `device`, `tail`, `device_get`, the checkpoint's `device_wait`, `meta`, `host_copy`, the drains). In sync mode
  every dispatch is followed by its wait inside one of these, so `busy` is an upper bound on the device's busy time
  and the idle fraction a lower bound. Transfer waits are idle compute (not in `busy`). `device_idle_frac_wall` is the
  same over the whole wall (one-time costs in it: in a short fit it is dominated by them, e.g. 0.991 in the tiny
  `pure_sync` run where the compiles were 12.8 of 13.1 s). `async` mode: `device_idle_frac_est = 1 -
  steady_micro_batches * step_ref / steady_wall_s` with `--profile-step-ref-ms`. `device_idle_frac_loop` and
  `device_idle_frac_est_loop` are the same over the loop window, with `busy` restricted to the phases inside it
  (`device_busy_loop_s`); these are the numbers for H1 / H5.
- `--jax-trace DIR --jax-trace-window A:B`: a `jax.profiler` trace of micro-batches [A, B).
- `--prefetch N` (`data.prefetch`): the train batches are read ahead in a thread; `data` is then the queue wait.
- Outputs: `<csv>` (one row per phase), `<csv>.json` (summary), `summary.json["profile"]`, a table on stdout.

## Verification (first session, before the review fixes; current counts: "Review 2 fix")

| check | result | evidence |
|---|---|---|
| `test_profile.py` (12 tests) | 12 passed | `PROF_evidence/test_runs.txt`, mutation baseline p00 |
| tests of the edited files: test_loop, test_cli, test_data, test_checkpoint, test_train_step, test_resume_cross_lane, test_profile | 139 passed, 0 failed, 0 skipped (2227 s) | `test_runs.txt` |
| full `tests/keras_native` (.venv-jax) | 842 passed, 16 skipped (12: hls4ml not installed; 4: real data not found) | `test_runs.txt` |
| the 16 skips: `test_data_real.py` with `CLIC_DATA_DIR=<main checkout>/data/clic`; `test_export.py` in hepattn-s7/.venv-jax (hls4ml 1.3.0) | 4 passed; 18 passed, 5 skipped (csim unavailable on darwin) | `test_runs.txt` |
| `tests/polaris/test_loop_profile_script.py` (new, 2 tests) | 2 passed | mutation baseline p00b |
| profiler off vs the S7 baseline (111aede) | bitwise equal | below |
| ruff (preview) on `src/hepattn/keras_native`, `polaris/18_loop_profile.py`, `tests/keras_native`, `tests/polaris`, `PROF_mutations.py` | clean. The 4 docstring findings are fixed: B903 (`_Raised` is a dataclass), TRY300 (`else`), DOC502 (`fit`), DOC402 (`active`) | |

**Bitwise baseline.** `PROF_evidence/baseline_cmp.sh` runs the same tiny CLI fit (`tiny_polaris.yaml`, dummy
events, 2 epochs, a log row and a step checkpoint at every optimizer step, all kept) with the S7 sources
(`git archive 111aede src`) and with this branch in three modes: profiler off, `--profile-phases` (async), and
`--profile-phases --profile-sync --prefetch 2`. `baseline_cmp2.py` compares every array of the 6 checkpoints
(13710 state arrays) byte for byte, the checkpoint metadata without wall-clock keys, and the CSV rows without `time/`
columns. All three runs are EQUAL to the S7 run. The only raw metadata difference is `time/ms_per_micro_batch` (a NaN
`pid_ebops` compares unequal leaf by leaf, but it is the same value). Controls (`baseline_cmp2.out`): a one-ulp change
of one kernel element gives DIFFERENT, and a changed `beta` in one checkpoint's metadata gives DIFFERENT.

**Mutations** (`PROF_mutations.py`: a scratch copy of the worktree, one snippet per mutant; the worktree is never
written). The baselines p00 (test_profile.py) and p00b (test_loop_profile_script.py) pass. All 23 mutants are killed:

| mutant | what it breaks | killed by |
|---|---|---|
| p01 | `NULL.wait` blocks (profiler off adds syncs) | test_syncs_added_by_each_mode |
| p02 | the profiled matching callback reorders its costs (the profiler changes the run) | test_profiler_and_prefetch_do_not_change_the_run |
| p03 / p04 / p05 | prefetch: LIFO order / swallowed error / close does not wait for the thread | test_prefetch_order_errors_and_shutdown |
| p06 | exclusive time = duration | test_exclusive_times_on_a_fake_clock |
| p07 | the summary adds durations, not exclusive times | test_phases_sum_to_the_wall_time[async] |
| p08 | `current()` ignores the thread | test_null_profiler_is_inert |
| p09 | async mode blocks every step | test_phase_counts_follow_the_cadence |
| p10 / p11 / p12 / p14 | `+compile` label lost / `train/ckpt`, `data`, `npz_write` phase missing | test_phase_counts_follow_the_cadence |
| p13 / p15 | drain wait missing / sync mode misses the transfer waits | test_syncs_added_by_each_mode |
| p16 / p17 | the CLI ignores `--profile-phases` / `--prefetch` | test_fit_command_writes_the_profile |
| p18 | compile phases counted in the steady groups | test_phases_sum_to_the_wall_time[async] |
| p19 / p20 | smoke-timing: run B counted from step 0 / first step not a one-time cost | test_smoke_timing_splits_one_time_costs |
| p21 / p22 / p23 | an unprofiled variant gets a compile-including rate / perturbation ratio inverted / perturbation valid with a cold compile (review fix: p22 / p23 now target the log-window ratio) | test_unprofiled_variant_and_steady_overhead, test_perturbation_compares_steady_log_windows (review fix) |

Evidence: `mutations_p00-p17.json`, `mutations_rerun_p00_p07_p18.json` (after the `compile` group change),
`mutations_p00b_p19-p23.json`.

## Review fix (2026-10-20)

Each review finding was checked against the code first; all were confirmed. What changed:

| finding | fix |
|---|---|
| the derived numbers (`s_per_micro_batch_excl_compile`, steady medians, idle fractions, compile incl. `train/first_step_wait`) had no test: 5 reviewer mutants survived | `test_derived_numbers_on_a_fake_clock`: a 4-micro-batch run on a fake clock in quarter seconds; every derived number is checked against hand arithmetic. Mutants m01-m12 (the reviewer's m1, m2, m3, m5, m6 and 6 more) are killed |
| the idle fraction was over the whole wall (compiles included: 0.991 in the tiny `pure_sync`) | `device_idle_frac` is over `steady_wall_s` (no compiles, no first train / validation batch); the raw value is `device_idle_frac_wall` |
| sync-mode `busy` left out validation's `device`, the log row's forward / wait and the checkpoint read-outs | `busy` = every phase in `profile.DEVICE_PHASES` (leaf names); compile phases never match (their leaf ends in `+compile`) |
| the perturbation `fit_s(pure) / fit_s(pure_noprof)` was dominated by compiles and cache loads | new pair `pure_log` / `pure_noprof` (log every `--pert-log-every` 8 optimizer steps); ratio of the median log windows after the first; `valid` needs 3 windows after the first in each; `fit_s_ratio` kept for reference |
| H5 decision rule could not separate a sync from back-pressure | H5 bullet rewritten (steady rate, since review 2 the loop rate, vs device step, dispatch times by micro-batch, trace gaps) |
| evidence files ignored by `.gitignore` | `git add -f` of the 5 files |
| `overhead_vs_g3_ms` included the loader / validation start; `prod` described as the production cadence | overhead uses `s_per_micro_batch_steady` (review 2: the loop rate); `prod` described as `pure` + validation + epoch checkpoint at 320 micro-batches |
| the identity assert in `test_phases_sum_to_the_wall_time` | removed (comment says why) |
| `pure_noprof` 9th of 12; the documented second job had no `pure` | the perturbation pair runs right after the H7 pairs; `report` merges JSONs, so the pair can come from two jobs |
| budget clock started after the prelude and ignored the next variant's length; a killed job has no `table` | `PROF_JOB_START` (pbs) starts the clock; a variant starts only if elapsed + the longest variant so far <= `--budget-min` 55; `report` rebuilds the table from a partial JSON |
| variant commands, the budget and `--profile-sync` had no test (5 reviewer mutants survived) | `test_run_fit_builds_each_variant` (argv and `XLA_FLAGS` of every variant through a fake `subprocess.run`), `test_budget_counts_from_the_job_start_and_the_longest_variant` (fake clock), `test_cli_profiler_flags` (`cli.profilers`, factored out of `cli.fit`) |
| zsh-only check line; `WORK_ROOT` not set | `bash -c` check; `WORK_ROOT` from the keras-jax `env.sh` |
| no cache state in the table; which H7 pair decides | `misses` column; `pure` vs `pure_det` decides, the smoke pair supports |
| ~10-15 GB of checkpoints per job | each fit's `ckpts/` is deleted after the fit (`ckpt_bytes_deleted` recorded; `--keep-ckpts`); cleanup step 4 |

The new budget test caught a bug in the first draft of this fix: `--job-start 0` (falsy) fell back to "now"; the default is now None.

Verification of the fix (`PROF_evidence/test_runs.txt`, `mutations_review_all.json`, `baseline_cmp2_reviewfix.out`,
the end of `lp_tiny_table.txt`):

- Full `tests/keras_native`: 844 passed, 16 skipped (hls4ml / real data, as before; 842 + the 2 new tests). It ran
  before the last edit of `profile.py` (the `is_compile` filter in `busy` removed as redundant); `test_profile.py` on
  the final code is the mutation baseline p00: 14 passed. `tests/polaris/test_loop_profile_script.py`: 6 passed.
- Mutations on the final code, all in one run: the baselines p00, p00b, p00c pass and all 49 mutants are killed:
  p01-p23 (p21-p23 rewritten for the new perturbation) and the 26 new ones below.
- Profiler off / async / sync + prefetch 2 are still bitwise EQUAL to the S7 run (13710 state arrays in 6
  checkpoints, metadata, CSVs); the controls ctl_bit and ctl_meta give DIFFERENT.
- `--tiny` run of smoke, smoke_nodet, pure, pure_det, pure_log, pure_noprof, pure_sync, prod, loader0 (fresh run root,
  9 min 49 s): all completed. `pure_sync`: `device_idle_frac` 0.531 over the 0.339 s steady window,
  `device_idle_frac_wall` 0.991 (compiles 17.64 of 17.98 s). Perturbation: 3 windows after the first in each run,
  13.4 / 14.6 ms per micro-batch = 0.922 (tiny CPU, 4 micro-batches per window: noise; it checks the plumbing). Every
  fit's `ckpts/` was deleted (4.5-13.6 MB each here). `report` of two partial JSONs (all variants but `pure_noprof`,
  without `table` / `perturbation`; `pure_noprof` alone) gives the same perturbation. Cache: smoke and smoke_nodet
  missed 4 programs each (different `XLA_FLAGS`), every later fit missed 0.

| mutant | what it breaks | killed by |
|---|---|---|
| m01 / m11 / m12 | `forward` + `tail` / `host_copy` / `metrics_device` left out of `busy` | test_derived_numbers_on_a_fake_clock |
| m02 | steady medians include the first optimizer step (`>=`) | same |
| m03 | idle estimate without the micro-batch factor | same |
| m05 | `s_per_micro_batch_excl_compile` = wall / micro | same |
| m06 | `train/first_step_wait` not a compile | same |
| m07 / m08 / m09 | idle over the whole wall / first validation batch counted as steady / steady count includes the first micro-batch | same |
| m10 | compile phases counted as busy (suffix stripped) | same |
| c01 | `--profile-sync` ignored by the CLI | test_cli_profiler_flags |
| k01 / k02 / k04 | determinism flag never set / `smoke_nodet` gets it / `pure_noprof` profiled | test_run_fit_builds_each_variant |
| s06 / s07 | checkpoints not deleted / perturbation pair without log rows | same |
| k03 / s01 / s02 | budget check off / ignores the next variant's length / counts from the script start | test_budget_counts_from_the_job_start_and_the_longest_variant |
| s03 / s04 | perturbation from whole fits / valid with too few windows | test_perturbation_compares_steady_log_windows |
| s05 / s10 | overhead from the compile-only rate / no cache misses in the table | test_unprofiled_variant_and_steady_overhead |
| s08 / s09 | `report` keeps the earlier file's variant / lists a variant that ran later as skipped | test_report_merges_partial_results |


## Review 2 fix (2026-10-21)

Each finding was checked first. Two majors (1 and 6) are the same issue. All were confirmed, with one correction to a
suggested assertion (below).

| finding | check | fix |
|---|---|---|
| the steady window still held the epoch-end `last` checkpoint, setup, epoch start, train end and summary (majors 1, 6) | `fix1/lp_tiny/pure_sync/phases.csv`: `epoch_end/ckpt_last*` 0.137 s of the 0.339 s steady window, `epoch_start` 0.0085 s | loop window (`profile.is_outside_loop`, `outside_loop_s`, `loop_wall_s`, `s_per_micro_batch_loop`, `device_busy_loop_s`, `device_idle_frac_loop`, `device_idle_frac_est_loop`); `overhead_vs_g3_ms` uses the loop rate, `overhead_steady_vs_g3_ms` the steady rate; table column `loop`; H1 / H5 (a) / H7 rules use the loop numbers |
| the driver's data positions untested (major 2): reviewer mutants r01 / r02 survived | `rev3/mut1.json` | `test_driver_sets_the_data_position`: the `train/step` events carry (i, (i-1)//K, epoch) for i = 1..MICRO; the `data` phase before each step carries the same; the compiles are at micro-batches [1, K+1]; `steady["train/step"]["n"] == MICRO - K - 1` |
| `--profile-step-ref-ms` path untested (r04) | `rev3/mut1.json` | `test_fit_command_writes_the_profile` passes 250 ms and checks `step_ref_s` and `device_idle_frac_est_loop` |
| trace window untested (r03) | `rev3/mut1.json` | the `runs` fixture wraps `jax.profiler.start_trace` / `stop_trace` and records how many micro-batches were dispatched: window [3, 6) gives `[("start", 3), ("stop", 6)]` |
| docstring listed two async waits | `loop.py` `_log_step`, `validate`, `_fit` | lists all five and the tail reordering |
| synthetic run too thin (rmut r01, r04, r05 survived) | `rev/rmut.out` | first `data` has a `data/loader` child; added `epoch_start`, `train/log/bits`, `epoch_end/drain`, a compiling validation forward, a short batch's `tail`, `epoch_end/ckpt_last`, `train_end/ckpt_last`, `train_end/drain/wait`, `summary`; every number by hand (wall 50.25 s, loop window 10.5 s) |
| 18_loop_profile logic untested (rmut r06-r08, r10, r11 survived) | `rev/rmut.out` | asymmetric log windows; `run_fit` with `--pert-log-every 3 --smoke-ckpt-every 7 --smoke-log-every 5`; `test_budget_counts_loader_variants`; `test_loader_runs_without_the_determinism_flag` |
| stale historical sections | PROF.md | headings marked "before the review fixes" |

Correction: the reviewer suggested `steady["train/step"]["n"] == MICRO - K`. In the test config, optimizer step 1
compiles the EBOPs-off train program at micro-batch K + 1 (`PeriodicEBOPs`: one program per `ebops_on` value), and
`steady` uses only non-compile `train/step` phases. So the right count is MICRO - K - 1 = 13; the test also asserts the
compile positions [1, K + 1], so the count is not taken on trust.

Effect on the earlier tiny run (`fix1/lp_tiny`, re-derived from its `phases.csv` with the new `summary()`;
`PROF_evidence/resummarize_review2.out`): `pure_sync` `device_idle_frac` 0.531 (steady window) vs
`device_idle_frac_loop` 0.214; `pure` 22.5 ms/micro steady vs 12.9 loop; `pure_det` 23.9 vs 13.5; `smoke` 62.3 vs
49.8; `prod` 37.5 vs 15.3. At production dims the `last` save is 2.5-3.2 s on the Mac SSD (~8-10 ms per micro-batch
over 319 steady micro-batches); on /eagle it is unknown (H2), and the loop rate no longer contains it.

Verification of the review 2 fix (`PROF_evidence/test_runs.txt`, `mutations_review2_all.json`,
`baseline_cmp2_review2.out`, the end of `lp_tiny_table.txt`):

- Full `tests/keras_native` on the final code: 845 passed, 16 skipped (844 + `test_driver_sets_the_data_position`;
  the same hls4ml / real-data skips). `test_profile.py`: 15 passed; `tests/polaris/test_loop_profile_script.py`: 8.
- Mutations, all in one run on the final code: the baselines p00, p00b, p00c, p00d pass and all 76 mutants are killed
  (the 49 earlier ones, p21 / m03 / m07 / s05 with snippets updated to the new code, and the 27 below).
- Profiler off / async / sync + prefetch 2 are still bitwise EQUAL to the S7 run; ctl_bit and ctl_meta give DIFFERENT.
- `--tiny` run of smoke, pure, pure_sync, pure_log, pure_noprof, loader0 (fresh run root): all completed, the table
  has the `loop` column, no `ckpts/` left. `pure_sync`: `device_idle_frac` 0.504 (steady window 0.262 s) vs
  `device_idle_frac_loop` 0.200 (loop window 0.156 s). Perturbation 1.034 (tiny CPU: plumbing only).

| mutant | what it breaks | killed by |
|---|---|---|
| v01 / v02 | the epoch-end `last` checkpoint stays in the loop window / the epoch drain leaves it | test_derived_numbers_on_a_fake_clock |
| v03 / v04 | outside-loop compiles / the first validation batch subtracted twice | same |
| v05 / v06 / v07 / v08 | loop busy not restricted / loop idle over the steady window / loop rate = steady rate / loop estimate over the steady window | same (v08 also test_fit_command_writes_the_profile) |
| v09 / v10 | setup / the train end counted in the loop window | same |
| v11 / v12 / v13 | overhead vs G3 from the steady rate / table `loop` column shows the steady rate / derive drops `device_idle_frac_loop` | test_unprofiled_variant_and_steady_overhead |
| v14 / v15 | the driver starts the trace one micro-batch late / stops it one early | test_sync_run_wrote_a_trace |
| q01 / q02 | (reviewer r01 / r02) micro-batch position off by one / optimizer-step position shifted | test_driver_sets_the_data_position |
| q03 / q04 | (r03 / r04) trace window starts late / `--profile-step-ref-ms` ignored | test_sync_run_wrote_a_trace / test_fit_command_writes_the_profile |
| q05 / q06 / q07 | (rmut r01 / r04 / r05) first train batch by exclusive time / drains not busy / bits, tail not busy | test_derived_numbers_on_a_fake_clock |
| q08 | (r06) log windows drop the second window too | test_smoke_timing_splits_one_time_costs |
| q09 / q11 | (r07 / r10) budget ignores loaders / loader gets the determinism flag | test_budget_counts_loader_variants / test_loader_runs_without_the_determinism_flag |
| q10 / q12 | (r08 / r11) `--pert-log-every` / `--smoke-ckpt-every` ignored | test_run_fit_builds_each_variant |

## Local check of every 18_loop_profile variant (`--tiny`, before the review fixes)

At that time there were 9 fit variants (`pure_log` came with the first review fix; 10 now). All 9 fit variants (smoke, smoke_nodet, prod, pure, pure_det, pure_sync, pure_noprof, pure_prefetch, pure_trace) and
loader0 / loader4 ran to completion (`PROF_evidence/lp_tiny_table.txt`). loader8 ran at production dims (below).
Unattributed time: 0.005-0.032% of the wall per variant. This checks the script, not the timing (tiny dims, CPU, other
sessions' test suites running at the same time). Two fixes came out of it:

- `pure_noprof` reported `(fit_s - first_step_s) / (micro - 1)` as its compile-free rate. That rate keeps every later
  compile (0.39 s vs 0.016 s per micro-batch for the profiled `pure`). Now an unprofiled variant reports only
  `fit_s / micro`. The perturbation was `fit_s(pure) / fit_s(pure_noprof)`, marked NOT VALID if either run missed the
  compilation cache (superseded by the review fix: steady log windows).
- The summary groups mixed compile phases into the steady groups (`epoch_end/val` held its forward compile). Now
  they are in group `compile`.

A final `--tiny` run of the finished script (`smoke_nodet`, `pure`, `pure_noprof` on a fresh run root; end of
`lp_tiny_table.txt`) shows the `compile` group and a perturbation ratio of 1.049 that the first version marked valid
(`pure` and `pure_noprof` both hit the programs `smoke_nodet` compiled). The review showed that this whole-fit ratio
measures little: 15.0 of the 15.3 s of `pure` were compiles. `bash -n` passes on `18_loop_profile.pbs` and `submit.sh`.

The cache key includes `XLA_FLAGS` (jax 0.11.2 `_src/cache_key.py` hashes the flags). In the tiny run, `smoke` (with
the flag) missed 3 programs, `smoke_nodet` then missed 4, and `pure_det` hit the programs that `smoke` compiled.

## CPU measurements at production dims

Mac (10 cores), `.venv-jax`, `polaris/03_make_config.py` with `HGQ_PRECISION=32` (dim 256, encoder 6 + decoder 4,
Linformer k 64, 150 queries, batch 32, accumulation 4), dummy CLIC events at the real shapes, 8 workers, one epoch of
16 micro-batches (`PROF_evidence/prod_cpu_phase_tables.txt`). Other sessions ran test suites during these runs (load
average 5.0-20.6, median 7.5: `cpu_load_during_prod_runs.txt`), so the times are upper bounds for this Mac.

| quantity | measured | notes |
|---|---|---|
| train step on the CPU | 24-31 s per micro-batch | not relevant to the GPU. On the CPU the dispatch itself takes the step time (`train/device` in sync mode: 2.6 ms) |
| checkpoint size | 1.11 GB (1,111,628,515 bytes): tv 243 MB (1316 arrays), ntv 134 MB (2907), ov 730 MB (3949); sidecar 1.6 MB | the same for step, epoch and last files |
| full checkpoint save | 2.97 s per save (4 step saves): npz_write 1.89 s, sha256 0.56 s, manager (outside the sub-phases) 0.42 s, host_copy 0.05 s, sidecar 0.03 s, meta 0.02 s. Epoch save 1.73 s | Mac SSD |
| `last` save after a save at the same step | 4.5 ms | |
| log row (each optimizer step here) | 1.66 s median: metrics_forward 1.34 s (CPU forward of 32 events), bits 0.11 s, device_get 0.04 s, metrics_host 0.02 s, csv 0.01 s. The first row compiles for 28 s (forward 18 s + bits 10 s) | host part ~0.31 s per row |
| validation, 256 events | 16.1 s: forward 10.2 s (CPU), val loader worker start 5.4 s (first `next()`, once per run), the rest 0.38 s | host part ~1.5 s per 1000 events; CPU forward 40 s per 1000 events |
| loader alone, dummy events (30 batches) | 0 workers 14.0 ms/batch median; 4 workers 2.9 ms (first batch 2.6 s); 8 workers 0.8 ms (first batch 3.8 s) | conversion + `device_put` < 1 ms |
| loader alone, REAL events (`train_clic_fix.root` of the main checkout, 49 batches) | 0 workers 20.3 ms/batch; 4 workers 1.5 ms (mean 5.6, p90 19.6; first batch 9.8 s); 8 workers 1.9 ms (mean 3.7, p90 11.1; first batch 19.8 s) | the worker start sends the loaded data set to every worker |
| main-thread `data` inside the fits (8 workers, dummy) | 35-44 ms median per micro-batch | the workers compete with the CPU step for the 10 cores; the loader-only rows are the A100-host case |
| build / first step | build 53-96 s; first step 187 s (cache miss), 93-101 s (cache hit) | a cache hit still costs ~95 s here (tracing and lowering run before the cache lookup) |
| profiler perturbation | not measurable here | `pure` ran first with a cold cache (2 misses) and `pure_noprof` hit: the whole-fit ratio 1.57 is not the perturbation (review fix) |

## Attribution at the smoke cadence

The smoke: 3 epochs x 400 micro-batches (300 optimizer steps; the worklog says run A reached 297), a log row every 10
steps, a step checkpoint every 50 steps (all kept), validation of 1000 events per epoch. So per run: 30 log rows,
6 step saves + 3 epoch saves (each epoch-end `last` save follows a step save at the same step, so it is cheap), and
3 validations.

| cost | per run (this Mac, measured) | s per micro-batch (/ 1200) | decided by the host-side numbers? |
|---|---|---|---|
| H0 one-time: build + first step (worklog, Polaris run A) | 110 + 218 s | 0.28 if the "end to end" figure includes them | No: `smoke-timing` on the smoke JSON decides it (no GPU) |
| H2 full checkpoint saves (9 x 2.97 s) | 27 s | 0.022 | An upper bound at SSD speed. To reach 0.30 alone, one save would need ~40 s (write and re-read 1.1 GB at ~55 MB/s). Only the run on /eagle measures this |
| H4 log rows, host part (30 x 0.31 s) | 9 s | 0.008 | Yes, small. The forward per row is GPU work (at most one 0.16 s step: <= 0.004 s per micro-batch) |
| H3 validation, host part (3 x 1.5 s) + val loader start once (5-20 s) | 10-25 s | 0.008-0.021 | Yes, small. The GPU forward (96 batches, each at most one 0.16 s step) adds <= 0.013 |
| H1 loader throughput | 1.5-1.9 ms per batch with 4-8 workers on real events | << the 0.16 s step | Yes: the loader keeps up (margin ~80x). Whether the main-thread data work serialises with the GPU step: A100 run |
| sum of the measured recurring host costs | | 0.04-0.07 | the rest needs the A100 run |

## Hypotheses: status

- **H0 (new) one-time costs in the end-to-end figure: OPEN. Check first; no GPU needed.** The worklog numbers give
  0.28 s per micro-batch for run A (above) and (61 + ~110) s / 588 micro-batches = 0.29 s for run B (B trains from
  step 150). Both are close to the 0.30 s gap. `18_loop_profile.py smoke-timing logs/jax_smoke_7763621*.json` prints
  `s_per_micro_wall`, `s_per_micro_fit`, `s_per_micro_fit_excl_first_step`, `one_time_s_per_micro` and the
  `steps.csv` log windows (median `time/ms_per_micro_batch`). If `s_per_micro_fit_excl_first_step` and the median log
  window are near 0.16-0.20 s, H0 explains most of the gap and the steady loop is fine.
- **H1 loader / host batch preparation: the loader keeps up (decided on the CPU, above).** Open: main-thread `data`
  work (~1 ms of conversion and `device_put` per batch here) that serialises with the step. A100: the loop rate
  (`s_per_micro_loop`) of `pure` vs `pure_prefetch`, the `steady` `data` median, and `device_idle_frac_loop` of
  `pure_sync` (over the loop window; `device_idle_frac` still holds the epoch-end `last` checkpoint and
  `device_idle_frac_wall` the compiles, so neither is the answer).
- **H2 checkpoints: small at SSD speed (0.022 s per micro-batch); the speed of /eagle is not known.** A100 run:
  `train/ckpt/{npz_write,sha256}` in `smoke` (a step save every 50 steps) vs `pure`. Side finding: each checkpoint is
  1.11 GB, so the smoke wrote ~10 GB (all kept).
- **H3 validation + host metrics: small (host 1.5 s per 1000 events).** The A100 run gives `epoch_end/val/forward`.
- **H4 logging every 10 steps: small (host 0.31 s per row).**
- **H5 hidden per-micro-batch host sync: OPEN (A100).** Without the profiler the driver calls `block_until_ready`
  exactly `1 + logged steps` times (test_syncs_added_by_each_mode). The `train/step` dispatch time of `pure` alone
  cannot decide H5: `pure` has no log, checkpoint or validation, so the host runs ahead until the device queue is full
  and then every dispatch waits about one device step anyway (back-pressure: PJRT's `compute_semaphore_` "limit[s]
  how many programs can be enqueued on the compute stream by the host ahead of the device",
  https://github.com/openxla/xla/blob/main/xla/pjrt/se/local_device_state.h). A sync and back-pressure give the same
  steady `train/step`. Decide H5 by (a) `s_per_micro_loop` of `pure` against the device step (`train/step +
  train/device` of `pure_sync`, or G3's 161.6 ms): no hidden sync and no other host cost means they are equal (the
  loop rate leaves out the epoch-end `last` checkpoint, whose /eagle write time is H2, not H5); a
  hidden sync adds the host time per micro-batch (`data`, `train/beta`, dispatch) to the device step; (b) the
  `train/step` durations by micro-batch in `pure`'s `phases.csv`: back-pressure starts only after the in-flight queue
  fills (the first several steady dispatches are short), a sync makes every dispatch from micro-batch 2 on as long
  as a step; (c) the `pure_trace` timeline: gaps between the train programs on the GPU stream. On the CPU the dispatch
  is synchronous, so the CPU cannot decide this.
- **H6 Python overhead: the unattributed time is 0.005-0.032% of the wall in every run.** The per-micro-batch Python
  phases (`train/beta`, the dispatch) need the A100 run.
- **H7 the smoke ran with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, G3 without: OPEN (A100 only).**
  `16_jax_smoke.pbs:53` sets the flag by default; `17_g3_step.pbs` sets none. OpenXLA: deterministic scatter "can lead
  to substantial runtime throughput loss" (https://openxla.org/xla/determinism). The CPU build ignores GPU flags, so
  only the A100 decides: `pure` vs `pure_det` decides H7 by their loop rates (`s_per_micro_loop`: no checkpoint write
  in the window, so no /eagle noise; the `steady` `train/step` and `data` medians as a check). Both are bare loops and
  both load warm programs (check that the `misses` column of the table is 0 for both). `smoke` vs `smoke_nodet` is
  supporting evidence only, because `smoke` (with the
  flag) may load the 16_jax_smoke programs from the cache while `smoke_nodet` compiles, and a cache miss changes more
  than the compile time (the review's tiny CPU run under load ~30: `pure` with 2 misses 0.213 s/micro without compiles,
  `pure_det` with 6 hits 0.078; not reproduced here). The flag also changes the compilation-cache key (above).
- **H8 (new) the host matching callback competes with the main thread (GIL): OPEN (A100 only).** The matching runs in
  a `pure_callback` on a host thread while the main thread converts batches and logs. G3 measured the callback alone
  (17.2 ms). The A100 run records `concurrent.match_callback` in every variant: compare `pure` (async), `pure_sync`
  (main thread blocked) and `pure_prefetch` with 17.2 ms.

## Polaris commands

After the merge into `keras-jax` (`6c46fef`, [PROF-merge.md](PROF-merge.md)), run from the keras-jax clone instead
(no second clone): [KERAS_NATIVE.md](../KERAS_NATIVE.md), "Profiling the driver". The commands below are for a
separate clone of `prof/loop`, as before the merge.

The branch is not pushed. First, on this Mac:

    git -C /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-prof push origin prof/loop

Then on a Polaris login node (steps 1 and 3 need no GPU). Use bash: `polaris/repo_dir.sh` finds its clone through
`BASH_SOURCE`, so the check below prints the wrong directory in zsh.

    # WORK_ROOT from the keras-jax clone's env.sh (env.sh.example: /eagle/${PBS_PROJECT}/hepattn-hgq); edit KJ
    KJ=/eagle/<project>/hepattn-hgq/hepattn-keras-jax
    export WORK_ROOT="$(bash -c "source $KJ/polaris/env.sh; echo \$WORK_ROOT")"; echo "$WORK_ROOT"
    # a clone of prof/loop that reuses the keras-jax env (no dependency change since keras-jax f30d650)
    cd "$WORK_ROOT"
    git clone -q --branch prof/loop https://github.com/erdemyigit/hepattn.git hepattn-prof-loop
    ln -s "$WORK_ROOT/hepattn-keras-jax/.venv-jax" hepattn-prof-loop/.venv-jax
    cp hepattn-keras-jax/polaris/env.sh hepattn-prof-loop/polaris/env.sh
    cd hepattn-prof-loop
    bash -c 'source polaris/env.sh; source polaris/repo_dir.sh; echo "$HEPATTN_DIR"'   # must print $WORK_ROOT/hepattn-prof-loop
    #   if it does not: remove the HEPATTN_BRANCH / HEPATTN_DIR lines from this copy of env.sh

    # 1. H0, from the smoke's own results
    .venv-jax/bin/python polaris/18_loop_profile.py smoke-timing "$WORK_ROOT"/hepattn-keras-jax/logs/jax_smoke_7763621*.json

    # 2. the A100 profile (debug queue, 1 h). Order: the H7 pairs, the perturbation pair (pure_log, pure_noprof),
    #    loader8, prod, pure_sync, pure_prefetch, pure_trace, loader4, loader16. A variant starts only if the time since
    #    the job start plus the longest variant so far is within 55 min.
    bash polaris/submit.sh 18_loop_profile.pbs
    #   a second job for what the budget skipped (the table lists them), for example:
    #   PROF_VARIANTS=pure_sync,pure_prefetch,pure_trace,loader4,loader16 bash polaris/submit.sh 18_loop_profile.pbs
    #   shorter fits: PROF_ARGS="--micro 200" (keep >= 4 x accumulation x --pert-log-every micro-batches for 3 windows)

    # 3. the results: the table and the perturbation of one or more JSONs (a job killed at the walltime has no
    #    'table' key; `report` rebuilds it; a later file's variant replaces an earlier one's)
    .venv-jax/bin/python polaris/18_loop_profile.py report logs/loop_profile_<jobid>.json [logs/loop_profile_<jobid2>.json]

    # 4. clean up: each fit's checkpoints are deleted after the fit (--keep-ckpts keeps them); what remains is
    #    the run directories, the jax trace and the compile cache entries
    du -sh "$WORK_ROOT"/loop_profile_<jobid>; rm -rf "$WORK_ROOT"/loop_profile_<jobid>/*/ckpts

The perturbation is the median log window after the first of `pure_log` over `pure_noprof` (both log every
`--pert-log-every` 8 optimizer steps: 9 windows after the first at 320 micro-batches). A window is the wall time
between two log rows over its micro-batches, synced at both ends, so it holds no compile, no loader start and no log
row. The earlier ratio `fit_s(pure) / fit_s(pure_noprof)` included the compiles and cache loads in both runs (15.0 of
the 15.3 s `fit_s` of the tiny `pure`), which pulls it toward 1; it is kept as `fit_s_ratio` for reference only. If
the two runs land in different jobs, `report` of both JSONs computes it.

## Evidence index (`PROF_evidence/`)

`test_runs.txt`, `baseline_cmp.sh`, `baseline_cmp2.py`, `baseline_controls.py`, `baseline_cmp2.out`,
`baseline_cmp2_reviewfix.out`, `mutations_*.json` (`mutations_review_all.json`: every mutant on the review-fix code), `lp_tiny_table.txt`,
`prod_cpu_phase_tables.txt` (all production-dim phase tables, the loader runs, the RESULT fields, the re-derived
perturbation), `cpu_load_during_prod_runs.txt`, `smoke_timing_tiny_local.json` (`smoke-timing` on a local
`16_jax_smoke.py --tiny` result).
Review 2: `mutations_review2_all.json` (every mutant on the review 2 code), `baseline_cmp2_review2.out`,
`resummarize.py` + `resummarize_review2.out` (the review-fix tiny run's `phases.csv` re-derived with the loop window).
