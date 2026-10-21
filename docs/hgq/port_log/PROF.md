# PROF: where the native driver's ~0.3 s per micro-batch goes

Branch `prof/loop` (from `s7/export` 111aede), worktree `hepattn-prof`. Question: the S6b smoke (Polaris job 7763621,
A100, real data, batch 32 x accumulation 4) ran at ~0.49 s (run A) and ~0.465 s (run B) per micro-batch "end to end".
The compiled step alone (G3, job 7763879, fp32 'highest', host matching included) is 161.6 ms at batch 32. So
~0.30 s per micro-batch is outside the step. Neither job's logs are in a local tree. The numbers come from the task
brief and from the main worktree's `WORKLOG.hgq.md` (lines 232-235 and 281-287). The worklog also records: run A
reached 297 optimizer steps, first step 218 s (run A) and 61 s (run B, cache hits), model build ~110 s, and the G3
matching callback 17.2 ms.

## Status (2026-10-20, end of the session)

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
  first. Not submitted. The branch is not pushed.

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
- The matching callback runs in another thread. Its wall time per call is a concurrent counter (`match_callback`),
  not a phase. Only the thread that activated the profiler records phases.
- Off: `profile.NULL`. `phase()` returns one shared `nullcontext`; `wait()` / `block()` return their argument. No clock
  read, no added sync.
- `async` mode adds `block_until_ready` only where the next statement reads the value on the host anyway (before each
  checkpoint save, `epoch_end/drain`, `train_end/drain`, before the host metrics of a logged step and of each
  validation batch). `train/step` is the dispatch time; it grows only under back-pressure.
- `sync` mode (`--profile-sync`) also blocks after every train step (`train/device`) and every device transfer
  (`data/device_put_wait`). `device_idle_frac = 1 - busy / wall` is then a lower bound on the idle fraction.
- `--jax-trace DIR --jax-trace-window A:B`: a `jax.profiler` trace of micro-batches [A, B).
- `--prefetch N` (`data.prefetch`): the train batches are read ahead in a thread; `data` is then the queue wait.
- Outputs: `<csv>` (one row per phase), `<csv>.json` (summary), `summary.json["profile"]`, a table on stdout.

## Verification (this session)

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
| p21 / p22 / p23 | an unprofiled variant gets a compile-including rate / perturbation ratio inverted / perturbation valid with a cold compile | test_unprofiled_variant_and_perturbation |

Evidence: `mutations_p00-p17.json`, `mutations_rerun_p00_p07_p18.json` (after the `compile` group change),
`mutations_p00b_p19-p23.json`.

## Local check of every 18_loop_profile variant (`--tiny`)

All 9 fit variants (smoke, smoke_nodet, prod, pure, pure_det, pure_sync, pure_noprof, pure_prefetch, pure_trace) and
loader0 / loader4 ran to completion (`PROF_evidence/lp_tiny_table.txt`). loader8 ran at production dims (below).
Unattributed time: 0.005-0.032% of the wall per variant. This checks the script, not the timing (tiny dims, CPU, other
sessions' test suites running at the same time). Two fixes came out of it:

- `pure_noprof` reported `(fit_s - first_step_s) / (micro - 1)` as its compile-free rate. That rate keeps every later
  compile (0.39 s vs 0.016 s per micro-batch for the profiled `pure`). Now an unprofiled variant reports only
  `fit_s / micro`. The perturbation is `fit_s(pure) / fit_s(pure_noprof)`, and the table marks it NOT VALID if either
  run missed the compilation cache.
- The summary groups mixed compile phases into the steady groups (`epoch_end/val` held its forward compile). Now
  they are in group `compile`.

A final `--tiny` run of the finished script (`smoke_nodet`, `pure`, `pure_noprof` on a fresh run root; end of
`lp_tiny_table.txt`) shows the `compile` group and a valid perturbation ratio (1.049; `pure` and `pure_noprof` both
hit the programs `smoke_nodet` compiled). `bash -n` passes on `18_loop_profile.pbs` and `submit.sh`.

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
| profiler perturbation | not measurable here | `pure` ran first with a cold cache (2 misses) and `pure_noprof` hit: the ratio 1.57 is NOT VALID |

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
  work (~1 ms of conversion and `device_put` per batch here) that serialises with the step. A100: `pure` vs
  `pure_prefetch`, and `device_idle_frac` of `pure_sync`.
- **H2 checkpoints: small at SSD speed (0.022 s per micro-batch); the speed of /eagle is not known.** A100 run:
  `train/ckpt/{npz_write,sha256}` in `smoke` (a step save every 50 steps) vs `pure`. Side finding: each checkpoint is
  1.11 GB, so the smoke wrote ~10 GB (all kept).
- **H3 validation + host metrics: small (host 1.5 s per 1000 events).** The A100 run gives `epoch_end/val/forward`.
- **H4 logging every 10 steps: small (host 0.31 s per row).**
- **H5 hidden per-micro-batch host sync: OPEN (A100).** Without the profiler the driver calls `block_until_ready`
  exactly `1 + logged steps` times (test_syncs_added_by_each_mode). An implicit sync (a `float()` of a device value)
  would show as a `train/step` dispatch time close to the device step time in `pure` (async). On the CPU the dispatch
  is synchronous, so the CPU cannot decide this.
- **H6 Python overhead: the unattributed time is 0.005-0.032% of the wall in every run.** The per-micro-batch Python
  phases (`train/beta`, the dispatch) need the A100 run.
- **H7 the smoke ran with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, G3 without: OPEN (A100 only).**
  `16_jax_smoke.pbs:53` sets the flag by default; `17_g3_step.pbs` sets none. OpenXLA: deterministic scatter "can lead
  to substantial runtime throughput loss" (https://openxla.org/xla/determinism). The CPU build ignores GPU flags, so
  only the A100 decides: `smoke` vs `smoke_nodet` and `pure` vs `pure_det` (first in the default order). The flag also
  changes the compilation-cache key (above).
- **H8 (new) the host matching callback competes with the main thread (GIL): OPEN (A100 only).** The matching runs in
  a `pure_callback` on a host thread while the main thread converts batches and logs. G3 measured the callback alone
  (17.2 ms). The A100 run records `concurrent.match_callback` in every variant: compare `pure` (async), `pure_sync`
  (main thread blocked) and `pure_prefetch` with 17.2 ms.

## Polaris commands

The branch is not pushed. First, on this Mac:

    git -C /Users/eertorer/Desktop/CMS_Group/LightweightTransformers4RECO/hepattn-prof push origin prof/loop

Then on a Polaris login node (steps 1 and 2 need no GPU):

    # a clone of prof/loop that reuses the keras-jax env (no dependency change since keras-jax f30d650)
    cd $WORK_ROOT
    git clone -q --branch prof/loop https://github.com/erdemyigit/hepattn.git hepattn-prof-loop
    ln -s $WORK_ROOT/hepattn-keras-jax/.venv-jax hepattn-prof-loop/.venv-jax
    cp hepattn-keras-jax/polaris/env.sh hepattn-prof-loop/polaris/env.sh
    cd hepattn-prof-loop
    (source polaris/env.sh; source polaris/repo_dir.sh; echo "$HEPATTN_DIR")   # must print $WORK_ROOT/hepattn-prof-loop
    #   if it does not: remove the HEPATTN_BRANCH / HEPATTN_DIR lines from this copy of env.sh

    # 1. H0, from the smoke's own results
    .venv-jax/bin/python polaris/18_loop_profile.py smoke-timing $WORK_ROOT/hepattn-keras-jax/logs/jax_smoke_7763621*.json

    # 2. the A100 profile (debug queue, 1 h; the H7 pairs run first)
    bash polaris/submit.sh 18_loop_profile.pbs
    #   a second job for what the 52 min budget skipped, for example:
    #   PROF_VARIANTS=pure_sync,pure_prefetch,pure_noprof,pure_trace,loader4,loader16 bash polaris/submit.sh 18_loop_profile.pbs
    #   shorter fits: PROF_ARGS="--micro 200"

    # 3. the results (written after every variant)
    .venv-jax/bin/python -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['table']); print(r.get('perturbation'))" logs/loop_profile_<jobid>.json

`pure_noprof` follows `pure`, and both load the train programs that `smoke_nodet` compiled, so the perturbation
ratio is valid there (the table says NOT VALID otherwise).

## Evidence index (`PROF_evidence/`)

`test_runs.txt`, `baseline_cmp.sh`, `baseline_cmp2.py`, `baseline_controls.py`, `baseline_cmp2.out`, `mutations_*.json`, `lp_tiny_table.txt`,
`prod_cpu_phase_tables.txt` (all production-dim phase tables, the loader runs, the RESULT fields, the re-derived
perturbation), `cpu_load_during_prod_runs.txt`, `smoke_timing_tiny_local.json` (`smoke-timing` on a local
`16_jax_smoke.py --tiny` result).
