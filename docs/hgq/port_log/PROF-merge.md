# PROF merge into `keras-jax` (2026-10-21)

`prof/loop` at `d813090` (the phase profiler after review 2, [PROF.md](PROF.md)) is merged into `keras-jax` at
`fb45632` (S7 merged, [S7-merge.md](S7-merge.md)) as `6c46fef` (`git merge --no-ff`). The merge base is `111aede`
(S7's last commit before its review rounds).

## Conflicts

After `111aede`, `keras-jax` changed `convert_to_native.py`, `eval_keras.py`, `port_to_keras.py`,
`keras_native/export.py`, the S7 tests and docs; `prof/loop` changed `keras_native/{profile,loop,cli,data,checkpoint}.py`,
`polaris/submit.sh`, `polaris/18_loop_profile.*`, its tests and docs. The only path both changed is
`docs/hgq/port_log/README.md` (adjacent table rows): both rows are kept, with the PROF row brought up to review 2.
So `loop.py` and `cli.py` needed no resolution: S7's `warm_start` is already in `111aede`, under the profiler edits.
Check on the merged tree: every path that differs from `fb45632` has the `prof/loop` blob, and every path that differs
from `prof/loop` has the `fb45632` blob, except the README (a loop over `git diff --name-only` with `git diff --quiet`).

## Suites on the merged tree

Code of every run: `6c46fef` (`src`, `tests`, `polaris` as committed; the later commit changes docs and adds evidence
files only). Logs: `scratchpad/profmerge/` of this session. Runs overlapped, so the times are not benchmarks.

| # | Command (from the worktree root) | Env | Result (verbatim summary) |
|---|---|---|---|
| 1 | `env -u KERAS_BACKEND PYTHONPATH=$PWD/src .venv-jax/bin/python -m pytest tests/keras_native -rs` | `hepattn-keras-jax/.venv-jax` (no hls4ml) | `869 passed, 16 skipped, 4823 warnings in 2095.08s (0:34:55)` (885 collected). Skips: 4 `test_data_real.py` (no `val_clic_fix.root`), 12 `test_export.py` "hls4ml not installed" |
| 2 | `KERAS_BACKEND=torch PYTHONPATH=$PWD/src ../hepattn/.venv/bin/python -m pytest -m 'not gpu and not requiresdata' tests/keras tests/polaris -rs` | `hepattn/.venv` | `340 passed, 2 skipped, 6 deselected, 61837 warnings in 531.25s (0:08:51)` (2 csim, macOS) |
| 3 | `env -u KERAS_BACKEND PYTHONPATH=$PWD/src .venv-jax/bin/python -m pytest tests/keras_native/test_profile.py tests/polaris/test_loop_profile_script.py -rs` | `hepattn-keras-jax/.venv-jax` | `23 passed in 235.67s (0:03:55)` |

How the counts line up:
- Row 1 = S7-merge's 854 + the 15 tests of `test_profile.py` (885 = 870 + 15 collected). Per-file counts against the
  S7-merge log differ only by `test_profile.py`. `test_resume_cross_lane.py` (5), `test_train_step.py` (21),
  `test_loop.py` (32) and `test_cli.py` (7) pass.
- Row 2 = S7-merge's 331 + 8 (`tests/polaris/test_loop_profile_script.py`) + 1
  (`tests/keras/test_polaris_scripts.py::test_scripts_parse_and_use_the_configured_clone[18_loop_profile.pbs]`: the
  test is parametrized over the `.pbs` files and is unchanged since `111aede`).
- Row 3 = 15 + 8, as on `prof/loop` (PROF.md "Review 2 fix").

## Profiler off is unchanged

The tiny CLI fit of `PROF_evidence/baseline_cmp.sh` (`tiny_polaris.yaml`, dummy events, 2 epochs, a checkpoint every
2 steps, all kept, a log row every step) ran with `git archive` copies of the pre-merge `src` (`fb45632`) and the
merged `src` (`6c46fef`), in `.venv-jax` (no hls4ml: the fit path does not need it). Runner:
`PROF_evidence/baseline_cmp_merge.sh`; result: `PROF_evidence/baseline_cmp2_merge.out`.

- `off` (profiler off), `async` (`--profile-phases`) and `sync_pf` (`--profile-sync --prefetch 2`) are EQUAL to the
  pre-merge run: 6 checkpoints, 13710 state arrays bitwise, `__meta__` without wall-clock keys, `steps.csv` and
  `metrics.csv` without `time/` columns.
- Controls (`baseline_controls.py` on copies of `off`): one flipped mantissa bit (`ctl_bit`) and `beta` + 1 in one
  checkpoint's metadata (`ctl_meta`) both compare DIFFERENT.

## The tests still discriminate on the merged tree

`PROF_mutations.py` on a copy of the merged worktree (`--only p00,p00b,p00d,p01,p02,q01,q03,v01,q08`;
`PROF_evidence/mutations_merge.json`): the 3 baselines pass (15, 8 and 3 tests) and the 6 mutants are killed, each by a
failing assertion: profiler off adds a sync (p01, `test_syncs_added_by_each_mode`), a profiled run changes the matching
(p02, `test_profiler_and_prefetch_do_not_change_the_run`), the loop window keeps the `last` checkpoint (v01), the
micro-batch position is off by one (q01), the trace window starts late (q03), and an `18_loop_profile` log-window
mutant (q08). The full set (76/76) ran on `prof/loop` (`mutations_review2_all.json`); the profiler files are the same
blobs here.

## Docs

[KERAS_NATIVE.md](../KERAS_NATIVE.md): a PROF row in the module table and a section "Profiling the driver" (CLI options,
the rates to read, the local `--tiny` check, the Polaris commands from the keras-jax clone).
[KERAS_JAX_PLAN.md](../KERAS_JAX_PLAN.md): "Status update (2026-10-21, PROF merged)". PROF.md "Polaris commands" points
to the keras-jax route. README: PROF and PROF-merge rows, the S5/S6b row's loop overhead item.

## Open

- No A100 numbers: H0 (`smoke-timing`, no GPU), H2 on /eagle, H5, H7, H8 need the Polaris job. `keras-jax` is not
  pushed (`origin/keras-jax` is `f30d650`); the Polaris route needs the push first (the user's decision).
- Carried from S7-merge: Lindsey Gray's checkpoint, G4, the beta scan, csim by pytest on Linux, the tolerance-0 golden
  rows on Linux/GPU, G3 diagnostics job 7764401.
