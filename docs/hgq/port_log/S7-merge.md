# S7 merge into `keras-jax` (2026-10-21)

`s7/export` at `2b15167` (S7 after review rounds 1 and 2, [S7.md](S7.md)) is merged into `keras-jax` at `f30d650` as
`f4aab4a` (`git merge --no-ff`, the form of the earlier stage merges). `f30d650` is the merge base, so there were no
conflicts, and `git diff --quiet s7/export f4aab4a` exits 0: the merged tree equals `2b15167`. The merge broke nothing,
so this stage changes no code; it adds this log and updates the docs (plan status, port-log README, KERAS_NATIVE.md).

## hls4ml stays optional in `.venv-jax`

`hepattn-keras-jax/.venv-jax` is not changed. `uv pip freeze` of it equals
[keras_native_env.txt](../keras_native_env.txt) (the freeze of `hepattn-s7/.venv-jax`) minus three lines:
`da4ml==0.6.0`, `hls4ml==1.3.0`, `pydigitalwavetools==1.1`. In it, the 12 `needs_hls4ml` tests of `test_export.py`
skip with "hls4ml not installed" and every other test runs (suite row 1). With `hepattn-s7/.venv-jax` on this
worktree's `src`, `test_export.py` gives 21 passed, 5 skipped (row 2); the 5 are the C-simulation tests, which skip on
macOS. The command is in [KERAS_NATIVE.md](../KERAS_NATIVE.md) (setup section).

## Suites on the merged tree

Code of every run: `f4aab4a` (`src`, `tests`, `polaris` as committed; the uncommitted edits are docs only). Logs:
`scratchpad/s7merge/*.log` of this session. The runs overlapped (EDT: row 1 00:44:49-01:22:08, row 3 00:45-00:54:08, rows 4-5 00:45-00:45:34, row 2
00:47-00:48:12), so the times are not benchmarks.

| # | Command (from the worktree root) | Env | Result (verbatim summary) |
|---|---|---|---|
| 1 | `env -u KERAS_BACKEND PYTHONPATH=$PWD/src .venv-jax/bin/python -m pytest tests/keras_native -rs` | `hepattn-keras-jax/.venv-jax` (no hls4ml) | `854 passed, 16 skipped, 4823 warnings in 2215.78s (0:36:55)` (870 collected). Skips: 4 `test_data_real.py` (no `val_clic_fix.root`), 12 `test_export.py` "hls4ml not installed" |
| 2 | `env -u KERAS_BACKEND PYTHONPATH=$PWD/src ../hepattn-s7/.venv-jax/bin/python -m pytest tests/keras_native/test_export.py -rs` | `hepattn-s7/.venv-jax` (hls4ml 1.3.0, da4ml 0.6.0) | `21 passed, 5 skipped, 596 warnings in 72.80s` (5 csim: "hls4ml csim toolchain unavailable on darwin") |
| 3 | `KERAS_BACKEND=torch PYTHONPATH=$PWD/src ../hepattn/.venv/bin/python -m pytest -m 'not gpu and not requiresdata' tests/keras tests/polaris -rs` | `hepattn/.venv` | `331 passed, 2 skipped, 6 deselected, 61837 warnings in 543.72s (0:09:03)` (2 csim, macOS) |
| 4 | same env, `pytest -m 'not gpu and not requiresdata' tests/models tests/loss` | `hepattn/.venv` | `519 passed, 173 skipped, 5 deselected, 15 warnings in 16.99s` |
| 5 | same env, `pytest -m 'not gpu and not requiresdata' tests/matching --junitxml=matching.xml`, then `python docs/hgq/port_log/R0_matching_gate.py matching.xml --max-failures 106` | `hepattn/.venv` | `106 failed, 66 passed, 14 skipped, 17 deselected, 2 xfailed, 23 warnings in 11.13s`; gate `ran=172 failures=106 lap1015_attr=106 other=0`, exit 0 (the unbuilt `lap1015` only, as in R5, S56-merge and S56-critique) |

Consistency with S7 review round 2 (same tree, `hepattn-s7/.venv-jax`: 861 passed, 9 skipped of 870): row 1 has
7 more skips and 7 fewer passes. Of the 12 `needs_hls4ml` tests, 5 skip on macOS in either venv (csim), so 7 run only
with hls4ml; row 2 shows them passing (21 = 14 tests without hls4ml + 7). Row 3 is round 1's 330 plus
`test_read_model_state_refuses_pickled_code` (added in round 2). Rows 4 and 5 equal S56-merge and S56-critique.

## Open (carried; see the plan's status of 2026-10-21)

- Lindsey Gray's PR #9 float checkpoint: convert, read `max_abs_diff` and `renorm_lane` in the `--report`, warm-start.
- G4 (S8), the Polaris beta scan, the loop overhead outside the train step (`prof/loop`, not merged).
- C simulation by pytest on Linux; the tolerance-0 golden rows on Linux/GPU; G3 diagnostics job 7764401.
- Push of `keras-jax`: the user's decision (`origin/keras-jax` is `f30d650`).
