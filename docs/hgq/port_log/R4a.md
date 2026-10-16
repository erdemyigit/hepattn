# R4a log: hybrid SwiGLU, sorting, Polaris config (retarget lane R4a)

Plan: `docs/hgq/RETARGET_ORG_MAIN.md` section 4 (option (a)) and the R4a row of section 6. Worktree `hepattn-rt-r4`,
branch `rt/r4` from R0 `5db27a8`. Date 2026-10-15, macOS arm64, CPU only, torch 2.10.0, keras 3.15.0, HGQ2 0.1.9.
`$S` is the scoping scratchpad; this lane's evidence is in `$S/r4a/` (probes, logs, mutants).

## What landed

| Commit | Content |
|---|---|
| `7718022` | `keras/dense.py`: SwiGLU as two factory dense halves (`<name>_hidden<i>_x1` linear, `_x2` with `activation="silu"`) and a factory einsum `bnc,bnc->bnc` (`_glu`; `QEinsum` when quantized), float and quantized; other ranks are folded into rank 3 around the einsum. `port_dense` splits the packed torch weight with `chunk(2, 0)`. Tests: `test_dense_parity.py` (3 org SwiGLU shapes, a swapped-halves control), new `test_swiglu_quantized.py`. |
| `5878fcd` | `KerasMaskFormer(input_sort_field=...)`; `KerasEncoder.forward` sorts with torch `sort_index` (padding last) before the registers and unsorts after, as org `Encoder.forward`. |
| `9fe9d7a` | `keras/factory.py` registers `EinsumOp` and `SoftmaxOp` as keras-serializable (deviation 2). |
| `a0e93db` | `polaris/03_make_config.py`: `base.yaml` + `linformer.yaml`, encoder unwrapped, solver `scipy`; `HGQ_ATTN=torch` for `base.yaml` alone. |
| `74e38a0` | New tests: `org_model_utils.py`, `test_org_model_parity.py`, `test_polaris_config.py`, `test_sorting.py`. |
| `397bb21` | `polaris/repo_dir.sh` (`HEPATTN_BRANCH`, `HEPATTN_DIR`), `01_setup_env.sh`, every `.pbs`, `env.sh.example`, `polaris/README.md`; new `test_polaris_scripts.py`. |
| `88fc24e` | `07_profile.py`, `15_g2_forward.py`: `linformer_k` (R0 open issue); ruff debt of `07_profile.py`. |
| `316cdb3` | `docs/hgq/USAGE.md`. |
| `921d87b` | `test_default_is_the_linformer_model` (mutant m6 survived `test_polaris_config.py` before it). |
| `a214354`..`a07ed1e` | Fix round 1 (section "Fix round 1" below). |
| `072433b`..`25d8b36` | Fix round 2 (section "Fix round 2" below). |

## Gates

Suite logs `$S/r4a/logs/v1.*` (runner `$S/r4a/run.sh`; hybrid and torch with `hepattn/.venv`, `KERAS_BACKEND=torch`,
native with `hepattn-keras-jax/.venv-jax`, `KERAS_BACKEND=jax`; `PYTHONPATH=<worktree>/src`, and `hepattn` resolves to
`hepattn-rt-r4/src`). `v1` ran on the tree of `316cdb3`; `921d87b` adds one test, re-run in `$S/r4a/logs/final.hybrid.log`.
The R1-R3 lanes ran their suites on the same machine at the same time, so wall times are long.

| Gate | Result | Control (makes it fail) |
|---|---|---|
| Hybrid float parity vs org torch, full model (2x measured) | **pass** for dense attention, seeds 0 and 1: outputs <= 3.7e-5, losses <= 5.5e-7 (fix round 2; were 3.3e-5 / 3.4e-7, and 2.4e-5 at R4a) (metric and numbers below), 4/4 attention-mask booleans equal. Linformer rows: strict xfail (`AttributeError`), R4b (deviation 1) | PR #8 controls at model level: `ca_dense_sa` 15.6 / 3.49e-2, `self_attn_norm_kv=False` 5.35 / 3.27e-3 (>= 10x), `hybridnorm` refused. Mutants m1, m2 |
| Quantized-SwiGLU paired delta | **pass**, 3 org shapes: high <= 2.6e-2 (measured <= 1.30e-2), low >= 10x high (11x-62x), low/float corr > 0.5; leaf classes `QDense, QDense, QEinsum`; each leaf's input quantizer live (>= 5.86e-2); per leaf `leaf.ebops > 0` and a differentiable loss term equal to `beta * ebops` (fix round 2); zero-bias high spec <= 8.6e-5 (measured <= 4.27e-5, fix round 1); at model level all 18 product leaves are in `keras_layers()`, the optimizer groups and `quant_losses()`, and each counts EBOPs in a training forward (fix rounds 1 and 2) | m2 (silu on x1), m11 (product not through the einsum: 4 rows fail), f2, f5, g1 |
| Generated Polaris config builds, float and quantized | **pass**, `linformer` and `torch`: built by jsonargparse with `MPflowHGQ` (as `main_hgq.py`); `proj_k` shapes {(168, 64), (160, 64)} (R0 carried gate); FastLayerNorm, `ca_sa_dense`, `self_attn_norm_kv`, phi sort, scipy solver, SwiGLU leaves. The YAML equals the org model section up to the documented edits; the default is the Linformer model | R0's generator (`5db27a8`): 11/11 rows fail (`$S/r4a/mut/m13.log`). Mutants m5, m6, m7, m8 |
| Generated config forward + matched loss | **pass** for `HGQ_ATTN=torch` and for the default config with dense attention (phi sort on, fix round 2), float and quantized: 17 finite loss terms (layer_0-3 and final). **deferred to R4b** for the default `linformer` config: strict xfail (`AssertionError`, Helen's layer refuses k=64 < 160 keys under the decoder mask) | — |
| torch/hybrid state_dict key + shape check | **pass** (dense, float and quantized): 184 torch entries, every element covered exactly once by 292 hybrid entries (counted per element since fix round 1); quantized adds only quantizer/beta/ebops entries (deviation 5). every relation must be the one its hybrid key name implies (fix round 2). Linformer rows strict xfail (R4b) | m12 (final linear not ported), f1 (x2 ported from the x1 chunk), g8, g9, g10: both rows fail |
| Sorting mechanics | **pass**: round trip exact on a non-identity sort; position-stub encoder parity 7.15e-7 (tol 1.43e-6), with prefix padding and with padding between valid tokens (fix round 1); `kv_sort_idx` = torch `sort_index` at every `q_ca`, `None` at `q_sa`/`kv_ca`, in hybrid and torch | unsorted hybrid 2.54 / 2.60; no `input_sort_field` gives `None`; m3, m4, m5, f3 |
| `HEPATTN_BRANCH` | **pass**: `repo_dir.sh` defaults (the branch of its own clone since fix round 2) and overrides; `env.sh.example` keeps a command-line branch; all `.pbs` parse and use `$HEPATTN_DIR`; setup (run offline from a local origin with a stub `uv`) refuses to switch an existing clone's branch, refuses a clone its jobs would not run in, leaves the keras-hgq2 clone alone, runs `uv sync --locked` except on keras-hgq2, and requires masked_linformer except on keras-hgq2 | the right-branch control passes the guard; m9, m10, g2-g7 |
| `tests/keras` + `tests/polaris` (hybrid env) | **pass**: fix round 2 (`25d8b36`, `$S/r4a/logs/fix2.hybrid.log`) 188 passed, 8 skipped, 6 xfailed, 0 failed, 0 xpassed; fix round 1 (`af84706`) 177 passed, 8 skipped, 6 xfailed, 0 failed, 0 xpassed; `a07ed1e` only reorders two asserts (state_dict rows re-run: 2 passed, 2 xfailed). R4a final (`921d87b`) 169; R0 119 / 8 | strict xfails: a premature R4b pass would be `XPASS(strict)` = FAILED |
| `tests/keras/test_pr8_layer_options.py` (carried from R0) | **pass** (part of the suite) | — |
| Native `tests/keras_native` (not changed by this lane) | 314 passed, 1 skipped, 41 xfailed, 0 xpassed: same as R0 (fix rounds 1 and 2 re-run: same) | — |
| Torch `tests/models tests/polaris tests/loss` | 522 passed, 173 skipped, 0 failed: same as R0 (fix rounds 1 and 2 re-run: same) | — |
| `prek run --files <changed>` | pass | — |

## Mutation check

Runner `$S/r4a/mut/run_mutants.sh`, mutant list `$S/r4a/mut/mutants.py`. Each mutant is applied to a `git archive HEAD`
copy (never the worktree) and the named tests run there with `PYTHONPATH=<copy>/src`. Results `$S/r4a/mut/results.txt`.

| Mutant | Change | Killed by |
|---|---|---|
| m1 | x1/x2 halves swapped in `port_dense` | 7 (5 dense rows, full-model parity seeds 0 and 1) |
| m2 | silu on x1 instead of x2 | 7 (dense rows, introspection, swapped-halves control, quantized leaf classes) |
| m3 | encoder does not unsort | 2 (round trip, position stub) |
| m4 | sort ignores `kv_mask` (padding not last) | 1 (position stub) |
| m5 | `input_sort_field` not passed to `MaskFormer` | 3 (`q_ca` sort index, 2 build rows) |
| m6 | generator default `HGQ_ATTN=torch` | first run: only a fixture error in `test_sorting.py`; after `921d87b`: `test_default_is_the_linformer_model` fails (`$S/r4a/mut/m6b.log`) |
| m7 | matcher stays `lap1015_late` | 6 (2 org-model rows, 4 build rows) |
| m8 | no `EinsumOp`/`SoftmaxOp` registration | 2 (both float build rows) |
| m9 | one clone dir for every branch | 3 |
| m10 | setup branch guard removed | 1 |
| m11 | SwiGLU product as a torch multiply | 4 (3 liveness rows, per-leaf EBOPs) |
| m12 | `port_dense` skips the final linear | 2 (state_dict float and quantized) |
| m13 | R0's `03_make_config.py` | 11 of 11 `test_polaris_config.py` rows |

## Deviations from the plan

1. **Full-model parity runs with dense attention.** The org model has masked-Linformer attention everywhere, and that
   core is lane R4b (plan section 6). R4a compares the hybrid built from the generated Polaris YAML with org torch
   built from `base.yaml` + `linformer.yaml`, both switched to dense attention by the same edit
   (`org_model_utils.to_dense_attention`). Everything else (SwiGLU, FastLayerNorm, `legacy`, `ca_sa_dense`,
   `self_attn_norm_kv`, phi sort, tasks, matcher, losses) is the org model. The Linformer rows are strict xfails with
   exact `raises=` types (`AttributeError`: `port_attention` finds no `q_proj` on Helen's layer; `AssertionError`:
   Helen's layer refuses k = 64 < 160 keys under the decoder mask), so R4b must turn them on.
2. **`EinsumOp`/`SoftmaxOp` registration** (not in the plan). Building the float model of a generated config through
   jsonargparse, as `main_hgq.py` does, fails in Lightning's `save_hyperparameters` deep copy:
   `TypeError: Could not locate class 'EinsumOp'`. This is not caused by R4a: at `5db27a8` the committed
   `pflow_keras_float.yaml` fails the same way on `encoder_l0_attn_scores` (probe `$S/r4a/probes/probe_jsonarg.py`,
   run from `experiments/clic`). The classes live in `keras_native/factory.py` (R2). I registered them in the hybrid
   shim `keras/factory.py`, guarded so that a later registration in `keras_native` is not overridden; see open issues.
3. **`HGQ_ATTN` variable in `03_make_config.py`** (not in the plan). `linformer` (default) is the plan's config;
   `torch` writes `base.yaml` alone (the clic-paper tag model, dense attention, no sort). It is the only generated
   config whose forward runs before R4b, so the "forward + matched loss" gate runs on it now.
4. **`HEPATTN_DIR` and `polaris/repo_dir.sh`** (the plan names only `HEPATTN_BRANCH`). Plan section 2 item 6 says
   running jobs use `$WORK_ROOT/hepattn` on `keras-hgq2`, so another branch needs another clone. Every `.pbs`
   hard-coded `REPO="$WORK_ROOT/hepattn"` and would have run the new branch's jobs in the `keras-hgq2` clone. Now
   setup and jobs read one rule; the default reproduces the old paths. The setup also refuses to switch the branch of
   an existing clone (it used to `git checkout keras-hgq2`), and installs other branches with `uv sync --locked`.
5. **"state_dict key and shape check" made concrete.** Hybrid keys cannot equal torch keys (keras names its
   variables), so the test traces every hybrid entry to its torch source by value: every torch floating tensor is
   refilled with distinct random values, the model is ported, and each hybrid entry must equal exactly one torch
   entry as the same tensor, its transpose, or one chunk of a packed one (`in_proj` in 3, SwiGLU in 2). Checks:
   reused torch modules keep their exact key; ported leaves stay under their torch module path; every torch entry is
   covered exactly once; in quantized mode the only extra entries are quantizer variables (`k`, `i`, `f`, `b`,
   `i_decay_speed`), `beta` and `ebops`. Dense float: 184 torch entries, 292 hybrid entries, 0 unmatched.
6. **SwiGLU low spec.** The shared `LOW_QUANT` (`kbi b0=4 i0=1`) rounds every weight of a 256-wide layer to 0
   (U(+-1/16) init), so its output does not depend on the input and per-leaf liveness reads 0.0 (measured). The SwiGLU
   tests use `kbi b0=5 i0=0` (75% of weights nonzero) and also assert the low output is correlated with float.
7. **Lint debt.** `07_profile.py` and `03_make_config.py` were cleaned only as far as `prek` asked for the files this
   lane changed; `ruff format` reflowed `07_profile.py`.

## Measurements behind the tolerances

- Full model, dense attention, metric `max |hybrid - torch| / (1 + |torch|)` over seeds 0-3
  (`$S/r4a/logs/measure_org_parity.out`): outputs 1.18e-5, 8.26e-6, 9.60e-6, 1.13e-5 (all at the final mask
  logits); losses 5.3e-8, 6.9e-8, 8.9e-8, 1.70e-7. Tolerances 2.4e-5 and 3.4e-7 (2x). All 4 per-layer attention-mask
  booleans equal. Controls (seed 0): `ca_dense_sa` 15.6 / 3.49e-2; `self_attn_norm_kv=False` 5.35 / 3.27e-3;
  `hybridnorm` refused (`post_norm mismatch`); no sort 9.29e-6 / 1.18e-7 (invisible to dense attention, as expected).
- Sorting stub (position marks added before encoder layer 0): 7.15e-7, tolerance 1.43e-6; unsorted hybrid 2.54.
- SwiGLU float parity (rank 3 / rank 2): 2.4e-7 / 1.2e-7 (27 -> 256), 3.4e-7 / 3.3e-7 (256). Swapped halves 3.05e-1.
- Quantized SwiGLU, 3 org shapes x seeds 0-2 (`$S/r4a/probes/probe_swiglu_q2.py`): high 1.12e-2 .. 1.30e-2
  (tolerance 2.6e-2); low 1.45e-1 .. 7.32e-1, ratio 11.1x .. 61.5x; low/float correlation 0.87 .. 0.98; per-leaf
  input-quantizer bypass changes the low output by 5.86e-2 .. 6.68e-1. The high floor is HGQ2's default bias
  quantizer, which the factory scopes do not set: a plain SiLU `Dense(256)` has the same floor (1.31e-2).

## Fix round 1 (review of R4a)

Evidence in `$S/r4a_fix/` (`LOG.md`, `probes/`, `logs/`, `mut/`). Mutants were applied to a `git archive` copy of
`af84706` by `$S/r4a_fix/mut/run_mutants.sh`, which also re-ran m1-m12: every one is still killed (m6 by
`test_default_is_the_linformer_model`, re-run in its copy: 1 failed). Results: `$S/r4a_fix/mut/results.txt`.

| Review issue | Change | Proof that the test now discriminates |
|---|---|---|
| (major) state_dict "covered exactly once" counted only element totals per torch entry | `state_map` keeps an int cover count per element; the chunk of a relation (`chunk j/n`) adds 1 to that chunk's rows only; `check_state_map` requires every element = 1, and the test then requires the `chunk*/2.T` and `chunk*/3.T` relations | f1 (`gates[i]` gets `w1, b1`): both dense rows fail on the coverage check (`input_nets.0.net.net.0.weight`: 1458 uncovered, 1458 covered twice; `$S/r4a_fix/mut/f1_x2_gets_w1.reorder.log`). Before: passed (review) |
| (major) SwiGLU products not checked in the model-level QAT walks | `test_swiglu_products_are_trained_with_the_model`: generated `HGQ_ATTN=torch` config, quantized; all 18 product leaves (1 input net, 6 encoder, 8 decoder, 3 task nets) are `QEinsum`, are in `keras_layers()`, have their trainable variables in `trainable_parameter_groups()`, and their EBOPs terms are in `quant_losses()` | f2 (`self.products = products`, the review's r7): fails (`innet0_net_hidden0_glu`, ... not in `keras_layers()`) |
| (minor) high-spec SwiGLU row blind below the bias-quantizer floor | `test_quantized_swiglu_high_spec_without_bias`: biases set to 0, tolerance 8.6e-5 = 2x the max of 2.96e-5 .. 4.27e-5 (3 shapes x seeds 0-2, `$S/r4a_fix/logs/swiglu_zero_bias.out`); the biased row is kept | f5 (quantized-only 1% product scale): the 3 zero-bias rows fail (4.97e-3 .. 7.58e-3); the biased paired-delta rows still pass, which confirms the review's finding |
| (minor, reported twice) the `kv_mask` gather after sorting is never exercised | `test_sorting.py`: the position-stub parity and its control also run with every third token padded (`test_interleaved_fixture_is_not_a_prefix` guards the fixture). Measured 7.15e-7 (same as prefix), unsorted control 2.60 | f3 (gather deleted, the review's r1): the interleaved row fails (0.813); the prefix row passes, as the review said |
| (minor) OUT_TOL headroom | Re-measured seeds 0-7 (`$S/r4a_fix/logs/measure_seeds.out`): outputs 1.18e-5, 8.26e-6, 9.60e-6, 1.12e-5, 1.44e-5, 1.21e-5, **1.65e-5**, 1.09e-5. OUT_TOL = 3.3e-5. Losses: the maximum for the same seed changes between runs (seed 0: 5.3e-8 at R4a, 1.60e-7 now; outputs reproduce to 4 digits); the max over all runs is 1.70e-7, so LOSS_TOL stays 3.4e-7 | controls unchanged: `ca_dense_sa` 15.6 and `self_attn_norm_kv=False` 5.35 are >= 10x OUT_TOL |
| (minor) generator-structure defects killed only through fixture errors | `test_generator_merge_is_a_deep_overlay` takes `merge()` from the generator source with `ast` (importing the script would write a config), checks a nested case, that the input is not changed, and equality with the test-local `merge` on base + linformer | f4 (`out[key] = value`): the assertion fails (not a fixture error) |
| (minor) `15_g2_forward.py --mode jax` ignores `--linformer-k` until R2 | `check_native_linformer_k` raises `SystemExit` when a native `LinformerAttention` has another `k`; it finds nothing once R2 removes that class | probe in `.venv-jax` (`$S/r4a_fix/logs/g2_k_guard.out`): `--linformer-k 256` accepted; `--linformer-k 64` refused, the native model has k = 256 |
| (minor) BetaPID setpoint describes the old model | `03_make_config.py`: comment and `OLD_MODEL_EBOPS`; the printout says the 1.6755e15 reference is pre-retarget and, for the Linformer config, warns that the setpoint is not re-derived | open issue below |

## Fix round 2 (second review of R4a)

Evidence in `$S/r4a_fix2/` (`LOG.md`, `probes/`, `logs/`, `mut/`). Mutants g1-g13 were applied to a `git archive` copy
of `25d8b36` by `$S/r4a_fix2/mut/run_mutants.sh`, which also re-ran m1-m12 and f1-f5 there: all 30 are killed
(m6 now lists `test_default_is_the_linformer_model`, which fails; the `test_sorting.py` fixture error remains).
Results: `$S/r4a_fix2/mut/results.txt`. All 10 review items were verified (the reviewer's probes and mutants re-read)
and fixed; none were pushed back.

| Review issue | Change | Proof that the test now discriminates |
|---|---|---|
| (major) SwiGLU EBOPs checks pass without EBOPs: `leaf.losses` also holds quantizer regularizer terms (3.3e-3 each) | `ebops_term(leaf)` requires `leaf.ebops > 0`, `beta > 0` and exactly one loss term equal to `beta * ebops` (rel 1e-5). Per-leaf test: x1, x2, product and final leaves. Model walk: beta is first set to 1e-6 on every `keras_layers()` entry, as `BetaScheduler`/`BetaPID` do (the generated config has `beta0: 0`, so the term would be 0), then every one of the 18 products must count EBOPs | g1 (the review's x1: product called without `training=`): both tests fail, `ffn_hidden0_glu: ebops 0.0` and `innet0_net_hidden0_glu: ebops 0.0`. Before: 12 passed (review) |
| (major) `env.sh.example` set `HEPATTN_BRANCH=keras-hgq2` unconditionally, so a command-line `keras-jax` became keras-hgq2 and setup pulled and rebuilt `$WORK_ROOT/hepattn`; jobs (no `qsub -v`) read only `env.sh` | `repo_dir.sh`: when `HEPATTN_BRANCH` is unset, it is the branch of the clone that holds `repo_dir.sh` (`git symbolic-ref`, only if `.git` exists at that root), else keras-hgq2. `env.sh.example` no longer sets it (a commented `${HEPATTN_BRANCH:-...}` form). `01_setup_env.sh` checks, before rebuilding the venv, that `$REPO/polaris/env.sh` + `repo_dir.sh` with an empty environment resolve to `$REPO`, and stops otherwise. Header comments, `USAGE.md`, `polaris/README.md` say that jobs read only the clone's `env.sh` | g2 (the old example line restored): 4 rows fail, including the review's row (example `env.sh`, `HEPATTN_BRANCH=keras-jax`) and the setup rows that assert the keras-hgq2 clone gets no `env.sh` copy. g3 (no derivation): 5 rows. g4 (no job-clone check): 3 rows |
| (minor) `--locked` and the masked_linformer check untested | `run_setup` now clones from a local bare origin, so setup passes `git pull`; the stub `uv` records its argv and its `venv` makes a python that runs the real verification heredoc on fake `torch`/`keras`/`hgq`/`hepattn` modules (`-S`, no site-packages), with or without a fake `masked_linformer` | g5 (never `--locked`, the review's g_no_locked): 4 rows; g6 (`if False:` around the re-raise): 1 row; g7 (always `--locked`): the keras-hgq2 row |
| (minor) LOSS_TOL headroom 1.24x | Seeds 0-15, 3 runs in one process and 1 in another (`$S/r4a_fix2/logs/loss_seeds.out`, `loss_seeds_proc2.out`): every repeat gives the same value to 4 digits; max outputs 1.855e-5, losses 2.75e-7 (both seed 12). OUT_TOL 3.7e-5, LOSS_TOL 5.5e-7 (2x) | controls: `ca_dense_sa` 15.6 / 3.49e-2 and `self_attn_norm_kv=False` 5.35 / 3.27e-3 stay >= 10x both tolerances |
| (minor) state_dict check accepts any chunk relation | `expected_relation(hkey)`: `_x1`/`_x2` -> chunk 0/1 of 2, `_q_proj`/`_k_proj`/`_v_proj` -> chunk 0/1/2 of 3, `/kernel` transposed, `/bias` and non-keras state the same | g8 (the review's x4, bias halves swapped), g9 (q and k swapped in `port_attention`), g10 (square `out_proj` kernel not transposed): each fails both dense rows with a named key; per-element coverage alone passes all three |
| (minor) generated quant values not tested | `QUANT_SPEC` pinned in `test_generated_model_is_the_org_model` (unchanged since `5db27a8`) | g11 (the review's x13, weight b0 8 -> 4): 2 rows fail |
| (minor) USAGE.md said `HGQ_ATTN=torch` "trains" | now "builds and runs a forward pass and the matched loss"; no optimizer step is tested | doc only |
| (minor) no quantized forward + loss with phi sorting | `test_generated_config_forward_and_matched_loss[linformer_dense_attn-*]`: the default config with `to_dense_attention`, float and quantized, asserts `input_sort_field == "phi"` and 17 finite terms | g13 (sort gather broken): both new rows fail, the `torch` rows pass |
| (minor) 07_profile / 15_g2_forward describe the pre-retarget model | docstring note in both; open issue below | doc only |
| (minor) deep merge equals LightningCLI's replace rule only while the overlay repeats every base key | `test_overlay_dicts_that_replace_repeat_every_base_key`: each overlay dict must be classified as merged (class specs, `init_args`, `decoder`) or replaced (`encoder.init_args.attn_kwargs`, `decoder.decoder_layer_config`), and a replaced dict must hold every base key | g12 (a new key in base `decoder_layer_config`): this test fails; `test_generator_merge_is_a_deep_overlay` still passes, as the review said |

Deviation (fix round 2): the plan says `HEPATTN_BRANCH` defaults to `keras-hgq2`. It now defaults to the branch of the
checkout the script runs from, with keras-hgq2 as the fallback. Reason: the new setup and job scripts exist only on
keras-jax and later, so with a fixed keras-hgq2 default a plain `bash polaris/01_setup_env.sh` from a keras-jax
checkout updates and rebuilds the production clone (the review's failure mode), and a job submitted from a keras-jax
clone runs in the keras-hgq2 clone. A keras-hgq2 checkout (or no clone) still resolves to `$WORK_ROOT/hepattn`.

## Open issues

- **R4b**: turn on the 6 strict xfails (`test_org_model_parity.py`: 2 parity, 2 state_dict; `test_polaris_config.py`:
  2 forward + matched loss). The state_dict check then needs torch `proj_k`/`proj_v` (160, 64) in `q_sa` to map to the
  hybrid (plan section 5.6, tail rows). `test_generated_config_builds` reads Helen's `proj_k` attribute; R4b updates
  that line if the attribute moves. Until R4b, the default Polaris config builds but cannot run a forward pass:
  do not launch Linformer jobs from this branch.
- **R2 (or R5)**: move the `EinsumOp`/`SoftmaxOp` keras registration into `keras_native/factory.py` (R2-owned); the
  hybrid shim then skips its own (guarded). keras_native under JAX has the same deep-copy exposure if anything copies
  a model.
- **R3/R5**: after this lane merges, `polaris/03_make_config.py` writes the org model (k=64, phi sort, FastLayerNorm,
  PR #8 decoder, scipy). The native `test_from_config_reads_generated_polaris_yaml` (strict xfail
  `raises=NotImplementedError`) stays xfail at this lane (native `MaskFormer` raises `NotImplementedError` for
  `input_sort_field` first), but its `check_clic_structure(model, 0.0)` expectations must match the new YAML when R3
  removes the marker.
- **R5 (goldens)**: SwiGLU float state_dict keys changed (`_hidden<i>` -> `_hidden<i>_x1`/`_x2`/`_glu`). No golden
  or committed checkpoint uses SwiGLU (`grep` of `tests/keras_native/golden/*.py`); quantized-SwiGLU golden cases
  (plan section 6) should use these names or map them.
- `export.build_functional_dense` still refuses SwiGLU nets (S7).
- Jobs get no environment from `qsub`: they resolve the clone from `polaris/env.sh` and `repo_dir.sh` in
  `PBS_O_WORKDIR`, so submit from the clone the setup made. A custom `HEPATTN_DIR` must be in `env.sh` (setup stops
  otherwise). `repo_dir.sh` calls `git` in jobs; not yet run on a Polaris compute node.
- `polaris/07_profile.py` and `15_g2_forward.py` still build the pre-retarget model (no SwiGLU, no FastLayerNorm,
  k = seq_len = 256, no sorting); their numbers do not describe the org/main model until R5 or G2 updates them.
- The verification heredoc of `01_setup_env.sh` is tested with fake packages only (import lines and the
  masked_linformer rule), not with a real venv.
- The HGQ2 default bias quantizer bounds the high-spec error (about 1.3e-2 on a Dense(256)); the factory scopes do
  not configure `place="bias"`. Not changed here (R2 owns the factory).
- Evidence (`$S/r4a/`: probes, mutant copies and logs) is in `/private/tmp`, which is not durable; R5 may copy it
  into `port_log/retarget/` like R0 did.
- **Before any QAT sweep on the new model**: re-measure total EBOPs on the R4b model and re-derive `HGQ_TARGET_EBOPS`
  (1.0e15), `HGQ_INIT_BETA` and `HGQ_PID_I`. They and the 1.6755e15 reference come from the pre-retarget model, whose
  EBOPs were 99.99% the masked-softmax fill term (plan section 7, risk 1). The generator prints a warning.
- **G2 (`15_g2_forward.py`)**: `--mode jax` with `--linformer-k` other than 256 exits until R2 makes native
  `build_attention` read `linformer_k`. Default runs (k = 256) are unaffected.
- **Carried gate**: re-measure the full-model `OUT_TOL`/`LOSS_TOL` (and the SwiGLU zero-bias tolerance) on the first
  Linux/GPU run. Fix round 2 could not reproduce the earlier report of run-to-run variation on one seed: 4 runs of
  seeds 0-15 in 2 processes gave the same values to 4 digits (the earlier 5.3e-8 for seed 0 came from the R4a tree).
- Nothing was pushed; no Polaris job was launched; the `hepattn` main working tree was not touched.
