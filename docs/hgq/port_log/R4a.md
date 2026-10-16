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

## Gates

Suite logs `$S/r4a/logs/v1.*` (runner `$S/r4a/run.sh`; hybrid and torch with `hepattn/.venv`, `KERAS_BACKEND=torch`,
native with `hepattn-keras-jax/.venv-jax`, `KERAS_BACKEND=jax`; `PYTHONPATH=<worktree>/src`, and `hepattn` resolves to
`hepattn-rt-r4/src`). `v1` ran on the tree of `316cdb3`; `921d87b` adds one test, re-run in `$S/r4a/logs/final.hybrid.log`.
The R1-R3 lanes ran their suites on the same machine at the same time, so wall times are long.

| Gate | Result | Control (makes it fail) |
|---|---|---|
| Hybrid float parity vs org torch, full model (2x measured) | **pass** for dense attention, seeds 0 and 1: outputs <= 2.4e-5, losses <= 3.4e-7 (metric and numbers below), 4/4 attention-mask booleans equal. Linformer rows: strict xfail (`AttributeError`), R4b (deviation 1) | PR #8 controls at model level: `ca_dense_sa` 15.6 / 3.49e-2, `self_attn_norm_kv=False` 5.35 / 3.27e-3 (>= 10x), `hybridnorm` refused. Mutants m1, m2 |
| Quantized-SwiGLU paired delta | **pass**, 3 org shapes: high <= 2.6e-2 (measured <= 1.30e-2), low >= 10x high (11x-62x), low/float corr > 0.5; leaf classes `QDense, QDense, QEinsum`; each leaf's input quantizer live (>= 5.86e-2); EBOPs > 0 and differentiable per leaf | m2 (silu on x1), m11 (product not through the einsum: 4 rows fail) |
| Generated Polaris config builds, float and quantized | **pass**, `linformer` and `torch`: built by jsonargparse with `MPflowHGQ` (as `main_hgq.py`); `proj_k` shapes {(168, 64), (160, 64)} (R0 carried gate); FastLayerNorm, `ca_sa_dense`, `self_attn_norm_kv`, phi sort, scipy solver, SwiGLU leaves. The YAML equals the org model section up to the documented edits; the default is the Linformer model | R0's generator (`5db27a8`): 11/11 rows fail (`$S/r4a/mut/m13.log`). Mutants m5, m6, m7, m8 |
| Generated config forward + matched loss | **pass** for `HGQ_ATTN=torch`, float and quantized: 17 finite loss terms (layer_0-3 and final). **deferred to R4b** for the default `linformer` config: strict xfail (`AssertionError`, Helen's layer refuses k=64 < 160 keys under the decoder mask) | — |
| torch/hybrid state_dict key + shape check | **pass** (dense, float and quantized): 184 torch entries all covered exactly once by 292 hybrid entries; quantized adds only quantizer/beta/ebops entries (deviation 5). Linformer rows strict xfail (R4b) | m12 (final linear not ported): both rows fail |
| Sorting mechanics | **pass**: round trip exact on a non-identity sort; position-stub encoder parity 7.15e-7 (tol 1.43e-6); `kv_sort_idx` = torch `sort_index` at every `q_ca`, `None` at `q_sa`/`kv_ca`, in hybrid and torch | unsorted hybrid 2.54; no `input_sort_field` gives `None`; m3, m4, m5 |
| `HEPATTN_BRANCH` | **pass**: `repo_dir.sh` defaults/overrides; all `.pbs` parse and use `$HEPATTN_DIR`; setup refuses to switch an existing clone's branch (run offline on a copy with a stub `uv`) | the right-branch control passes the guard; m9, m10 |
| `tests/keras` + `tests/polaris` (hybrid env) | **pass**: final (`921d87b`) 169 passed, 8 skipped, 6 xfailed, 0 failed, 0 xpassed (`v1`: 168; R0: 119 / 8) | strict xfails: a premature R4b pass would be `XPASS(strict)` = FAILED |
| `tests/keras/test_pr8_layer_options.py` (carried from R0) | **pass** (part of the suite) | — |
| Native `tests/keras_native` (not changed by this lane) | 314 passed, 1 skipped, 41 xfailed, 0 xpassed: same as R0 | — |
| Torch `tests/models tests/polaris tests/loss` | 522 passed, 173 skipped, 0 failed: same as R0 | — |
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
- `HEPATTN_BRANCH` must be set in `polaris/env.sh` (or exported) for both setup and job submission; jobs read it
  from the `env.sh` in `PBS_O_WORKDIR`.
- The HGQ2 default bias quantizer bounds the high-spec error (about 1.3e-2 on a Dense(256)); the factory scopes do
  not configure `place="bias"`. Not changed here (R2 owns the factory).
- Evidence (`$S/r4a/`: probes, mutant copies and logs) is in `/private/tmp`, which is not durable; R5 may copy it
  into `port_log/retarget/` like R0 did.
- Nothing was pushed; no Polaris job was launched; the `hepattn` main working tree was not touched.
