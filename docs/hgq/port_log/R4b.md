# R4b log: hybrid masked-Linformer core, gate H0 (retarget lane R4b)

Plan: `docs/hgq/RETARGET_ORG_MAIN.md` section 4 (option (a), gate H0) and the R4b row of section 6. Worktree
`hepattn-keras-jax`, branch `rt/integration` (R1-R4a merged, start `dc6a706`). Date 2026-10-15, macOS arm64, CPU,
torch 2.10.0, keras 3.15.0, HGQ2 0.1.9; hybrid env `hepattn/.venv` (no jax: `find_spec("jax") is None`;
masked_linformer dce2e15 installed), native env `.venv-jax` (JAX 0.11.2). `$S` is the scoping scratchpad; this lane's
evidence is in `$S/r4b/` (`LOG.md`, `probes/`, `logs/`, `mut/`).

## What landed

| Commit | Content |
|---|---|
| `4da3e5e` | `keras/attention.py`: `KerasAttention(attn_type="linformer", linformer_seq_len, linformer_k, linformer_core)`. It keeps the dense path's q/k/v/out projections, torch qkv norms and value residual, gathers k, v, `kv_mask` and the key axis of `attn_mask` by `kv_sort_idx` in torch (org `_linformer_attention`), and runs the keras_native `AttendCore` / `ProjectCore` under the torch backend. The core is built at the first forward (`build_core`, inside `factory.scopes()`); `"auto"` resolves from that call (attend with an `attn_mask`, else project), as org decides per call. q_sa's dead rows (org `proj[150:]`) are buffers `proj_{k,v}_dead`. `load_projections` holds ported projections until the core exists; `full_projections` returns org's (seq_len, k) layout. `build_keras_attention` raises `TypeError` for `linformer_proj_dim` and unknown options (Helen's layer is no longer routed to). `keras/decoder.py`: q_sa gets `linformer_core="project_unmasked"`. `keras/porting.py`: `port_attention` ports proj_k/proj_v |
| `5ab1c1c` | `tests/keras/test_linformer_parity.py` rewritten (gate H0 + module parity); `linformer_backend_cases.py`, `make_linformer_backend_golden.py`, `golden/linformer_backend.npz` (536 kB, written under JAX); `test_build_attention_linformer_k.py` rewritten |
| `94190b9` | The 6 strict R4b xfails of R4a turned on (`test_org_model_parity.py`, `test_polaris_config.py`), with own linformer tolerances, a sorting control and the state_dict trace for row blocks |
| `e9a9d1f` | `keras/callbacks.py`, `keras/maskformer.py`: beta and EBOPs callbacks reach nested HGQ2 leaves (deviation 3); new `test_beta_reaches_every_leaf.py` |
| `299c700` | Redundant warm-start copy removed (mutant m09 survived it); test that lazily built cores take the factory spec; `USAGE.md`, `org_model_utils.py` updated |

## Gate H0 (plan section 4)

H0 passes, so the core is reused and not rewritten in hybrid code (the 5.6 fallback is not needed).

| H0 item | Result | Evidence; control / mutant that fails it |
|---|---|---|
| core runs under `KERAS_BACKEND=torch` without jax | **pass** | `test_core_runs_under_torch_with_jax_blocked`: subprocess with a meta-path hook that makes `import jax` raise; hybrid linformer attention float and quantized, `AttendCore` and `ProjectCore` with tail rows; no `jax*` module loaded. Control `test_the_jax_blocker_works` (`keras_native.matching` fails under the hook). The suite itself runs in `hepattn/.venv`, which has no jax. m06 (core imports jax): every R4b test file errors at collection |
| float parity with `masked_linformer.attend` / `project` at <= 2x the JAX-side measurement | **pass** | Same fixtures and seeds as `tests/keras_native/test_masked_linformer.py`; tolerances = R2's (2x the JAX measurement). Torch backend vs JAX side (R2): attend forward 8.345e-7 vs 8.345e-7; gradients q 7.153e-7 vs 8.345e-7, k 3.874e-7 vs 3.278e-7, v 3.576e-7 vs 3.576e-7, E 1.907e-5 vs 4.244e-5, F 1.717e-5 vs 5.913e-5; project enc 1.013e-6 vs 1.013e-6, enc_short 5.960e-7 vs 5.960e-7, q_sa 0.0 vs 2.980e-7. Max ratio to the JAX value 1.18 (k gradient), inside 2x. All-False rows and gradients into padded keys exactly 0. Control: masked_linformer without renorm is 0.647 (attend) / 0.921 (project) away, >= 1e4x the tolerances |
| quantized: torch-backend output == JAX-backend output, same weights and inputs, 2x measured | **pass** | Golden written under JAX (`make_linformer_backend_golden.py`), inputs pinned by sha256. 5 call sites (q_ca, kv_ca, enc, enc_short, q_sa) x {high, prod} x {eval, train}: max difference 0.0 on all 20 outputs, so tolerance 0 (plan section 5: a row measured at 0 keeps 0; re-measure on Linux/GPU). Control: one datalane bit less moves the output >= 2.84e-4, and the float core is >= 3.56e-2 from the quantized golden (`test_backend_golden_resolves_one_datalane_bit`) |
| every core variable a registered torch parameter that gets optimizer updates | **pass** | Generated Polaris model (production spec), materialized as `MPflowHGQ.setup`: 18 cores, 1840 variables, all in `model.parameters()` and the state_dict; the 404 trainable ones in `trainable_parameter_groups()` (bitwidths in the quantizer group, kernels/tails in the decay group), no non-trainable one. One AdamW step (weight decay 0, task losses + EBOPs): 404/404 have finite nonzero gradients and move. m05 (core outside the module tree): 4 tests fail. SGD at lr 1e-2 moves only 190/404 (EBOPs gradients of 1e-8 below float32 resolution), so the test uses AdamW, as `test_optimizer_updates_weights.py` |

## Other gates

| Gate | Result | Evidence |
|---|---|---|
| hybrid attention vs org `Attention` (float) at the 5 call sites, qkv_norm, perturbed affines, sort | **pass** | max 6.855e-7 (q_ca), tolerance 1.37e-6 (2x). Distinct k/v, value residual (both layers), batch of one (ones and None): <= 4.768e-7. Controls (`$S/r4b/logs/controls.out`): v = k 0.326 / 0.334, no sort 0.333, masks not gathered 0.0855, no value-residual mix 0.0780, mask of ones vs None at n < seq_len 0.120 (8.9e-8 at n = seq_len) |
| exactness | **pass** | padded tokens (NaN in float, +100 at the high spec) leave valid outputs bit-identical, a valid token moves them; a key outside query 0's mask leaves query 0 bit-identical, every query that sees it moves (float and high); org q_sa rows 150-159 are dead (exactly 0 change), row 149 is not |
| quantized paired delta (hybrid module) | **pass** | q_ca: high 2.146e-2 (tolerance 4.29e-2), coarse 0.7650, 35.7x; native measured 2.147e-2 / 0.7650 on the same case |
| full model float parity vs org (base + linformer) | **pass** | seeds 0-15 (`$S/r4b/logs/model_parity_seeds.out`): outputs max 5.088e-6, losses max 1.875e-7; tolerances 1.02e-5 / 3.75e-7; 4/4 attention-mask booleans equal for every seed. Controls (seed 0): no sort 3.47 / 2.39e-2 (0/4 masks equal), `ca_dense_sa` 3.25 / 4.97e-4, `self_attn_norm_kv=False` 2.41 / 5.57e-4 |
| state_dict key and shape trace (float and quantized) | **pass** | every org proj_k/proj_v (2 x 18) covered exactly once: whole (encoder 168, q_ca 160), head 150 + core tail 10 (kv_ca), head 150 + q_sa buffer 10. m02 (E/F swapped), m03 (q_sa rows not ported), m05, m10, m12 (no buffers) fail it |
| generated config forward + matched loss, default Linformer config | **pass** | float and quantized, 17 finite loss terms; cores built as org's path at every call site: (Project, 168, 168), (Attend, 150, 160), (Project unmasked, 150, 150), (Attend, 160, 150); projections (168, 64) / (160, 64) |
| beta/EBOPs callbacks reach every leaf (deviation 3) | **pass** | with beta from BetaScheduler or BetaPID, the EBOPs loss grows by delta-beta x total EBOPs (rel 1e-3), linformer and dense; monitor = sum of each leaf's own EBOPs. m13, m14, m15 fail it |
| `tests/keras` + `tests/polaris` (hybrid env) | **pass** | see "Suites" |
| prek on changed files | **pass** | |

### Mutants

Runner `$S/r4b/mut/run_mutant.sh`, list `$S/r4b/mut/mutants.py`: each mutant is applied to an rsync copy of the worktree
(never the worktree) and the R4b targets run there (`test_linformer_parity`, `test_build_attention_linformer_k`,
`test_beta_reaches_every_leaf`, `test_beta_pid`, and the linformer rows of `test_org_model_parity`,
`test_polaris_config`). Unmutated copy: 107 passed (`m00`). Results `$S/r4b/mut/summary{,2}.txt`.

| Mutant | Failing targets |
|---|---|
| m01 no `kv_sort_idx` gather | 6 (q_ca module rows, sort control, paired delta, full model seeds 0 and 1) |
| m02 E and F swapped in `load_projections` | 18 |
| m03 q_sa rows not ported | 3 (q_sa layout, state_dict float and quantized) |
| m04 q_sa left on `"auto"` (masked project core) | 7 (optimizer step: the 8 q_sa tail variables get no gradient; state_dict x2; build and forward rows x4) |
| m05 core outside the module tree | 4 (registration, optimizer step, state_dict x2) |
| m06 core imports jax | all 4 files error at collection (no jax in the hybrid env) |
| m07 `kv_mask` not merged into the attend mask | 6 (incl. padding exactness float and high) |
| m08 core built outside the factory scopes | 2 (paired delta, factory-spec test) |
| m09 core keys left out of the warm-start state_dict load | 1 (warm start). Before `299c700` the mutant was "no explicit core copy" and survived: the state_dict load already did it, so the explicit copy was removed |
| m10 `"auto"` always attend | 21 |
| m11 `linformer_proj_dim` dropped instead of raising | 2 |
| m12 no q_sa buffers | 6 |
| m13 callbacks walk top-level layers only | 6 |
| m14 `quantized_leaves` top-level only | 4 |
| m15 monitor reads the `ebops` property (QSoftmax adds its tables twice) | 4 |

## Suites (final, HEAD `299c700`; logs `$S/r4b/logs/final.*`)

| Suite | Result | Before R4b (R-merge) |
|---|---|---|
| hybrid `tests/keras` + `tests/polaris` (`hepattn/.venv`, torch backend) | 263 passed, 8 skipped, 0 xfailed, 0 failed, EXIT 0 | 188 passed, 8 skipped, 6 xfailed (strict R4b) |
| native `tests/keras_native` (`.venv-jax`, unchanged by this lane) | 457 passed, 50 xfailed (strict R5), 0 failed | same |
| torch `tests/models tests/polaris tests/loss` (`not gpu and not requiresdata`) | 522 passed, 173 skipped, 0 failed | same |

`tests/matching` was not re-run: this lane changes no file it imports.

## Deviations from the plan

1. **Lazy core build in float mode too.** The core has static (q_len, kv_len) shapes that only the first call knows
   (org builds nothing shape-dependent), so it is built at the first forward in float and quantized mode. The hybrid's
   float layers used to be built at construction. `MPflowHGQ.setup` already runs a forward before the optimizer and
   the state_dict key set exist, so training is unaffected; tests that port right after construction rely on
   `load_projections` holding the tensors until the build (`state_map` now runs a forward first in float mode too).
2. **The core kind is chosen per call site, from the first call.** Org chooses attend or project per call; a static
   keras graph cannot. `"auto"` resolves at the first call from the presence of `attn_mask` (org's rule), and the
   decoder fixes q_sa to `"project_unmasked"`, the path org always takes there. A later call of the other kind raises.
   With mask attention off, q_ca/kv_ca resolve to project, as org; kv_ca then has unused tail rows (not production).
3. **Callbacks fixed (not in the plan).** BetaScheduler, BetaPID and EBOPsMonitor walked only top-level keras layers,
   but each HGQ2 leaf adds `beta * ebops` with its own beta. Measured on the generated model at initial bitwidths
   (`$S/r4b/logs/beta_walk.out`): the Linformer model has 377 leaves, 165 top-level; the walk missed 212 (every core
   leaf), 26.5% of the EBOPs, and the EBOPs readout gave 2.117e11 of 2.509e11. The same walk on the dense-attention
   model missed the 38 QSoftmax exp/inv tables, which hold 99.99% of its 1.907e15 EBOPs: on the pre-retarget model
   beta never reached the dominant EBOPs term. That is consistent with the earlier observation that `val/ebops` did not
   move across four decades of beta (`callbacks.py` BetaPID docstring), but it was not re-run here. The fix changes
   what the existing beta settings do on the dense path too.
4. **Golden with 2 heads.** The backend-consistency golden uses 2 heads instead of 16 at the production sequence
   shapes (536 kB instead of about 4 MB). A 16-head probe (`$S/r4b/probes/h0_backend.py`) was exact except one query
   row of 16 elements in 3 of 20 outputs (up to 1.28e-4 at the high spec): rounding ties that the two backends' float32
   sums break differently. The committed rows measure 0.0, so their tolerance is 0.
5. **`*.npz` is in `.gitignore`.** The golden was added with `git add -f` (the R0 file `.gitignore` is not this lane's;
   keras_native's goldens use a negation line there). A tracked file stays tracked; a negation line for
   `tests/keras/golden/*.npz` is an R5/R0 item.
6. **Module parity tolerance 1.37e-6** (2x the measured 6.855e-7), not the native 1.25e-6: the hybrid uses torch qkv
   norms and keras Dense under torch, so it was measured on its own.
7. **q/k/v init unchanged** (keras glorot on (256, 256)): the hybrid ports weights from torch or warm-starts from the
   float twin; the native port uses org's packed xavier (R2). Only from-scratch hybrid QAT init statistics differ.

## Open issues

- **Before any QAT sweep on the new model:** total EBOPs of the generated Linformer model at initial bitwidths is
  2.509e11 (2 events), against 1.907e15 for the same model with dense attention. `HGQ_TARGET_EBOPS` 1.0e15 is far above
  the whole Linformer model, so BetaPID would drive beta to `min_beta`. Re-derive `HGQ_TARGET_EBOPS`, `HGQ_INIT_BETA` and
  `HGQ_PID_I` (`polaris/03_make_config.py`) on this model, now that beta reaches every leaf (deviation 3).
- `quant_losses()` counts the task heads' EBOPs terms twice (keras_layers yields them under `tasks.` and
  `decoder.tasks.`, S3 deviation 9); the new callbacks count every leaf once, so `val/ebops` drops by the task heads'
  share (2.7e10 on the dense model, 1.4e-5 relative). Not changed here.
- R5: regenerate the hybrid goldens for the new attention (plan section 6 table); delete `keras/linformer.py` and
  `port_linformer` once `tests/keras_native/golden/make_golden*.py` stop importing them (they are the only users).
- R5/R0: `.gitignore` negation for `tests/keras/golden/*.npz` (deviation 5).
- Carried gate: re-measure the H0 backend rows (tolerance 0) and the new tolerances on the first Linux/GPU run.
- `polaris/07_profile.py`, `15_g2_forward.py` still build the pre-retarget model (R4a open issue, unchanged).
- Evidence (`$S/r4b/`) is in `/private/tmp`, which is not durable.
- Nothing was pushed; no Polaris job was launched; the `hepattn` main working tree was not touched.
