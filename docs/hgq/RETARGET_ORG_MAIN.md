# Plan: retarget the HGQ2 model to the org/main float Linformer model

Status: plan, 2026-10-15. Nothing here is implemented yet. Branch `keras-jax` (local HEAD `37ff127`; `origin/keras-jax` is `197ede2`).
Request (user, 2026-10-15, `hepattn/WORKLOG.hgq.md:159`): fix bugs 1 and 2, edit the HGQ2 implementation to match
the latest org float Linformer model, then push. This plan does not push.

Path names used below:
- `$KJ` is this worktree (`hepattn-keras-jax`).
- `$S` is `/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad`.
- `$O` is `$S/orgmain`, a snapshot of org/main `ac0da8f`.
- Org `file:line` references are on org/main `ac0da8f` under `src/hepattn/`. Native references are `src/hepattn/keras_native/` at `37ff127`.
  Hybrid references are `src/hepattn/keras/` at `37ff127`.

Evidence comes from three investigations in this workflow, plus planner checks:
- Spec and gap list: `$S/spec_gap_log.md`, probes in `$S/specgap/`.
- Hybrid trial merge: `$S/trial-logs/LOG.md`, scratch worktree `$S/trial-merge` (branch `scratch/trial-merge`, HEAD `f110dcc`), probes in `$S/trial-probes/`.
- Quantized masked-Linformer design: `$S/qml_proto/LOG.md` and the prototypes next to it.
- Planner checks: `$S/retarget_plan_log.md`.

## Summary

- **Target.** org/main `ac0da8f` = `configs/linformer.yaml` over `configs/base.yaml`: PR #9 (masked-linformer in every attention, k=64,
  phi sort) on PR #8 (the clic-paper tag model), with PR #3 (bug 2), `a4263c1` (bug 1) and PR #1 (matching). keras_native cannot
  build this model today: 8 `from_config` blockers, and `linformer_k` is silently ignored (k=256 instead of 64).
- **Bugs 1 and 2** reach torch and the hybrid through the merge without conflicts. The hybrid reuses the torch task and loss code,
  so it gets both fixes for free. keras_native copies the old maths and must change (section 3, about 250 LOC with tests).
- **Hybrid:** recommend option (a), update it to the new model. Reuse the native quantized masked-Linformer core inside it
  (section 4). The recommendation depends on one user decision (section 7, Q1). That decision blocks only lane R4.
- **Work:** 6 stages. After the merge stage, 4 lanes run in parallel with disjoint file ownership. Total about 2,150-2,550 LOC
  (estimate, section 6).
- **Corrections to earlier notes:**
  - `WORKLOG.hgq.md:156` says the PR #5 cos/sin bug is not on keras-jax. That is wrong. `pflow_data.py:240-241` on keras-jax
    reads `torch.cat([track_cosphi, topo_phi], -1)` (sinphi likewise), the same as org/main.
  - The merged `uv.lock` is stale: `uv lock --check` exits 1 on the trial merge and 0 on `37ff127`
    (`$S/trial-logs/uv_lock_check{,_baseline}.log`). The trial investigation did not report this.

## 1. Target, and what changes compared with keras-jax

### 1.1 Pin

| Item | Value | Evidence |
|---|---|---|
| Model | org/main `ac0da8f`, CLIC, `base.yaml` + `linformer.yaml`. The overlay's key sets contain all of base's keys, so merging and replacing give the same effective config | `$S/specgap/probe_chain.py` |
| Why this pin is "the latest float Linformer model" | On 2026-10-15 the live org `main` is `ac0da8f` (`git ls-remote org`), PR #9 merged 2026-10-01. It is newer than this user's own Linformer branches (`linformer-float` `731db27`, last commit 2026-09-17; `linformer-into-unified-main` `9a86b47`, 2026-09-24), which carry Helen's Linformer path. Org PR #2 (that path) is open but superseded by #9. masked-linformer `main` is still `dce2e15` | `git ls-remote org`; `git log -1 --format=%cd`; `gh pr list -R compressed-maskformer-reco/hepattn --state all`; `gh api repos/compressed-maskformer-reco/masked-linformer/commits/main` |
| Commits in it, not on keras-jax | 20 (`git log 1df05cc..org/main`): #9 `ac0da8f`, #8 `e9ef1a6`, #3 `a2bc9c7`, `a4263c1`, #1 (`84dd040`..`cfc40a0`), CI #4 | `git log` |
| masked-linformer | `dce2e15` (MIT), pinned in org `pyproject.toml:42` as a GitHub archive URL. Only dependency: `torch>=2.0` | `$S/masked-linformer/pyproject.toml:8` |
| Data | `pflow_data.py` is unchanged by the retarget. The PR #5 bug (`topo_phi` used as topo cos/sin) is in the target, and PR #5 is still open. Torch and native read the same data, so parity is not affected | `pflow_data.py:240-241` on both trees; https://github.com/compressed-maskformer-reco/hepattn/pull/5 |

### 1.2 The model the HGQ2 implementation must reproduce (float)

| Part | org/main behaviour | Evidence |
|---|---|---|
| Input | `node_features` (B,160,27). Net: `Dense(27→256, SwiGLU)` = Linear(27,108), SwiGLU, Linear(54,256). `SwiGLU(x) = x1·silu(x2)`, where x1 is the first half. Fourier posenc on (eta, phi) is added after the net | `activation.py:6-9`, `input.py:53-57`, `posenc.py:118-138` |
| Sorting | `key_phi` = concatenation of `node_phi`. `key_sort_idx = argsort(masked_fill(~valid, +inf))`: padding last, not a stable sort. The encoder sorts, gathers and unsorts internally. `kv_sort_idx` goes only to decoder `q_ca` | `maskformer.py:116-134`, `encoder.py:22-26,222-296`, `decoder.py:316,465` |
| Encoder | 6 layers, dim 256, 16 heads, 8 register tokens (L=168). FastLayerNorm (non-affine, eps 1e-5), `hybrid_norm`, `dense_norm_placement=legacy`. Layer 0: `x += Attn(FLN x)`, then `x = FLN x; x += FFN x`. Layers 1-5: `x += Attn x`, then `x += FFN(FLN x)`. `qkv_norm` on q, k, v at full width. Value residual in layers 1-5. FFN `Dense(256, SwiGLU)` | `norm.py:19-27,64-99`, `encoder.py:56-99,137-149` (Residual placement), `attention.py:273-303` |
| Attention | Packed `in_proj_weight` (768,256), xavier init, bias 0, plus `out_proj`. Linformer `proj_k`, `proj_v` (seq_len, 64), U(±1/8). With `kv_sort_idx`: gather k, v and both masks first. `attn_mask` None: `project` (where-zeroing + column renorm when `kv_mask` is given) then unmasked SDPA. Otherwise: `attend(q,k,v,E,F, attn_mask & kv_mask)` | `attention.py:202-213,317-331`; masked_linformer `project` L39-53, `_rescale` L56-63, `attend` L66-87 |
| Decoder | 4 layers, 150 queries. Mask `sigmoid(logit).detach() >= 0.1`; all-False rows restored to all-True. Order `ca_sa_dense`, `self_attn_norm_kv=True`. Call sites: `q_ca` = attend, mask (B,150,160), `kv_mask`, `kv_sort_idx`; `q_sa` = project with no masks (uses `E[:150]`); `kv_ca` = attend, mask (B,160,150), no `kv_mask`. No value residual | `decoder.py:293-296,451-493`, `task.py:569-575` (threshold); measured in `$S/specgap/orgmain_forward.out` |
| Batch of one, all valid | If `batch_size == 1` and every key is valid, `key_valid` becomes `None` (not for flash-varlen). The encoder then calls `project` with `mask=None`: `E[:n]`, **no renorm**. A mask of all ones is not the same as no mask when `n < seq_len`. Production never takes this path: events with ≥ 160 nodes are dropped, so at most 159 of 160 nodes are valid. Test fixtures at small shapes can take it, and native always passes the mask, so parity fixtures must give every event at least one padded node. Add one B=1 all-valid test that asserts the documented difference at `n < seq_len` and equality at `n = seq_len` | `maskformer.py:112-113`; masked_linformer L16-18, L45-46; `experiments/clic/pflow_data.py:120` |
| Tasks | Classification at every layer (`has_first_layer_loss`), mask at every layer, incidence and regression at the final layer only. Classification `Dense(256→6)`, hidden [256,128,32], SiLU. Mask `4·object_net(q)·constituent_net(kv)`, both SwiGLU Linear(256,1024) → Linear(512,256). Incidence `net` 256→256→256→256 SiLU, `node_net` SwiGLU Linear(256,512) → Linear(256,256). Regression `mode=scale`, `use_nodes`, input 518 → 512→256→128→64→32→5, SiLU, output `net·proxy` | `task.py:201-217,552-567,1340-1347,1465-1640` |
| Costs | `2·object_ce_cost(logits)` (bug 2 fix), `1·mask_dice_cost`. At the final layer also `1·kl_div_cost` and `10·L1` | `task.py:240-251`, `loss.py:41-60,96-118` |
| Matcher | `nan_to_num` (nan → big, +inf → big, -inf → -big; big = f32max/10). Solver `lap1015_late`, with fallback to scipy on an invalid permutation; parallel, `n_jobs` 16; `device_solver` off | `matcher.py:125-147,290-332`; `base.yaml:150-162` |
| Losses | `2·CE` with class weights [1,3,8,1.5,1] and null weight 0.5, over all 150 queries (`mask` is ignored). `5·mask_bce` and `1·mask_dice` normalised **per event** (bug 1 fix). `1·kl_div` and `10·L1` at the final layer. The total is the sum over layer_0..3 and final | `loss.py:36-38,63-93,202-241`, `wrapper.py:46-63` |
| Trainer (not model) | Lion, LR 1e-6 → 8e-5 → 1e-6, wd 1e-4, clip 0.1, bf16-mixed, batch 512 | `base.yaml`; out of scope here, see Q-defaults in section 7 |

### 1.3 Gaps in keras_native at `37ff127` (measured in the spec investigation)

| Component | Native today | Evidence |
|---|---|---|
| `linformer_k` | **Silently ignored.** `build_attention` pops only `linformer_proj_dim`, so k=256 is built | `attention.py:200-201` |
| Linformer attention | Helen's layer: bias-free `to_q/k/v`; `qkv_norm`, `value_residual`, `kv_mask` and `q_mask` ignored; mask applied in the rank-k space. Padding leak measured at 130.75 (org: 0.0). Raises when k=64 < kv_len=160 | `linformer.py:95,117-122`; `$S/specgap/probe_pad_native.py` |
| Sorting | Raises | `maskformer.py:76`, `encoder.py:213-215` |
| FastLayerNorm, `dense_norm_placement` | Missing. Placement is hard-coded to hybridnorm | `norm.py:8-25,32` |
| Decoder order, `self_attn_norm_kv`, `kv_sort_idx` | `ca_dense_sa` hard-coded; `q_sa(q, k=q, v=q)`; no sort index | `decoder.py:106-116,241-243` |
| Quantized SwiGLU | Raises `NotImplementedError` | `dense.py:91-92` |
| object_ce cost | Re-softmaxes the probabilities: cost range 0.187 against org 0.881 | `objective.py:130-133`, `losses.py:49-59` |
| mask_bce, mask_dice | Cross-event broadcast: 1.3090 against org 1.4038, and 0.89237 against 0.89392 | `losses.py:8-17,113-154`; `$S/specgap/probe_losses.py` |
| Matcher | scipy only. `lap1015_late` raises. No `nan_to_num` | `objective.py:345-361`, `matching.py:45-56` |
| `from_config` | 8 blockers, in order: `input_sort_field`; encoder `class_path` not unwrapped; duplicate `dim`; encoder `dense_norm_placement`; FastLayerNorm; decoder PR #8 keys; quantized SwiGLU; `lap1015_late` | `$S/specgap/probe_chain.py` |
| Init | glorot per (256,256) kernel. Org uses xavier on the packed (768,256) matrix | spec investigation; org `attention.py:202-204,230-233` |
| bf16 | Not ported (fp32 first, `KERAS_JAX_PLAN.md` section 3.7) | |
| Unchanged | Input net in float, posenc, registers, mask attention and restore, task forwards and widths, CE / kl_div / L1, total | spec investigation (org `task.py` forwards changed only in the cost since keras-jax) |

The hybrid after the merge: the torch forward, tasks, losses and matcher come from org/main. Hybrid-specific gaps (trial merge):
- the PR #8 attributes on the hybrid layers (fixed by the trial in `f110dcc`, +18/−5);
- `x_sort_value` rejected (`keras/encoder.py:132-133`);
- quantized SwiGLU raises (`keras/dense.py:106-107`);
- Helen's Linformer only (`keras/attention.py:196-201`, same silent `linformer_k` bug);
- `polaris/03_make_config.py:189-202` hard-codes the old encoder block.

## 2. Merge plan (stage R0)

1. Branch `rt/integration` from `keras-jax` in a sibling worktree `../hepattn-rt-int`, the same layout as `hepattn-kn-a`/`-b`.
   Copy `.serena` into each new worktree and re-index.
2. `git merge org/main` (`ac0da8f`). Resolve the conflicts as the trial did (`$S/trial-logs/LOG.md`):
   - `src/hepattn/models/attention.py`: take **theirs** (org) unchanged. The hybrid imports Helen's torch class directly
     (`keras/porting.py:131-133`), and `models/linformer.py` does not conflict, so it stays until R5 deletes it.
   - `.gitignore`: the union of both sides.
   - `pixi.lock`: take theirs, then run `pixi lock`. Org's lock has 0 `hgq2`/`keras` lines; keras-jax's has 25 (grep counts).
     pixi is installed locally (`/opt/homebrew/bin/pixi`). If solving for the linux platforms fails on macOS, run it on a linux host.
   - `uv.lock`: **not a textual conflict, but stale.** Run `uv lock`, then require `uv lock --check` to exit 0. Before:
     exit 1, "needs to be updated", 227 packages against 226. Without this, Polaris `uv sync` (`01_setup_env.sh:68`) re-locks on the login node.
3. Apply the hybrid hunks of `f110dcc` with a proper commit message:
   - `KerasMaskFormerDecoderLayer` and `KerasEncoderLayer` accept and store `dense_norm_placement`, `query_update_order` and
     `self_attn_norm_kv`.
   - `build_keras_attention` reads `linformer_k`.
   - `03_make_config.py` writes `linformer_k`.
4. Native tests: mark the 41 tests that fail after the merge (list in `$S/trial-logs/final.native.log`) with
   `pytest.mark.xfail(strict=True, reason="<stage that fixes it>")`. The groups:
   - 29 `linformer_proj_dim` TypeErrors → R2/R5;
   - 11 objective assertions → R1;
   - 1 quantized-SwiGLU `from_config` → R3.

   `strict` turns a forgotten marker into a failure once its stage lands. Each lane removes the markers only in the files it owns.
5. Interface stub, so that lanes R2 and R3 do not edit the same lines: native `Attention.call` and `LinformerAttention.call`
   accept `kv_sort_idx=None` and ignore it. Org's dense attention ignores it too (`attention.py:317-331` uses it only in the
   Linformer branch).
6. Environment (one time):
   - Local, already done by the trial (verified: `import masked_linformer` works with `-I` in both venvs). To recreate:
     `uv pip install --python <venv>/bin/python --no-deps "masked-linformer @ https://github.com/compressed-maskformer-reco/masked-linformer/archive/dce2e151e9d25cefd85454b3dffd5c83ec217097.tar.gz"`
     for `$KJ/.venv-jax` and `hepattn/.venv`. Record it in `docs/hgq/keras_native_env.txt` (`uv pip freeze`) and in the
     setup section of `KERAS_NATIVE.md`. `--no-deps` keeps torch 2.10.0, as in the trial.
   - Polaris hybrid env: use a **separate clone**, because jobs that are running use `$WORK_ROOT/hepattn` on `keras-hgq2`
     (`01_setup_env.sh:38-41`):
     `git clone --branch keras-jax https://github.com/erdemyigit/hepattn.git $WORK_ROOT/hepattn-kj && cd $WORK_ROOT/hepattn-kj && uv venv --python 3.12.0 .venv && uv sync --group hgq --no-install-project --locked --python .venv/bin/python`.
     masked-linformer comes from GitHub, so the login node needs outbound HTTPS. `--locked` fails loudly if `uv.lock` is stale.
   - Polaris `.venv-jax`: the line in `KERAS_JAX_PLAN.md` section 5.1, plus the masked-linformer line above. The latter is needed
     only to run the torch parity tests there; the JAX training path does not import it.
   - lap1015: Polaris installs with `--no-install-project`, so `import lap1015` gives an empty namespace package and fails at the
     first match (`AttributeError: lap_late`, reproduced in the trial). R4 makes the generated config use `scipy`.
     lap1015 is not importable locally in either venv (`ModuleNotFoundError`).
7. Keep the evidence: copy `$S/spec_gap_log.md`, `$S/specgap/*.py`, `$S/trial-logs/LOG.md`, `$S/trial-probes/*.py` and
   `$S/qml_proto/{LOG.md,*.py,*.out}` and `$S/review_rt/{LOG.md,*.log,*.out}` into `docs/hgq/port_log/retarget/`. `/private/tmp` is not durable, and R2 ports
   `qml_layers.py`.
8. Remove the scratch worktrees `$S/trial-merge` and `$S/baseline-kj` and the branch `scratch/trial-merge`. They share this repository's `.git`.

**R0 gate** (commands as in `$S/trial-logs/run.sh`):
- (a) Hybrid `tests/keras` + `tests/polaris`: 107 passed / 8 skipped, as the trial measured at `f110dcc`.
- (b) Native `tests/keras_native`: 311 passed / 41 xfailed / 2 skipped / 0 failed.
- (c) Torch `tests/models tests/polaris tests/loss -m 'not gpu and not requiresdata'`: 0 failed. The trial ran
  models + polaris: 310 passed / 173 skipped. `tests/loss` brings org's `test_loss_per_event_normalisation.py`, and
  `tests/models` brings `test_object_classification_cost.py`.
  `tests/matching` cannot be green locally: lap1015 is not built, so `PYTHONPATH=src` makes `src/lap1015` an empty namespace
  package. Measured (critic, `$S/review_rt/loss_matching{,_baseline}.log`): 106 failed on the trial merge and 105 on `37ff127`,
  every one `AttributeError: module 'lap1015' has no attribute 'lap_late'/'lap_early'/'releases_gil'`; `tests/loss` had 0 failures.
  Gate: every `tests/matching` failure is one of these lap1015 AttributeErrors (grep the log). Otherwise build lap1015 first
  (org `pyproject.toml:55-68`) and require 0 failed.
- (d) `uv lock --check` exits 0.

Between R0 and R4, `polaris/03_make_config.py` writes a mixed model: old encoder, PR #8 decoder and tasks. Quantized builds of it
fail on SwiGLU (trial). Do not launch Polaris jobs from keras-jax in that window.

## 3. Bugs 1 and 2 (stage R1, first lane to land)

### 3.1 How they arrive
- Torch: `a4263c1` changes `models/loss.py` (`_mean_over_valid_objects`, `mask_bce_loss`, `mask_dice_loss`, and the focal and
  `mask_kl_div` losses, which CLIC does not use). `a2bc9c7` changes `ObjectClassificationTask.cost` to read
  `outputs[self.logits_key]` (`task.py:240-251`). Neither file conflicts in the merge.
- Hybrid: `keras/tasks.py` subclasses only `ObjectHitMaskTask` (line 19) and `IncidenceRegressionTask` (line 44), and overrides
  no `cost` or `loss`. So the hybrid trains with both fixes as soon as R0 lands. No hybrid code is needed.
- Native: `losses.py` and `objective.py` copied the old torch maths on purpose (`losses.py:8-17`). They must change, and 11 tests
  that pinned the old semantics must be rewritten, not just unmarked.

### 3.2 Native changes
- `losses.mask_bce_loss` / `mask_dice_loss`: `per_object[b,p] = sum_n(loss·pad[b]) / max(count_b, 1)`, then
  `sum(per_object·valid) / max(sum(valid), 1)`. This mirrors `_mean_over_valid_objects` (`loss.py:202-207`). The
  `_cross_event_object_mean` helper and the `einsum("qpn,bn->bqp", …)` cross terms go. Dice keeps the unmasked target sum
  (`loss.py:88-90`). Rewrite the module docstring (`losses.py:8-17`).
- `objective.ObjectClassificationTask.cost`: pass `outputs[f"{output_object}_logit"]` instead of `_class_prob`
  (`objective.py:130-133`). `losses.object_ce_cost` then applies the single softmax.
- When no object is valid, native gives 0. For `mask_bce` / `mask_dice`, torch now also gives 0, because of `clamp_min(1.0)`
  (`loss.py:207`). The documented deviation in `losses.py:5-6` stays for the other losses: the regression L1 loss
  (`task.py:1590-1595`, boolean indexing then a mean) and `kl_div_loss` (`loss.py:271-277`) still give NaN on an empty
  selection in torch. Narrow the docstring; do not delete it.
- `docs/hgq/PARITY.md`: the note "`mask_bce` can be `inf` on randomly-initialized models" no longer holds. Padded-node terms are
  multiplied by that event's pad mask, which is 0. At random init the spec measured mask_bce ≈ 2.9, finite. Regenerate the file
  in R5 (`tests/keras/gen_parity_report.py`).
- This closes the carried gate "S4 → user" in `port_log/README.md`, and makes "S4 → S6: no target 1 at padded nodes" non-fatal.

### 3.3 Discriminating tests (new file `tests/keras_native/test_org_loss_fixes.py`)

Fixture F1 is `$S/specgap/probe_losses.py`, seed 0: B=3, P=150, N=160, valid nodes [120, 37, 159], valid objects [40, 9, 60],
mask logits 3·N(0,1) set to `finfo.min` at padded nodes, targets Bernoulli(0.05) on valid (node, object) pairs, class logits N(0,1) for 6 classes.

| Test | Asserts | Before (native `37ff127`) | After = org torch | Why it discriminates |
|---|---|---|---|---|
| B1-a | mask_bce, mask_dice equal org torch on F1 | 1.3090 / 0.89237 | 1.4038 / 0.89392; an independent numpy per-event reference gives 1.4038 | gaps 0.0948 and 1.55e-3, against a tolerance of 2x the measured difference (expected ~1e-7) |
| B1-b | Decomposition: `L([e0,e1,e2]) == Σ n_b·L([e_b]) / Σ n_b`, with n_b the valid-object count, to 2x measured | violated (a test-local copy of the old formula must show the violation ≥ 10x the tolerance) | holds by construction | per-event normalisation is the only thing that makes events independent |
| B1-c | Control: at B=1 the old and new formulas agree | equal | equal | shows B1-a/b react only to batch coupling |
| B1-d | Target 1 at a padded node with a `finfo.min` logit: finite and equal to torch | inf | finite | padding is multiplied by the event's own mask |
| B1-e | No valid object: 0, equal to torch | 0 (torch NaN) | 0 = 0 | replaces `test_no_valid_object_gives_zero_where_torch_gives_nan`. Before and after agree, so this is a regression pin; its control is a mutant that drops `max(sum(valid), 1)` (gives NaN) |
| B2-a | `ObjectClassificationTask.cost` equals the org task cost on F1 | max diff 0.5626 | 1.2e-7 when fed logits (probe rerun by the planner) | gap 0.56 against a tolerance of about 2.4e-7 |
| B2-b | ptp of the class cost on F1: native = torch; the `_class_prob` mutant compresses it ≥ 4x | 0.187 | 0.881 (4.7x) | |
| B2-c | 2×2 assignment. Logits rows [5,0,0,0,0,0] and [0,5,0,0,0,0], targets classes [0,1], dice [[.95,.30],[.30,.95]], cost `2·ce + dice`. Native and torch `Matcher` pick [0,1]; the double-softmax mutant picks [1,0] | [1,0] | [0,1] | measured with scipy on both cost formulas (planner check, `$S/retarget_plan_log.md`): totals −1.9696 vs 0.0747 |
| B1/B2-e2e | `test_objective_e2e.py`: matched MaskFormer loss equals the merged torch loss at the existing 1e-4/1e-3 rows; mask_bce finite | rows fail after the merge (3 tests) | pass | |

Mutation check, as in S4 (22/22 killed): in a scratch copy, revert each fix (cross-event broadcast; `_class_prob` input) and
record which new tests fail. Each mutant must be killed by at least 2 tests.

Rewritten after R1 (11 tests that pinned the old semantics; listed in `final.native.log`):
- `test_objective_parity.py`: `test_task_costs_equal_torch` (3), `test_task_losses_equal_torch` (2),
  `test_mask_losses_use_the_torch_cross_event_broadcast` (rename it to per-event), `test_no_valid_object_gives_zero_where_torch_gives_nan`,
  `test_mask_bce_sample_weight_with_non_unit_null_weight_equals_torch`;
- `test_objective_e2e.py::test_maskformer_loss_equals_torch` (3).

R1 also takes the matcher (section 5.7), because it owns `objective.py` and `matching.py`.

## 4. Hybrid decision

| Option | Work | Keeps | Loses |
|---|---|---|---|
| **(a) Update the hybrid to the new model** | 250-650 LOC including Polaris config and tests. Arithmetic on the trial estimate of 450-700: reusing the native core replaces the 150-250 LOC attention branch with about 50-100 LOC of glue. If reuse fails gate H0: 450-700 | Trainable new-model QAT on Polaris now, with the existing Lightning path. G2, S7 and G4 stay like-for-like. Independent quantized goldens for everything except the attention core | An independent quantized implementation of the masked-Linformer core |
| (b) Keep the hybrid at the old model semantics | About 50-60 LOC: `f110dcc` (+18/−5), pinning `03_make_config.py` to the pre-merge model (30-40, trial), and forcing scipy | Existing goldens; checkpoint compatibility with the old runs | No trainable new-model QAT until native S5+S6 (about 1,490 LOC estimated, `KERAS_JAX_PLAN.md` section 4). The S7 and G4 gates (`KERAS_JAX_PLAN.md:512,516`) compare against the hybrid and would have to be redefined. Not truly "old" either: the hybrid gets bugs 1-2 through the shared torch losses (section 3.1) |
| (c) Freeze at a tag (`hgq2-hybrid-legacy` at `37ff127`) and delete it from the branch | A tag; delete `src/hepattn/keras/` (1,950 lines, `wc -l`) and `tests/keras/` | A smaller branch | The fallback if G2 fails (`KERAS_JAX_PLAN.md:468-471`: if G2 fails, "stop before S4-S8 merge"; G2 has not been run). The S7/G4 references. Golden regeneration. Any QAT run before native S8 |

**Recommendation: (a).**
1. It is what the request says: the hybrid is the production HGQ2 path until G4 (`KERAS_JAX_PLAN.md` section 3.1).
2. G2 has not been run. If it fails, the hybrid is the only path.
3. S7 and G4 are defined against the hybrid.
4. Most of PR #8 comes free, because the hybrid inherits the torch forward:
   - after `f110dcc`, the suite is green (107 passed);
   - with PR #8 settings, hybrid and torch differ by ≤ 7e-7 in float (`$S/trial-probes/probe_pr8_hybrid.py`);
   - the controls discriminate: wrong order 0.41, `self_attn_norm_kv=False` at layer 0 0.13, wrong placement refused with "post_norm mismatch".

**Reuse, so the core is written once.** `keras_native/__init__.py` imports no submodule, and only `matching.py` imports jax
(`grep`). The hybrid already re-exports the native factory under the torch backend (`keras/factory.py:7`). So the new module
`keras_native/masked_linformer.py` can sit inside the hybrid's `KerasAttention`, in the same way as the HGQ2 leaves it already holds.

**Gate H0 (first task of R4b):**
- the core runs under `KERAS_BACKEND=torch` in `hepattn/.venv`, which has no jax;
- float parity with `masked_linformer.attend` / `project` at ≤ 2x the JAX-side measurement;
- quantized (high spec): torch-backend output equals the JAX-backend output for the same weights and inputs, at 2x measured
  (backend consistency, R9 of `KERAS_JAX_PLAN.md`);
- every core variable is registered as a torch parameter and gets optimizer updates (the pattern of
  `tests/keras/test_optimizer_covers_weights.py`).

If H0 fails, write the core in hybrid code from the leaf table in section 5.6.

**What the quantized oracle becomes:**
- Float: the org torch model in the same process, for every component. This is the primary oracle.
- Quantized:
  - hybrid golden files for everything except the attention core (independent code): projections, norms, SwiGLU (written
    independently in `keras/dense.py` to the same leaf rule), encoder and decoder wiring including sorting, tasks, the full model;
  - for the core: backend consistency (H0), the paired delta, high-bitwidth convergence, a leaf inventory with a liveness check,
    and the renorm-precision bound (section 5.6). None of these can be met by inert quantizers.

## 5. Native changes per component, with gates

**Tolerance policy:**
- Measure on the final implementation, then set `atol` / `rtol` to 2x the measured maximum (PARITY.md form
  `|a−b| ≤ atol + rtol·|b|`, valid slots only). Record the measured value next to each tolerance.
- A row measured at 0 keeps tolerance 0 and joins the carried gate "re-measure on the first Linux/GPU run" (`port_log/README.md`).
- A tolerance is accepted only together with a named control or mutant that exceeds it by at least 10x.
- Booleans (mask-attention masks), structural zeros (all-False rows, gradients into padded keys) and padding-invariance checks are exact.

**Shared JIT gate for every new layer and for the full model:**
- `jax.jit(jax.value_and_grad(stateless_call …))` keeps `_cache_size() == 1` over 3 calls;
- every gradient is finite;
- every trainable variable gets a nonzero gradient, except a documented list (none expected after section 5.6, item "tail");
- EBOPs are read as `sum(losses)/beta` > 0 for every new HGQ2 leaf. Do not read the uint32 `_ebops` (`KERAS_JAX_PLAN.md` section 3.6).

### 5.1 Configuration (R3; the full `linformer.yaml` part in R5)

Change:
- `from_config` unwraps encoder `class_path` / `init_args` and checks the encoder `dim` against the MaskFormer `dim`.
- It accepts `input_sort_field`, the PR #8 keys, FastLayerNorm, `activation: SwiGLU` in quantized mode, and
  `linformer_seq_len` / `linformer_k`.
- **Unknown keys raise.** No `**_ignored` anywhere in attention; this is the root cause of the silent k=256. `build_attention`
  is in `attention.py`, which R2 owns, so R2 makes this change (section 5.6); R3 only passes the keys through.
- A test helper loads `base.yaml`, then deep-merges `linformer.yaml`.

Gates:
- every one of the 8 blockers in `probe_chain.py` is gone (critic rerun: the 8 reproduce, and with all 8 patched the float model
  builds with k=256; run the probe with the clic directory as `argv[3]`, or `scale_dict_path` does not resolve);
- variable shapes and counts equal the torch model built from the same merged config: `proj_k` (168,64) in the encoder and
  (160,64) in `q_ca` and `kv_ca`. One listed exception: `q_sa` is (150,64) in native and (160,64) in torch
  (`$S/specgap/orgmain_forward.out`), because native drops the dead tail rows (section 5.6). This gate needs R2's Linformer
  branch, so R3 runs it after rebasing on the R2 merge (R3 merges after R2);
- `linformer_proj_dim` and a misspelt key raise. Control: at `37ff127` the same config builds k=256.

### 5.2 FastLayerNorm and `dense_norm_placement` (R3)

Change:
- `make_norm("FastLayerNorm")` = `keras.layers.LayerNormalization(epsilon=1e-5, center=False, scale=False)`. It has 0 trainable
  variables (checked locally).
- `get_hybrid_norm_config` gains `dense_norm_placement`, copied from org `norm.py:64-99`.
- Encoder and decoder layers pass the new keys through.

Gates:
- FastLayerNorm parity with torch. This alone does not discriminate: an affine LayerNorm at its default init (gamma 1, beta 0)
  gives the same output. The variable-count gate below is the control;
- encoder layer parity at depth 0 and depth 1 with `legacy`. Control: `hybridnorm` placement differs (measure it; the trial's
  hybrid porter refused the mismatch);
- variable count: native and torch FastLayerNorm have 0 variables, against 2 for LayerNorm.

### 5.3 Decoder order, `self_attn_norm_kv`, `kv_sort_idx` (R3)

Change: `query_update_order` (`ca_dense_sa` | `ca_sa_dense`), `_query_self_attn` as in org (`decoder.py:489-493`: with
`self_attn_norm_kv`, `q_sa` takes k and v from its normalised input), and `kv_sort_idx` passed to `q_ca` only.

Gates:
- decoder-layer parity at depth 0 and 1;
- controls at depth 0: wrong order (hybrid measured 0.41), `self_attn_norm_kv=False` (0.13);
- at depth ≥ 1, `self_attn_norm_kv` has no effect, as expected under hybrid norm (no pre-attention norm there). The hybrid
  probe measured 4.8e-7, the same as its matched-configuration floor (critic rerun of `probe_pr8_hybrid.py`). Assert that too,
  so the test documents why the control must be at depth 0.

### 5.4 Quantized SwiGLU (R3; independent twin in R4)

Leaf rule (one rule for both implementations):
- the packed `Linear(in, 2h)` becomes two `factory.dense` halves. x1 is the first half (`activation.py:6-9`); x2 has
  `activation="silu"` inside `QDense`, as every other quantized activation already does (`dense.py:110-118`);
- the product `x1·x2` is a `factory.einsum` elementwise equation (`QEinsum` when quantized). hls4ml generated code for an
  elementwise-broadcast `QEinsum` in the QML probe (`bhqk,bqk->bhqk`). `QMultiply` exists in HGQ2 0.1.9 (`hgq/layers/ops/merge.py:53`)
  but is not checked with hls4ml;
- float mode uses the same two-half graph ("one graph, two leaf sets").

Gates:
- float parity with torch SwiGLU (the PARITY row `swiglu_ffn` was 2.38e-7; re-measure). Mutant: swap the halves → fails;
- quantized: paired delta (high close to float, low ≥ 10x further) and liveness (the `QEinsum` and both `QDense` are live: forcing
  each quantizer to float changes the low-bit output);
- JIT gate;
- `test_from_config_reads_generated_polaris_yaml` passes again.

### 5.5 Input sorting (R3; discriminating end-to-end check in R5)

Change:
- `MaskFormer` builds `key_phi` and `key_sort_idx = keras.ops.argsort(where(valid, phi, +inf))`;
- `Encoder.call` sorts, prepends the registers, runs the layers, strips the registers and unsorts (org `encoder.py:222-296`);
- the decoder forwards `kv_sort_idx`.

Dense attention is permutation-equivariant, so sorting is invisible with `attn_type: torch`. **The R3 gates therefore test the
mechanics directly:**
- `sort_index` equals torch `sort_index` exactly on the valid prefix (the first `n_valid` entries of each event), and the padding
  suffix holds the same set of indices. Full equality is not achievable: torch argsort is not stable, so it orders the tied
  `+inf` padding entries differently from `keras.ops.argsort` (JAX `stable=True` default). Critic check on unique phi,
  B=4, N=160, valid [120,37,159,80]: full equal False, valid prefix equal True, padding sets equal True. Control: dropping
  the `+inf` fill interleaves padding with valid tokens and fails the prefix check;
- `unsort(sort(x)) == x` exactly, with an assert that the fixture's sort index is not the identity (else the round trip is vacuous);
- a position-marking stub layer (adds a bias that depends on the sorted position) gives the same output as the torch encoder with
  the same stub. Control: without the sort, the outputs differ.

R5 adds the discriminating end-to-end check with the Linformer encoder. Control: removing the sort changes the outputs; measure it.
`keras.ops.argsort` kept tied `+inf` entries in their input order (local check). Torch argsort is not stable, but ties among
padding do not change valid outputs: padded rows are zeroed in `project` and masked in `attend`. Padded output rows do differ,
so every parity comparison (here, in 5.8 and in the hybrid goldens) uses valid slots only.

### 5.6 Masked-Linformer attention (R2)

Change:
- Stop routing to `LinformerAttention` (Helen). R5 deletes the file once nothing imports it (today: `tests/keras_native/{golden_utils,porting,test_float_parity,test_init,test_masked_query}.py`). Add `attn_type="linformer"` as a branch of native `Attention`, mirroring
  `Attention._linformer_attention` (org `attention.py:317-331`): it reuses `q_proj`/`k_proj`/`v_proj`/`out_proj`, `qkv_norm`
  and the value residual (mixed before the projection), and adds `proj_k`/`proj_v`.
- New module `keras_native/masked_linformer.py`, ported from `$S/qml_proto/qml_layers.py`: `SeqProjection`, `rescale`,
  `AttendCore`, `ProjectCore`. `ExplicitCore` is dropped: it measured 7.62e10 EBOPs against 6.51e9 for the implemented form.
- The leaf constructors of the prototype's `Leaves` move into `LayerFactory` as `einsum_dense`, `lut` and `mask_einsum`.

Leaf table (QML design; the masks are `keras.ops.where`, with no EBOPs):

| Step | HGQ2 leaf |
|---|---|
| `S = q·kᵀ` | `QEinsum bhqd,bhnd->bhqn` |
| `S·E[:n]`, `A·F[:n]ᵀ` | `QEinsumDense` with kernel `E[:n]` (resp. `F[:n]`) and a weight quantizer |
| `× r_k`, `× r_v` | `QEinsum bhqk,bqk->bhqk` |
| softmax over the k slots | `QSoftmax`, **never with `mask=`** (the `-1e9` fill raises the exp-table input to 30 integer bits: 1.27e11 instead of 1.9e3, `probe_softmax_mask.py`) |
| renorm `M·W`, with `W = E²/Σ_rows E²` from the quantized rows | `QEinsum bqn,znk->bqk`: 1-bit mask lane, weight lane per element |
| rsqrt | `QUnaryFunctionLUT`, one homogeneous table, input reshaped to **rank 2** (a rank-3 input fails hls4ml conversion) |
| `·v` | `QEinsum bhqn,bhnd->bhqd` |

Specifics:
- **Tail rows.** `E[n:]` exists only when the renorm is on and `seq_len > n`. That happens only in `kv_ca` (n=150, seq_len=160);
  the tail enters only the renorm numerator. In `q_sa`, org calls `project` with `mask=None`, which uses `proj[:n]` and no rescale
  (masked_linformer L45-46). Org's rows 150-159 there get no gradient. Native does not create them, and porting drops them.
  Test: perturbing those torch rows changes no torch output (exactly 0). This keeps the "nonzero gradient" JIT gate free of exemptions.
  The hybrid (R4b) keeps torch's (160,64) shape in `q_sa`, so its state_dict keys and shapes still match torch (R4a gate);
  only native drops the rows, and the 5.1 shape gate lists that one exception.
- **Renorm precision.** Set the LUT input and the energy-lane fractional bits to `2·f_E + 4`, where `f_E` is the largest fractional
  bit count of E's weight quantizer. That is 16 at the production spec (`kbi b0=8 i0=2`, step 1/64).
  - Prototype at 12 bits: 18.5% max relative error at 1-2 valid keys, 0.9% at 20.
  - At 16 bits: 1.5% and 0.08% (`renorm_prec.out`).
  - hls4ml caps the table at 1e6 entries, so cap `f_E` at 7.
- `build_attention` reads `linformer_seq_len` and `linformer_k`, and raises on anything else.
- Init: q/k/v kernels U(±√(6/(4·dim))), bias 0 (xavier on the packed (768,256) matrix); `proj_*` U(±1/√k) (masked_linformer `init_`).
  Test the bounds.

Float gates (in-process against masked_linformer and org `Attention`; prototype measurements in brackets):
- `attend` forward [4.8e-7 f32; 7.8e-16 f64]. Gradients for q, k, v, E, F [≤ 4.8e-6]. All-False rows exactly 0. Gradient into
  padded keys exactly 0. Inputs carry NaN in padded rows.
- `project` + unmasked softmax attention [3.6e-7].
- Full `Attention(attn_type="linformer")` with `qkv_norm`, value residual and `kv_sort_idx`. Prototype through the HGQ2-structured
  layers: attend 5.4e-7, project 6.6e-7 at production shape.
- Exactness:
  - perturbing padded tokens changes valid outputs by 0.0;
  - perturbing a key outside query 0's mask changes query 0 by 0.0, while queries that see that key change (positive control).
    Org measured both.
- Mutants that must fail at least one gate:
  - from the QML list: mask by multiply (NaN); renorm numerator over `E[:n]` only; no final `where` (6.9e7 on all-False rows);
    renorm dropped; tail dropped;
  - new: no `kv_sort_idx` gather; Helen's mask in the rank-k space; `linformer_k` ignored.

Quantized gates:
- Paired delta at the high spec against the low spec, ≥ 10x. Prototypes: native attend at production shape, high 7.0e-3; hybrid prototype branch, high 7.6e-3 against low 2.3e-1 (30x).
- Convergence: the error against float falls monotonically over at least 3 bit settings. Raising the renorm lanes alone takes the
  high-spec error from 7.0e-3 to 6.1e-4 (prototype).
- Renorm bound: the max relative rescale error at 1 valid key, with the production E grid and the chosen lanes, stays at or below
  2x measured (the prototype measured 1.5% at 16 bits).
- Leaf inventory: in quantized mode, every multiply in the table is the listed HGQ2 class, and each leaf is live.
- JIT gate at the production spec (the prototype passed: one compile, 0 zero-gradient trainables, finite gradients, EBOPs > 0).
- Golden parity (R5) against the hybrid at H0. This is backend consistency only, see section 4.

### 5.7 Matcher (R1)

Change:
- `matcher_n_jobs_from_config` accepts `default_solver: lap1015_late` and solves with scipy. It logs once that the optimum is the
  same and ties may break differently.
- `host_match` applies `nan_to_num(nan=big, posinf=big, neginf=-big)`, with big = f32max/10, before solving (org `matcher.py:309`).

Gates:
- matched total cost equals org `Matcher(default_solver="scipy")` over 100 batches (rel 1e-6), including costs that contain NaN
  and −inf. Control: without the sanitising step, scipy raises `ValueError: matrix contains invalid numeric entries` (checked
  locally for NaN and −inf; +inf is accepted);
- rewrite `test_unsupported_matcher_configs_raise` and `test_matcher_n_jobs_from_pflow_hgq_yaml_equals_torch_matcher`.

### 5.8 Full model (R5)

Gates:
- native against the org torch MaskFormer built from `base.yaml` + `linformer.yaml` at test shapes, with ported weights: the
  per-layer outputs at the `maskformer.e2e` rows (5e-5/1e-4, re-measured); the mask booleans exactly equal per layer (float),
  on a fixture where every valid mask logit is at least 10x the logit tolerance away from `logit(0.1)` (assert this margin,
  or a 1e-6 difference can flip a boolean and the exact check fails for no model reason); every event has a padded node
  (section 1.2, "Batch of one, all valid");
  the matched loss at the `maskformer.loss` rows;
- controls: no sorting; `ca_dense_sa`; old losses. Each must exceed the tolerance by ≥ 10x;
- quantized full-model paired delta on the layer-0 mask logits (S3 measured 0.30 against 31.09);
- golden parity against the updated hybrid. Production WRAP rows are gated before the first mask feedback (S3 deviation 8);
- `test_lane_integration.py` and `test_mf_jit.py` on the new model: 1 compile over 3 batches, one host match per step.

## 6. Work breakdown

Each lane runs in its own worktree from the R0 commit and merges into `rt/integration` in the order R1 → R2 → R3 → R4b → R5.
R1, R2, R3 and R4a run in parallel. Each lane follows the S1-S4 practice: implementation, 2 reviewers with mutation-proven
discrimination, at most 2 fix rounds, and a log in `docs/hgq/port_log/R<n>.md`.

| Stage / lane | Content | Owns (only this lane edits) | Depends on | LOC (estimate) | Gates |
|---|---|---|---|---|---|
| R0 integrate | Merge, lock files, `f110dcc` hunks, strict xfails, `kv_sort_idx` stub, environment, evidence copy | everything before the lanes fork | — | ~60 own, plus the regenerated locks | section 2 |
| R1 objective | Bugs 1 and 2, matcher | `keras_native/{losses,objective,matching}.py`; `tests/keras_native/{test_objective_parity,test_objective_e2e,test_matching,objective_utils}.py`; new `test_org_loss_fixes.py` | R0 | ~250 | section 3.3, section 5.7; the 11 objective xfails rewritten |
| R2 attention | Masked-Linformer core and `Attention` branch, factory leaves; `build_attention` stops routing to Helen's Linformer | `keras_native/{attention,masked_linformer,factory}.py`; `tests/keras_native/{test_masked_query,test_jit,test_float_parity,test_import_hygiene}.py`; new `test_masked_linformer.py`; in `tests/keras_native/porting.py` only `port_attention`, `port_linformer`, `port_any_attention` | R0 | ~650 (prototype 293 lines) | section 5.6; import hygiene also under `KERAS_BACKEND=torch` without jax |
| R3 layers | Config, norms, placement, decoder order, quantized SwiGLU, sorting | `keras_native/{norm,dense,encoder,decoder,maskformer,input}.py`; `tests/keras_native/{test_maskformer,test_variables,test_init,test_helpers}.py` (`test_init.py` drops its Helen-Linformer checks; R2 tests the new init bounds); new `test_pr8_layers.py`; in `tests/keras_native/porting.py` all other `port_*` functions | R0 | ~500 | sections 5.1-5.5 |
| R4a hybrid | Quantized SwiGLU (independent, same leaf rule), sorting in `KerasEncoder`/`KerasMaskFormer`, `03_make_config.py` = base + linformer + HGQ settings with solver `scipy`, a `HEPATTN_BRANCH` variable in `01_setup_env.sh` (default `keras-hgq2`) | `src/hepattn/keras/**` (except deletions), `tests/keras/**`, `polaris/**`, `docs/hgq/USAGE.md` | R0 | 200-450 | Hybrid float parity against org torch for the full model (2x measured; PR #8 controls as in section 4); quantized-SwiGLU paired delta; the generated Polaris config builds in float and quantized and runs a forward + matched loss (`probe_polaris_build.py` pattern); torch/hybrid state_dict key and shape check |
| R4b hybrid core | Masked-Linformer branch of `KerasAttention` reusing `keras_native.masked_linformer`; rewrite `tests/keras/test_linformer_parity.py` | as R4a | R2 merged | 50-200 (450-700 for R4 overall if H0 fails) | H0 (section 4); `tests/keras` green |
| R5 integrate | Goldens, full-model gates, deletions, docs | `tests/keras_native/golden/**`, `tests/keras_native/{test_mf_golden,test_quantized_golden,test_lane_integration,test_mf_jit,conftest,golden_utils,mf_golden_utils}.py`; deletes `src/hepattn/models/linformer.py`, `keras/linformer.py`, `keras_native/linformer.py` (−127) once `grep` finds no user; `docs/hgq/**` | R1-R4b | ~450 | section 5.8; the native, hybrid and torch suites green with no xfails left (torch `tests/matching`: as R0 gate (c), only lap1015 AttributeErrors unless lap1015 is built); hls4ml code-generation smoke for the new leaf kinds (`KERAS_TORCH_DEVICE=cpu`); critic pass |

Total: about 2,150-2,550 LOC. These are estimates; reference sizes: QML prototype 293 lines, native `linformer.py` 127,
trial estimate 450-700 for the hybrid. After R5, fast-forward `keras-jax` to `rt/integration`. Push only when the user has
answered Q1 and seen the R5 log. `origin` is the **public** fork `erdemyigit/hepattn`.

**Tests and goldens to regenerate or rewrite:**

| What | Stage | Why |
|---|---|---|
| The 11 objective tests (section 3.3 list); `test_unsupported_matcher_configs_raise`, `test_matcher_n_jobs_from_pflow_hgq_yaml_equals_torch_matcher` | R1 | bugs 1 and 2; `lap1015_late` accepted |
| `test_float_parity.py::test_clic_linformer_encoder_parity`, `test_jit.py` (2), `test_masked_query.py` | R2 | Helen's Linformer is replaced; masked attend makes key independence exact |
| `test_maskformer.py` (7 failing plus `from_config`), `test_variables.py` (3), `test_init.py` | R3, R5 | PR #8 options, sorting, new variable lists (`proj_*`, no affine norms, SwiGLU halves) |
| `golden/lin_self.npz`, `lin_cross.npz`, `enc_lin.npz`, `mf_lin.npz`, `mf_measured.json`; `test_mf_golden.py` (12), `test_quantized_golden.py` (4) | R5 | regenerate from the updated hybrid with `cases.py` / `mf_cases.py` cases for project (encoder), attend with sort (`q_ca`), attend with tail (`kv_ca`), PR #8 encoder, and the full `linformer.yaml` model at test shapes. Re-measure; tolerances at 2x |
| `golden/attn_self`, `attn_cross`, `dense_ffn`, `dense_innet`, `innet`, `enc_torch`, `mf_torch` | R5 | keep (org defaults keep these configs valid: `decoder.py:360-362`, `encoder.py:113`). Re-run the generator and require bit-identical npz on macOS: a regression check that the hybrid edits left the old paths unchanged. Add quantized-SwiGLU cases |
| `tests/keras/test_linformer_parity.py`, `test_dense_parity.py`, `test_encoder_parity.py`, `test_maskformer_parity.py`, `tests/polaris/*` | R4 | masked Linformer, quantized SwiGLU, sorting, generated config |
| `docs/hgq/PARITY.md` | R5 | regenerate; drop the mask_bce `inf` note |
| `KERAS_JAX_PLAN.md`: section 3.2 Linformer row ("Keep Helen's semantics", line 190), section 3.3 mask normalisation, S2 Linformer gate, S4 "`mask_bce` `inf` semantics" (line 473), Status; `port_log/README.md` carried gates (close "S4 → user"; update "S4 → S6" and "merge → S5": the decoder softmax no longer takes a mask) | R5 | the plan must describe the new model |

## 7. Risks and open questions

**Q1 (blocks lane R4 only): option (a), (b) or (c) for the hybrid (section 4).** The recommendation is (a). Lanes R0-R3 do not
depend on the answer. With (b), R4 becomes about 60 LOC and S7/G4 need new references. With (c), G2 loses its fallback.

Decisions with a default (they do not block):
- PR #5 cos/sin bug: keep it, so that the model matches `ac0da8f` exactly. Fixing it is a shared data change, not model work.
- Trainer recipe for QAT (org: Lion 8e-5, batch 512, bf16; Polaris: AdamW 5e-5, clip 0.1, 32×4): decide before S5/S8.
  It does not change the model.
- Run G2 after R5 on the new model in both arms, not on the old model, because the attention compute changes.

Risks:
1. **The EBOPs scale changes. Re-derive the beta/BetaPID setpoints before any sweep.**
   - Today 99.99% of the high-spec penalty is the 4 decoder cross-attention softmax exp tables (`port_log/critique.md`). That
     comes from QSoftmax's `-1e9` mask fill. The attend softmax takes no mask, so this term disappears.
   - Per module at the `q_ca` shape (QML, batch 1, initial bitwidths): attend 1.050e10, 1.83x the unmasked quadratic attention
     (5.74e9). Core only: 3.6x. Masked quadratic attention: 1.27e14, almost all of it from the fill.
   - The trial's "quadratic 2.5e14 vs masked-linformer 9.3e9" is the same fill effect.
2. **Renorm precision under trained E bitwidths.** The LUT is sized for a fixed `f_E`, but HGQ2 trains E's bitwidths. Cap E's
   fractional bits at 7 (the table limit), size the LUT for the cap, and watch E's `f` with the bitwidth monitor.
3. **uint32 EBOPs readout.** One E projection already reads 2.36e9, 55% of the uint32 limit. Use `sum(losses)/beta` (section 3.6
   of the plan).
4. **WRAP datalane overflow** makes production-spec forward parity chaotic after the first mask feedback (S3 deviation 8).
   Production golden rows are gated at layer 0 only.
5. **Export** needs a rewrite in S7: `where` becomes a {0,1} multiply; `M·W` becomes a `QEinsumDense` with W as its kernel.
   The q·d^-0.5 scaling is an exact shift only because head_dim is 16. hls4ml was checked by code generation only, never with C simulation.
6. **Sort ties.** `keras.ops.argsort` and torch's unstable argsort can order equal valid phi values differently. That would change
   which E row a token uses. It is rare; measure the duplicate rate on real CLIC events in S6.
7. **lap1015 against scipy:** same optimum, ties may differ. Gates compare matched totals, not indices.
8. **Between R0 and R4 the generated Polaris config is a mixed model.** Do not launch Polaris jobs from keras-jax in that window.
   Running jobs use the `keras-hgq2` clone and are not affected.

## 8. Critic review (2026-10-15)

Checked against org/main `ac0da8f`, masked-linformer `dce2e15`, HGQ2 0.1.9, the trial merge and the prototypes. Log:
`$S/review_rt/LOG.md`. Reruns that confirm the plan: `probe_chain.py` (8 blockers, k=256), `probe_losses.py` (all numbers in
sections 1.3 and 3.3), `probe_pr8_hybrid.py` (≤ 7.2e-7; controls 0.407 and 0.135), the scipy non-finite behaviour, the B2-c
totals (hand calculation), `git merge-tree` (3 conflicts), the 41-failure breakdown in `final.native.log`.

Corrected in this file:
- R0 gate (c) and the R5 gate: `tests/matching` fails locally on lap1015 alone (106 / 105 failures), not 0.
- Section 5.5: torch and keras argsort agree only on the valid prefix, because torch breaks `+inf` ties differently.
- Section 3.2: the empty-selection deviation remains for the regression L1 and `kl_div` losses.
- Section 5.1: the `q_sa` shape exception, R2 ownership of `build_attention`, and the gate's dependency on R2.
- Section 1.2: the batch-of-one all-valid path (`maskformer.py:112-113`), which turns the renorm off.
- Discrimination notes: B1-e, FastLayerNorm parity, the unsort round trip, exact mask booleans.
- References: `WORKLOG.hgq.md:156`, `KERAS_JAX_PLAN.md:468-471`, `tests/keras_native/porting.py`, the pin rationale.

Not verified here (they rest on the cited investigation logs): hls4ml acceptance of an elementwise same-shape `QEinsum` for
SwiGLU; the 1e6-entry hls4ml table cap; `pixi lock` solving linux platforms on macOS; the LOC estimates; that the hybrid
`KerasAttention` can hold the native core under the torch backend (gate H0 exists for this).
