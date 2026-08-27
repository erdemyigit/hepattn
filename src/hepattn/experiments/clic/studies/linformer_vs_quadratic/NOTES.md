# Linformer vs quadratic attention — float comparison for FastML

**Owner:** Erdem Ertorer · **Started:** 2026-08-26 · **Branch:** `linformer-float`
**Goal:** a jet-E median/IQR curve for Linformer, overlaid on the reference MaskFormer
curves, on identical code and identical evaluation.

## Base selection (done)

| fork | vs `lgray/main` (1df05cc) | verdict |
|---|---|---|
| `asgrover-cmu/main` (Akum) | **5 behind**, 0 ahead, last push 2026-05-14 | wrong base — stale, no changes of its own |
| `mmfsz/main` (Maria) @ 6ee4c60 | **5 ahead** | **chosen** — carries `configs/eval.yaml`, RNTuple-capable `performance/reader.py`, and `studies/glow_jet_iqr/` |
| `helenllii/linformer-hardmask` @ e207a434 | 9 ahead / 5 behind | source of the Linformer |

Linformer exists **only** on Helen's branch and our `keras-hgq2`. Our torch
`linformer.py` is semantically identical to hers (token-level diff: one unused import,
`0.` vs `0.0`).

Ported: `models/linformer.py`, `configs/linformer.yaml` (verbatim), `models/attention.py`
(3-way merge — Helen branched before lgray/main, so her edits replay over Maria's tree;
**both** her linformer branch and Maria's bf16 cast survive), `models/__init__.py` export.
`encoder.py` skipped (blank line only).

## Measured on this tree (both configs built through the real CLI)

```
configs/base.yaml       10,126,115 params  (10.13M)   attn: 17 torch + 7 flash-varlen
configs/linformer.yaml  12,471,587 params  (12.47M)   attn: 24 linformer
```

The baseline reproducing Maria's documented **10.1M exactly** is the evidence the merge
left the quadratic path untouched. `tests/models/` failure set is byte-identical before
and after the port (6 pre-existing macOS inductor / flash-attn failures).

## ⚠️ Two findings that constrain the physics claim

### 1. Masked Linformer CANNOT compress: `k >= sequence length` is required

The attention mask is built in the **projected** space (last dim `k`) but filled from an
original-sequence mask (last dim `kv_len`), so `k < kv_len` raises. Measured boundary at
kv_len=168:

| k | 64 | 128 | 167 | 168 | 200 | 256 |
|---|---|---|---|---|---|---|
| masked | CRASH | CRASH | CRASH | OK | OK | OK |
| unmasked | OK | OK | OK | OK | OK | OK |

The CLIC config leaves `linformer_proj_dim` unset → **k=256 default, against 168 hits**.
So the "compression" matrix is an **expansion**, and it is *forced by the mask handling*,
not chosen. The decoder always masks (MaskFormer mask-attention, `decoder.py:273`), so the
decoder can never compress as implemented. The encoder only masks if a `mask_mod`/window
is configured — an unmasked encoder *could* take k < n.

**Consequence: this configuration is not an efficiency win.** It is +23% parameters and a
larger attention inner dimension. Any FastML claim must be about *accuracy at fixed
mechanism*, or the mask handling must be fixed first.

### 1b. Linformer is parameter-ADDITIVE, and at n=168 it cannot save compute either

Measured parameter buckets (both configs built through the real CLI):

| bucket | quadratic | Linformer | delta |
|---|---|---|---|
| attn Q/K/V/O projections | 4,737,024 | 4,723,200 | -13,824 (bias only) |
| attn E/F projection | 0 | 2,359,296 | **+2,359,296** |
| MLP / Dense | 5,291,219 | 5,291,219 | **0** |
| norms | 36,864 | 36,864 | 0 |
| other | 61,008 | 61,008 | 0 |
| total | 10,126,115 | 12,471,587 | +2,345,472 |

W^Q/W^K/W^V/W^O are `dim x dim` — their size does not depend on sequence length, so
compressing n cannot shrink them. Linformer shrinks the n x n attention matrix, which is an
*activation*, not a parameter. E/F are extra weights on top:

    params(Linformer) = params(standard) + 2 * seq_len * k   per attention module

Measured: 2*256*256 = 131,072 x **18** modules (6 encoder + 12 decoder) = 2,359,296 = the
entire delta. **It is not the MLPs** (byte-identical) and **not a small head_dim** (the
Q/K/V/O block is unchanged).

FLOPs at n=168, d=256, B=8 (analytic and measured agree exactly):

```
attention core (n^2) : 0.231 GFLOP = 4*B*n^2*d    <- only 25%
Q/K/V/O proj (d^2)   : 0.705 GFLOP = 8*B*n*d^2    <- 75%
```

| k | GFLOP | vs quadratic |
|---|---|---|
| 32 | 0.793 | 0.85x (best case) |
| 84 | 0.936 | **1.00x — break-even at k = n/2** |
| 168 | 1.167 | 1.25x (minimum the masked path allows) |
| 256 | 1.409 | **1.51x (our config)** |

Core fraction is `n/(2d+n)`; with d=256 you need **n >~ 512** before the quadratic term is
even half the cost. **Break-even k=84 is below the minimum k=168 the mask requires, so on
this model Linformer can never be cheaper than quadratic attention.**

> Measurement trap: `torch.utils.flop_counter.FlopCounterMode` does NOT count SDPA's
> attention core, but DOES count Linformer's explicit einsums. Comparing them naively
> understates quadratic by 25%. Force the core to explicit matmuls before comparing.

**Framing consequence:** an efficiency claim is unsupported at CLIC's n=168. The honest
claim is accuracy at fixed mechanism; the efficiency argument belongs at large n
(HL-LHC-scale hit multiplicities), which is where Linformer was designed to pay off.

### 2. Linformer's 12.47M is dangerously close to the paper's 12.1M

Akum's reference plot has "Paper MaskFormer 12.1M" and "Maskformer 10.1M". Our Linformer
lands at 12.47M for an unrelated reason (the E matrices). **Label it explicitly** or the
curves will be misread as a parameter-matched comparison.

## Comparability requirements (from Maria's `glow_jet_iqr/`)

- Overlay tool: `studies/glow_jet_iqr/00_cross_run/compare_runs_iqr.py`
- Convention in that tool: `network_type: mpflow`, **`ind_threshold: 0.65`**,
  truth `test_clic_common_raw.root`, jets `dr_cut=0.1, leading_n_jets=2, pt_min=10`,
  IQR = p75−p25 of `e_rel` in 20 GeV bins of `ref_e`.
- Eval must use `configs/eval.yaml` (`precision: 32-true`, `matmul_precision: highest`,
  `is_inference: true`).

> ### ⚠️ `eval.yaml` will silently destroy a Linformer evaluation
> It sets `model.model.encoder.attn_type: torch`, rebuilding the **encoder** as standard
> attention. Measured: the two modules share **zero** state-dict keys —
> linformer has `attn.{proj_k,proj_v,to_q,to_k,to_v,to_out}`, torch has
> `{in_proj_weight,in_proj_bias,out_proj.*}`. Loading a Linformer checkpoint gives
> `missing=4 unexpected=7`: a strict load raises, a non-strict load leaves the encoder
> **randomly initialised** and the run produces plausible-looking garbage.
> Override it back with `--model.model.encoder.attn_type=linformer` (the decoder is not
> touched by eval.yaml and stays linformer either way).

## The run: configs/linformer_polaris.yaml

Generated from v6 `base.yaml` (the config behind the 10.1M reference curves) by
`mkcfg` — regenerate rather than hand-edit, so provenance stays checkable.

| change | v6 | ours | why |
|---|---|---|---|
| encoder `attn_type` | flash-varlen | **linformer** | the thing under test |
| decoder `attn_type` | (absent → torch) | **linformer** | the thing under test |
| `optimizer` | Lion | **AdamW** | Lion reportedly does not converge with Linformer (Erdem, measured) |
| `lrs_config.max` | 8e-5 | **1e-4** | Lion is sign-based; its LR does not transfer. 1e-4/0.03 is THIS repo's last validated AdamW recipe for the CLIC MaskFormer (`base.yaml` @ 22bb033, 2025-07-28, immediately before the Lion switch) |
| `weight_decay` | 1e-4 | **0.03** | pairs with the AdamW LR above; Lion's 1e-4 would be near-zero regularisation for AdamW |
| `batch_size` | 2048 | **256** | v6's 2048/GPU was tuned for a 192 GB B200. Maria's 24 GB L4 ran 256/GPU; 256 x 4 A100-40GB = global 1024, same family as the PLOTTED baseline (global 768) |
| `num_workers` | 16 | 8 | 32 cores / 4 ranks |
| paths | cmsuf | `/eagle/<project>/clic` | Polaris; `test_path` -> val file (the infer file has -9999 sentinels) |

Unchanged: architecture, 200 epochs, bf16-mixed, devices 4, scaling dict, OneCycle
schedule shape. Verified through the real CLI: 12,471,587 params, all 24 attention
modules linformer, AdamW instantiated with wd 0.03 / max_lr 1e-4 and **all** trainable
params in the optimizer.

#### Polaris bring-up: three failures, all environmental

1. **`.venv/bin/activate` not found.** venv lives under `WORK_ROOT` on /eagle; the branch
   is checked out in $HOME. hepattn is NOT installed into the venv, so PYTHONPATH decides
   which clone runs -- and the /eagle clone is on keras-hgq2. A preflight now asserts
   `hepattn.__file__` is inside `$REPO_DIR/src`.
2. **`nvc-Error: Unknown switch -Wno-psabi`.** Polaris puts NVHPC's `nvc` on PATH; Triton
   builds its CUDA helper with `$CC` and passes GCC-only flags. Fixed by exporting
   `CC=gcc` (7.5.0 on Polaris works). Not avoidable by config -- loss.py:352-359 wraps
   every cost function in torch.compile at import.
3. **`RuntimeError: shape '[1344, 256]' is invalid for input of size 64`** in the compiled
   BACKWARD. torch.compile (via the `Compile` callback, `dynamic=True`) generates broken
   inductor code for the linformer path on torch 2.10: the kernel allocates a 64-element
   workspace then views it as `[64 + 8*s27, 256]`. Forward and the validation sanity check
   pass; it dies on the first backward. **Dropped the `Compile` callback.** This is a
   deviation from the v6 baseline, which trained WITH it -- speed only, not numerics, but
   worth stating. loss.py's compiled cost functions are unaffected and still work.

Also a smoke-harness artifact worth remembering: OneCycleLR's first phase spans
`pct_start*total_steps - 1` steps, so a short `limit_train_batches` (<40 at pct_start
0.05) makes `get_lr()` divide by zero. Not a config fault.

## Confounds to state on any plot (ranked)

1. **Optimizer**: ours AdamW, the v6 baseline Lion. Unavoidable if Lion truly fails with
   Linformer, but it means the curve is not a pure attention ablation. Closing it needs an
   AdamW *quadratic* baseline — a second 200-epoch run.
2. **`value_residual` AND `qkv_norm` are both silently inactive under Linformer.** The
   linformer branch skips `_prepare_qkv`, where both are applied. v6 sets
   `value_residual: true` on the encoder, and `hybrid_norm: true` forces qkv_norm on for
   every attention module (`qkv_norm = qkv_norm or hybrid_norm`, norm.py). Measured on the
   real model: **48,208 parameters receive no gradient** — `value_residual_mix` on 5
   encoder layers (layer 0 is `is_first_layer`) plus `q/k/v_norm` on all 18 modules.

   This is not cosmetic: DDP aborts with *"parameters that were not used in producing the
   loss"*. Worked around with `strategy: ddp_find_unused_parameters_true`. It cannot be
   fixed by disabling `hybrid_norm`, which also drives `attn_norm`/`dense_post_norm` and
   would change the real architecture. **So the Linformer model is missing qkv-norm and
   the value residual relative to the quadratic baseline** — the largest architectural
   confound in this comparison. Proper fix: apply both inside `LinformerAttention`.
3. **Parameters**: 12.47M vs 10.13M (+23%), entirely the E/F matrices — see §1b. Do not
   present this as parameter-matched, and do not let 12.47M be confused with the paper's
   12.1M.

## Corrections to earlier assumptions

- The neutral-pT `predictionwriter.py` no-op **is a real bug but is immaterial here**:
  Maria measured max |ΔIQR| = 0.0012 and <1% shift in neutral pT, because the model
  regresses `e` and `pt` consistently (massless relation). It does **not** explain the
  neutral gap. Do not re-open (`glow_jet_iqr/00_cross_run/jet_iqr_discrepancy.md` §4.3).
- The paper-vs-HEAD IQR gap is **temporal drift, not fork drift**: HEAD is 100 commits
  past the `clic-paper` tag (`fb90390`). Leading suspect is the `Dense` refactor (#212)
  doubling the incidence-regression MLP width. So our Linformer curve is comparable to
  the **10.1M HEAD** baseline, not to the paper-tag 12.1M curve.

## Open — needs Akum

Which repo/commit and which branch (`mpflow_proxy` vs regression) produced his latest
plot. If it predates Maria's eval tooling the baselines need regenerating.
