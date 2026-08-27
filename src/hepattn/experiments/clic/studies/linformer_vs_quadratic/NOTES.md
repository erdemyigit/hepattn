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
