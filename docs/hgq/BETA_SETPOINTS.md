# Beta / BetaPID setpoints for the org/main HGQ2 model (PROVISIONAL)

Status: derived on 2026-10-16 in stage S5 ([port_log/S5.md](port_log/S5.md)), CPU only, CLIC dummy events. **Provisional
until a Polaris beta scan** on real events confirms them. Used as the defaults of `polaris/03_make_config.py`; pinned by
`tests/polaris/test_beta_setpoints.py`, with the EBOPs input re-measured by `tests/keras/test_quant_losses_dedupe.py`.

| Setting (per EBOPs-active step) | Old default | New default | Rule |
|---|---|---|---|
| `HGQ_TARGET_EBOPS` | 1.0e15 | **1.5e11** | 40% cut of the model's EBOPs at init (old: 1.0e15 / 1.6755e15 = 0.597) |
| `HGQ_INIT_BETA` | 2.6e-10 | **8.7e-11** | keep the old ratio of penalty gradient to task gradient on the bitwidths |
| `HGQ_MAX_BETA` | 1.0e-7 | **3.3e-8** | the old 2.585 decades of headroom above `INIT_BETA` |
| `HGQ_MIN_BETA` (new) | 0 | **2.3e-13** | the same 2.585 decades below `INIT_BETA`; must be > 0 for the JAX EBOPs read-out |
| `HGQ_PID_I` | 0.5 | **0.5** | about one decade of beta per ten epochs at the initial error (0.224 old and new) |
| `HGQ_BETA_START` (new, schedule mode) | 0 | **1e-3 x `HGQ_BETA_END`** | > 0, for the JAX read-out; the schedule mode itself (`HGQ_BETA_END` 4e-15) is NOT re-derived |

The generator still multiplies every beta by `HGQ_EBOPS_EVERY` (8) for PeriodicEBOPs; the table gives the per-active-step
values. If the native JAX driver drops PeriodicEBOPs (S5.md recommends it if its compiled share stays below 5% on the
GPU), generate its config with `HGQ_EBOPS_EVERY=1`.

## Why the old values are stale

- The old 1.6755e15 EBOPs were 99.98% decoder cross-attention softmax tables (QSoftmax's -1e9 mask fill). The org model's
  masked-Linformer attend softmax takes no mask, so that term is gone (RETARGET_ORG_MAIN.md section 7, risk 1).
- The old callbacks wrote beta only to the top-level keras layers (R4b deviation 3), and the old `quant_losses()` counted
  the task heads twice. The penalty the old runs differentiated was beta x 2.880e11 (measured below), not beta x 1.6755e15.
- `INIT_BETA` 2.6e-10 came from a beta scan at the old learning rate (1.25e-5 instead of 5e-5, plan R15).

## Measurements (all on the same 3 batches x 2 CLIC dummy events, padded-node targets zeroed)

Scripts: [S5_make_batches.py](port_log/S5_make_batches.py), [S5_grad_ratio.py](port_log/S5_grad_ratio.py) (hybrid
torch env, each tree's own `MPflowHGQ` built by jsonargparse from its own generated config, beta written by that tree's
own `BetaScheduler`). Results: [S5_evidence/ratio_old.out](port_log/S5_evidence/ratio_old.out),
[ratio_new.out](port_log/S5_evidence/ratio_new.out). Old tree: `git archive 37ff127` (pre-retarget keras-jax, Helen's
Linformer k = 256, its own `03_make_config.py`). New tree: this branch.

| Quantity | Old model (37ff127) | New model (org/main) |
|---|---|---|
| Total EBOPs at init, every leaf, exact int64 | 1.6756e15 (Polaris read 1.6755e15) | 2.509e11, 2.512e11, 2.512e11 (mean **2.511e11**; G2 real events 2.513e11) |
| EBOPs that beta multiplies (`quant_losses` at beta 1 minus at beta 0) | 2.880e11 | 2.509e11 - 2.512e11 (= the total) |
| Matched task loss | 155.8, 126.4, 135.0 | 58.5, 55.0, 57.4 |
| `|grad_q task|` (bitwidth variables: 684 old, 982 new) | 6.35, 4.59, 5.33 | 0.606, 0.493, 0.603 |
| `|grad_q penalty|` per unit beta | 8.408e7 | 2.652e7, 2.661e7, 2.661e7 |
| s_q = penalty / task | 1.325e7, 1.830e7, 1.576e7 | 4.376e7, 5.402e7, 4.415e7 |
| s_q(old) / s_q(new) per batch | | **0.303, 0.339, 0.357** (mean 0.333) |
| same with all trainable variables | | 0.312, 0.348, 0.366 (mean 0.342) |

`INIT_BETA` = 2.6e-10 x 0.333 = 8.65e-11, rounded to 8.7e-11 (the batches span 7.9e-11 to 9.3e-11; the all-variable norm
gives 8.9e-11). With PeriodicEBOPs x 8 the generated `init_beta` is 6.96e-10.

## Method and its limits

- The gradient-ratio rule keeps the relative pull of the EBOPs penalty on the bitwidths that the old sweep judged as
  "moves /f without destabilising the task loss". It does not carry over the LR change (the old scan ran at 1/4 of the
  current LR, so the same beta now moves bits about 4x faster per step), nor the Adam per-parameter normalisation.
- Dummy events, initial weights, two events per batch. Real events change the EBOPs at init by 0.2% (G2: 2.513e11) and
  the task gradient by an unmeasured amount.
- The target asks a question we can answer in one run (does closed-loop pressure move the cost); it is not the hardware
  budget (an XCVU13P holds about 2.15e6 EBOPs).

## Before a sweep (Polaris)

1. Beta scan around 8.7e-11 (for example 2.2e-11, 8.7e-11, 3.5e-10 per active step) at the production LR on the
   generated config; watch `val/bits_f_mean`, `val/ebops` and `val/loss` for one epoch.
2. Re-measure EBOPs at init on real events (the S6 smoke run logs it) and re-apply the 40% rule if it moved.
3. Then BetaPID with the confirmed `INIT_BETA`; check that `train/pid_saturated` stays 0 for the first epochs.

## Betas from `polaris/09_measure_beta.py` before and after S5

`09_measure_beta.py` calibrates beta against the slope `E_eff` of `quant_losses()` (`evaluate.effective_ebops`). Before
the S5 `quant_losses()` fix (`7f3747c`) that slope was the leaf total plus the task heads, which the walk counted twice:
total x 1.108 on the org model (the heads are 10.8% of its EBOPs, R4b). After the fix it is the leaf total. So a beta
that `09_measure_beta.py` computes for a given penalty fraction is about 11% larger after S5 than before; do not mix
betas from the two versions. The setpoints above use the leaf total (`penalty / beta`), the post-fix definition.
