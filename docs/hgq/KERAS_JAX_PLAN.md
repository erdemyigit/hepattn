# Plan: move HGQ2 QAT training to pure Keras on JAX

Status: G1 GO (its trainer gap still OPEN until G3); S1-S4 merged into `keras-jax` on 2026-10-15; retargeted to the org/main float Linformer model (`ac0da8f`)
on 2026-10-16; S5, S6a, S6b done on 2026-10-20 (Polaris: real-data smoke passed, job 7763621, and G3 passed, job 7763879: recorded in the main worktree's `WORKLOG.hgq.md`, not on this branch; [port_log/README.md](port_log/README.md) points there); S7 done on 2026-10-20 and merged into `keras-jax` on 2026-10-21 (`f4aab4a`) ([RETARGET_ORG_MAIN.md](RETARGET_ORG_MAIN.md), stages R0-R5; see "Status" below). Originally a proposal that waited for G1.
Date: 2026-10-14. Branch: `keras-jax` (pushed up to `f30d650`, after S5/S6; the S7 merge is not pushed), worktree `hepattn-keras-jax`.
Scope: only the CLIC HGQ2 quantization-aware training, evaluation and export path. The other
experiments (`atlas_muon`, `trackml`, `tide`, ...) and the float torch model stay on torch and Lightning.

All paths are relative to the `hepattn-keras-jax` worktree unless they start with `SP/`, which means
`.venv-jax/lib/python3.12/site-packages/` (keras 3.15.0, hgq2 0.1.9, jax 0.11.2, scipy 1.18.1).
Commits are on `keras-jax` (`git log main..keras-jax`).

## Status update (2026-10-21, S7 merged)

**Done:** S1-S7. `s7/export` (`2b15167`, after two review rounds) is merged into `keras-jax` as `f4aab4a`; the merged
tree equals `2b15167`. Suites on the merged tree: [port_log/S7-merge.md](port_log/S7-merge.md). G3 and the real-data
smoke passed on Polaris (jobs 7763879 and 7763621; numbers in the main worktree's `WORKLOG.hgq.md`).

**Open:**
- **Lindsey Gray's PR #9 float checkpoint:** not received. When it arrives, convert it with `convert_to_native`
  (`--target-config` = the QAT config, read `max_abs_diff` and `renorm_lane` in the `--report`), then warm-start QAT
  with `main_jax fit --init-from` ([KERAS_NATIVE.md](KERAS_NATIVE.md)).
- **G4 (S8):** JAX vs hybrid from the same converted weights and data order (section 4, "S8 / G4").
- **Beta scan:** the EBOPs setpoints are PROVISIONAL ([BETA_SETPOINTS.md](BETA_SETPOINTS.md),
  [port_log/S56-critique.md](port_log/S56-critique.md)); the Polaris beta scan is still due.
- **Loop overhead:** the Polaris smoke spent about 0.3 s per micro-batch outside the 0.16 s train step (data, logging,
  validation, checkpoints; `WORKLOG.hgq.md`). Being profiled on `prof/loop` (worktree `hepattn-prof`, not merged).
- Carried: csim by pytest on Linux (the falcon login node kills the process; `S7_csim.py` covers it), the
  tolerance-0 golden rows on Linux/GPU, the G3 diagnostics job 7764401 (G1 gap, TF32).

## Status update (2026-10-20, after S7)

**Done:** S7 ([port_log/S7.md](port_log/S7.md)), on branch `s7/export` (worktree `hepattn-s7`, from `f30d650`):
- converter `python -m hepattn.experiments.clic.convert_to_native` (outside `keras_native`: imports torch) from a
  Lightning `.ckpt` + its saved `config.yaml` of (a) the org float model and (b) the hybrid HGQ2 model, into the native
  model and S6a's checkpoint format; `main_jax fit --init-from` warm-starts from it ([KERAS_NATIVE.md](KERAS_NATIVE.md));
- export under JAX (`keras_native/export.py`): functional assembly from the trained layers (SwiGLU included), export
  forms for the renorm `kept` leaf (exact) and inline silu / sigmoid (approximate); every org-model leaf kind generates
  hls4ml code; C simulation bit-exact for 22 of 22 models on falcon ([FPGA_STATUS.md](FPGA_STATUS.md));
- hybrid CLIs `port_to_keras` / `evaluate_hgq` run on the org model (R4b / S56 open issues closed).
Section 3.9's "existing Polaris `.ckpt` files are loaded through the S7 converter" holds for hybrid checkpoints
written by `MPflowHGQ` (they hold every keras weight) and org float checkpoints; the optimizer state is not converted.
**Not run:** G3 (queued on Polaris); S8.

## Status update (2026-10-20, after S5, S6a and S6b)

**Done:** S5 train step / optimizer / EBOPs / beta ([port_log/S5.md](port_log/S5.md)), S6a config, checkpoints, data,
metrics, prediction writer ([port_log/S6a.md](port_log/S6a.md)), their merge ([port_log/S56-merge.md](port_log/S56-merge.md)),
and S6b, the single-device training driver ([port_log/S6b.md](port_log/S6b.md)):
`python -m hepattn.experiments.clic.main_jax fit|test` ([KERAS_NATIVE.md](KERAS_NATIVE.md), "Training"). On CPU it
trains the generated Polaris config end to end (at test sizes), resumes bit for bit from every checkpoint position
tested (inside an epoch, at an epoch end, after the PID's epoch step), compiles 2 train + 1 val programs per run
(plus a loss-only `tail` program for the short last val batch and a bit-statistics program:
`test_trace_counts_over_the_run`), and logs the hybrid's names (`val/loss` = task loss, measured on the hybrid: S5's "task + quant_losses" was wrong). G2 was
measured on Polaris (native fwd + bwd 165.2 ms vs hybrid ~1615 ms at b16, fp32; at JAX's default matmul precision, not
the production "highest"). **Not run:** **G3** (`polaris/17_g3_step.pbs`: >= 1.8x the hybrid in-run step at b32,
matching <= 15%; also the G1 gap, TF32 and the EBOPs share at b32) and the real-data smoke
(`polaris/16_jax_smoke.pbs`). **G1 is not resolved** (the Keras-trainer gap is measured only by G3). Next: the two
Polaris jobs (G3 in two submissions: the gate set first, then `G3_JOBS=jax-g1:32,jax-precision:32,hybrid-tf32:16`;
both arms at full fp32 matmuls, the hybrid with `--matmul_precision highest`; S6b.md "Fix round 2"), then S7
(converter, export) and S8 (G4).

## Status update (2026-10-16, after the retarget R0-R5)

Both HGQ2 implementations now target org/main `ac0da8f` (`base.yaml` + `linformer.yaml`): masked Linformer
(`masked_linformer` `dce2e15`, k = 64) in every attention, phi sorting, the PR #8 layers (FastLayerNorm, legacy dense-norm
placement, `ca_sa_dense`, `self_attn_norm_kv`), SwiGLU everywhere (quantized as two dense halves and an elementwise
`QEinsum`), bugs 1 (`a4263c1`, per-event mask losses) and 2 (PR #3, class cost on the logits), and the `nan_to_num` /
`lap1015_late` matcher (solved with scipy). The hybrid reuses the native quantized masked-Linformer core (`keras_native/
masked_linformer.py`, gate H0). Helen's Linformer (`models/linformer.py`, `keras/linformer.py`, `keras_native/linformer.py`)
is deleted. Logs: [port_log/R0.md](port_log/R0.md) .. [port_log/R5.md](port_log/R5.md), [port_log/R-merge.md](port_log/R-merge.md).
Polaris G2 (`15_g2_forward.pbs`) now builds the org model from the generated config in both arms (R5).
Rows of the table below that mention Helen's Linformer, `mask_bce` `inf` or the cross-event mask losses describe the
S1-S4 state; the retarget replaced them (R1, R2, R5). The plan's model description in section 3.2 is updated in place.

## Status (2026-10-15, after the S1-S4 merge)

Logs: `docs/hgq/port_log/` (one file per stage, plus `merge.md`). Lanes `kn/lane-a` (`hepattn-kn-a`) and
`kn/lane-b` (`hepattn-kn-b`) are merged into `keras-jax` and kept. `origin/keras-jax` is at `197ede2` (pre-S1); nothing after it is pushed.

**Stages done:** S0 (G1 = GO), S1, S2, S3 (except G2), S4. **Not started:** S5, S6, S7, S8.

| Gate | Result | Evidence |
|---|---|---|
| G1 (Polaris, job 7746577, A100-40GB, fp32, spike graph) | GO (user) | `hepattn/WORKLOG.hgq.md`: torch loop 617 ms vs jax jit 69 ms at b16 (8.9x); torch OOM at b32. The "fixed share <= 25%" criterion failed as written (53% at b32) |
| S1 import hygiene; hybrid suite through the factory shim | pass | [S1.md](port_log/S1.md) Gates |
| S2 float parity at PARITY.md tolerances | pass | `test_float_parity.py`; [S2.md](port_log/S2.md) Gates (max 2.38e-6 on `encoder.clic_shaped`, through the rtol term) |
| S2 fully-masked-query key independence + positive and leaky-softmax controls | pass | `test_masked_query.py`; S2.md |
| S2 quantized golden parity at 2x measured; paired delta | pass | `test_quantized_golden.py`; 7/9 cases >= 10x, the 2 single-Linformer cases match the hybrid's own 5.3x/8.5x (S2.md deviation 6). Full model: 0.30144 vs 31.0916, 103x ([S3.md](port_log/S3.md)) |
| S2 per-layer JIT (1 compile, nonzero grads, EBOPs > 0) | pass | `test_jit.py`; S2.md |
| S3 `maskformer.e2e` float at 5e-5/1e-4 | pass | `test_maskformer.py`; S3.md |
| S3 per-layer mask booleans exactly equal | pass for float; quantized rows held at 2x measured flips | S3.md deviation 11 |
| S3 all-False-row restore + restore-off control; 1 compile over 3 batches; unsupported configs raise | pass | `test_maskformer.py`, `test_mf_jit.py`; S3.md |
| S3 quantized MaskFormer vs hybrid golden files | pass, prod (WRAP) rows gated at layer 0 only | `test_mf_golden.py`; S3.md deviations 8, 12 |
| **G2** (Polaris) jitted native fwd+bwd <= 0.5x hybrid bare-loop step | **not run** | `polaris/15_g2_forward.pbs`, validated on CPU only. Decided at b16 fp32, not the plan's b32 (S3.md "G2 decision rule") |
| S4 loss rows 1e-4/1e-3 incl. `mask_bce` inf (since R1: finite, per-event, bug 1) | pass | `test_objective_e2e.py`; [S4.md](port_log/S4.md) Gates (max abs 1.9e-6) |
| S4 task costs to 1e-5; matched total over 100 batches (rel 1e-6) | pass | `test_objective_parity.py`, `test_matching.py` (rel 0.0; identity control worse 100/100) |
| S4 `pure_callback` gradients = fixed-index reference, 1 callback per step | pass | `test_matching.py` (grad diff 0.0; identity control 0.649) |
| S4 NaN targets, weighted CE, permutation | pass | S4.md Gates. The NaN control needs a squared residual (deviation 2) |
| Merge: lane A model outputs train through lane B's matched loss | pass | `test_lane_integration.py`; [merge.md](port_log/merge.md) (3 mutations caught) |

Test runs on the merged tree: [merge.md](port_log/merge.md) (native 350 passed / 1 skipped incl. the 2 merge tests;
hybrid `tests/keras` 104 passed / 8 skipped, same as S1; `tests/polaris` 3 passed). After the critique
([critique.md](port_log/critique.md), 3 tests added, 13 new mutations): native 353 passed / 1 skipped.

**Deviations from this plan** (details in the stage logs):
- G2 is decided at batch 16 fp32, because torch fp32 OOMed at b32 in G1. The G2 torch arms have no optimizer, so b32
  may fit; a reviewer suggests also trying it (S3 review).
- Torch `mask_bce` / `mask_dice` average across events at B > 1 (probably an upstream bug). S4 keeps the torch
  semantics. **This needs a user decision** (S4.md deviation 1). *Resolved upstream:* org `a4263c1` (bug 1)
  normalises per event; torch, hybrid and native follow it since R1 ([R1.md](port_log/R1.md)).
- The plan's claim that `where(mask, |out - nan|, 0)` gives NaN gradients is false in JAX for l1 (S4.md deviation 2).
- The hybrid's `quant_losses()` counts the task heads twice; S5 must use `stateless_call(return_losses=True)`
  (S3.md deviation 9). Eager `model.losses` over-counts too.
- Exact mask equality only in float; prod WRAP rows gated before the first mask feedback (S3.md deviations 8, 11).
- Golden rows measured at 0 have tolerance 0; re-measure on the first Linux/GPU run (S2.md deviation 7, S3.md).
- The single Linformer layer separates high/low only 5.3x/8.5x in the hybrid itself (S2.md deviation 6). *Gone with
  Helen's layer:* the masked-Linformer golden cases separate >= 121x ([R5.md](port_log/R5.md)).

**What remains:**
- Polaris: G2 (`bash polaris/submit.sh 15_g2_forward.pbs`), then G3 (S5) and G4 (S8).
- S5 train step on top of `test_lane_integration.py`: optimizer, schedule, clipping, accumulation (count host
  calls in the accumulation path), EBOPs/beta, PeriodicEBOPs, BetaPID; pass the YAML matcher `n_jobs` (16).
  The constant 3.2e12 HGQ2 penalty at the `high` test spec (merge.md) is explained in
  [critique.md](port_log/critique.md): 99.99% of it is the 4 decoder cross-attention softmax exp tables (8.3e11 and
  7.8e11 each, table f0 = 16), which do not depend on the data. It is not uint32 saturation (the loss is float).
  `MaskFormer` already raises on `sorter` and on `input_sort_field` (both rows in `test_unsupported_configs_raise`).
- S6 driver, data, checkpoints, metrics, eval (check real CLIC batches: no target 1 on padded nodes, prefix-contiguous
  `particle_valid`); S7 converter + export (key quantizer variables by leaf name and position, not `var.path`);
  S8 Polaris validation.
- Minor review items not fixed in the lanes: `LinformerAttention` accepts any keyword (`**_ignored`, as this plan's
  3.2 row asks; *gone*: the class is deleted and `build_attention` raises on unknown keys since R2); the G2 hybrid-07 arm calls `quant_losses()` inside the timed step (as 07 does); dense-attention
  quantized parity is loose (S3.md deviation 12). Fixed in the critique ([critique.md](port_log/critique.md)): the
  JIT gate now rejects NaN gradients, a test pins the `>=` mask threshold, G2 records `hepattn.__file__`, and the
  `MaskFormer` docstring no longer calls eager `model.losses` equivalent.

## 0. Decision summary

1. **Keras cannot train "with no backend".** Keras 3 trains only on JAX, TensorFlow or PyTorch. The
   NumPy backend raises `NotImplementedError("fit not implemented for NumPy backend.")`
   (`SP/keras/src/backend/numpy/trainer.py:170`), and OpenVINO refuses `fit` (`SP/keras/src/backend/openvino/trainer.py:235`).
2. **Lightning is not the main cost.** Lightning can add at most about 1.3x. The step is dominated by a
   fixed per-op cost of the eager torch backend, and most of those ops come from HGQ2 quantizer arithmetic.
   A local spike without Lightning shows the same cost (section 1).
3. **A compiled step is the lever.** On a local CPU, the same HGQ2 graph runs **8.5x faster at batch 4 and
   1.74x faster at batch 32** under JAX `jit` than in a plain torch loop. Without `jit`, JAX is 34x
   (batch 4) to 9.0x (batch 32) *slower* than with it, so the gain comes from compiling, not from JAX itself.
   Both arms of the spike ran fp32; the A100 production baselines (1140 ms bare loop, 1.008 s in-run) ran
   bf16 autocast (`polaris/07_profile.py:134`, `polaris/03_make_config.py:46`).
4. **Recommendation:** run `polaris/14_backend_spike.pbs` on one A100 first (gate G1, section 5.1).
   - **GO** (JAX at least 2x faster at batch 32, all checks pass): port the CLIC QAT path to a pure
     `keras.ops` model on JAX. Train it with a custom stateless `jax.jit` step. Run Hungarian matching
     on the host through `jax.pure_callback`. At least 4,100 LOC in 8 stages, each with a parity gate
     (a lower bound; see section 4).
   - **NO-GO:** do not port. Use Helen's `hgq_train_compile` option (commit `cdc2f3b` on
     `helen/perf/hgq2-train-compile`) on the existing hybrid. It measured 768.6 -> 465.6 ms/step (1.65x)
     and 21.5 -> 9.4 GB on one A100, as a 274-line diff (216 of them tests). **Caveat:** that measurement
     is on a "CLIC decoder-scope model" (`cdc2f3b` message), i.e. partial quantization via `quant_scope`
     (`2641296`), not the all-quantized production model (1140 ms bare loop). Its gain on the full model is
     not measured. Adopting it also needs the relock `da29f54`, and removing `TORCHDYNAMO_DISABLE=1` from
     `polaris/05_sweep.pbs:34`, because the option refuses to run under that variable (`cdc2f3b` message).

## 1. Verdict on the premise

| Claim | Verdict | Evidence |
|---|---|---|
| "Lightning makes HGQ2 training slow" | **Not the main cause. Lightning adds at most about 1.3x.** | Bare torch loop without Lightning: 1140 ms/step at batch 32 (`093757c`, `be28dfa`, `polaris/07_profile.py`). The 12.7 h epoch / (31,075 x 1.14 s = 9.8 h) = 1.30 (`be28dfa`, `polaris/08_profile_backend.py:4-10`). The in-run Lightning step is 1.0083 s per micro-batch on 1 GPU (`polaris/README.md:96`, run 7598775), which is *below* the bare loop's 1140 ms. The conditions differ (EBOPs every 8 steps in production). |
| The step is mostly a fixed per-op cost | **Supported.** | A100 batch scan of 893.6 / 933.5 / 945.4 / 1141.3 ms at batch 4/8/16/32: "8x the work for 28% more time: ~890 ms of every step is fixed dispatch cost" (`e998873`). In the CPU spike, `torch-loop` costs 680.7 ms fixed + 0.44 ms/event, which is 98% fixed at batch 32 (section 2). |
| The overhead is Keras-on-torch | **Partly.** It is HGQ2's quantizer ops, not Keras layers. | aten ops per fwd+bwd step (TorchDispatchMode counter, CPU): quantized with EBOPs = 55,590; quantized without EBOPs = 44,266; float = 6,048. The count is the same at batch 2 and batch 8. The float model has the same Keras layer graph and costs 161 ms/step at batch 32 (`093757c`). |
| Compiling the step removes the overhead | **Supported on CPU. Not measured on GPU.** | CPU spike: the fixed cost falls from 680.7 ms (`torch-loop`) to 34.7 ms (`jax-jit`). Whole-module `torch.compile` measured 0.77x, i.e. slower, because "Keras-torch dispatch breaks the graph at every layer" (`polaris/README.md:90`). Compiling only HGQ2's quantizer functions on torch gave 1.65x on a decoder-scope model (`cdc2f3b`). |
| "Keras with no backend" | **Not possible.** | See section 0, item 1. On the torch backend, `jit_compile=True` means `torch.compile` and `"auto"` means eager (`SP/keras/src/trainers/trainer.py:130-137, 256-259`). |

Not yet measured, and not needed for the decision:
- A matched Lightning A/B: same commit, config and DataLoader, bare loop vs `Trainer.fit`, mean and median per iteration.
  The 13 timer misses data fetch and logging (`polaris/13_step_time_fit.py:40-48`).
- A CPU-vs-GPU self-time profile of the production step.

## 2. Backend: JAX

**Recommended: JAX.** The numbers below are from the local CPU spike (`polaris/14_backend_spike.py`, commit `1a94dfe`).
It used an M2 Pro, dim 64, 4 encoder + 2 decoder layers, the `pflow_hgq.yaml` quantizer config and the EBOPs
penalty, all in fp32. Each value is the median ms/step over 20 steps (3 warmup), each run in its own process.

| mode | batch 4 | batch 32 | fixed ms | ms/event | peak RSS GB (b32) | first step s |
|---|---|---|---|---|---|---|
| `torch-loop` (production-like, no Lightning) | 682.4 | 694.7 | 680.7 | 0.44 | 0.52 | 3.2 |
| `torch-eager` (Keras `train_on_batch`) | 831.0 | 839.9 | 829.7 | 0.32 | 0.53 | 1.2 |
| `jax-jit` (`compile(jit_compile=True)` + `train_on_batch`) | **80.4** | **400.1** | 34.7 | 11.42 | 3.02 | 13.5 |
| `jax-eager` (`jit_compile=False`) | 2718.2 | 3588.6 | 2593.9 | 31.09 | 4.39 | 33.5 |
| `jax-stateless` (one `jax.jit` step: `stateless_call` + `stateless_apply`) | **75.3** | **403.8** | 28.4 | 11.73 | 2.73 | 13.0 |

With Linformer attention (k=256), the medians at batch 4 / 32 were: `torch-loop` 691.7 / 895.3 ms,
`jax-jit` 103.3 / 528.3 ms, `jax-stateless` 103.8 / 545.8 ms (`sweep_linformer_dim64.json`; median of
10 steps after 2 warmup). With Linformer the batch-32 gain is 1.69x (`jax-jit`) and 1.64x (`jax-stateless`).

Why JAX:
- **Only a compiled step removes the fixed cost.** `jax-eager` is 34x (batch 4) and 9.0x (batch 32) slower than `jax-jit`.
- **A hand-written stateless step costs nothing extra.** `jax-stateless` is within 7% of Keras's own
  jitted `train_on_batch` (6.3% faster at batch 4, 0.9% slower at batch 32; 3.3% slower with Linformer at
  batch 32). So the custom loop we need (host matching, beta updates, EBOPs gating) costs no speed.
- **HGQ2 trains correctly under `jit`.** In every mode, all 264 bitwidth variables moved and the loss fell.
  The step-0 loss minus the task loss equals sum(beta x EBOPs) to a relative error of at most 2.8e-6
  (`spikeHJpB/final_check.json`). Independent evidence: HGQ2 issue #10 reports
  "bit-identical memory across 400 steps" on JAX with `jit_compile=True`
  (https://github.com/calad0i/HGQ2/issues/10). The maintainer wrote "Try JAX, which solves 99% of
  tf-specific problems" (https://github.com/calad0i/HGQ2/issues/14).
- **TensorFlow is a worse choice.** It was not measured (TensorFlow is not installed in `.venv-jax`).
  Keras turns XLA off on CPU-only machines only for `jit_compile="auto"`
  (`SP/keras/src/trainers/trainer.py:256-267`); an explicit `jit_compile=True` is kept unless a layer
  reports `supports_jit=False` or TF sees an Apple GPU (`trainer.py:245-254,1208-1228`). So a CPU
  measurement on our Macs was possible; it was not done. The reasons for JAX are the measured JAX numbers
  and HGQ2 issue #14 below, not a TF limitation.
- **JAX runs on Polaris.** It is in the ALCF `conda` module (0.11.1 in `conda/2026-10-01`,
  https://github.com/argonne-lcf/user-guides/blob/main/docs/polaris/data-science/frameworks/jax.md),
  and `jax[cuda12]` pip wheels exist (https://docs.jax.dev/en/latest/installation.html).

Caveats:
- **Batch scaling on CPU.** On CPU, the XLA cost per event (11.4 ms) is 26x torch's (0.44 ms), so the
  two lines cross near batch 59 at dim 64. The cause is unknown (a plain matmul is equally fast in both).
  This is why the A100 measurement decides.
- **Host memory.** JAX also used 2-6x more host memory than torch on CPU.

**If HGQ2 cannot be jitted on the GPU** (compile error, failed checks, or OOM where torch fits): stop
the port. Adopt `cdc2f3b` (`MPflowHGQ(hgq_train_compile=True)`) on the hybrid. That option pins HGQ2 to
the fork `helenllii/HGQ2@4235490`, so upstreaming that fork becomes the dependency risk. TensorFlow is
not a fallback: it is unmeasured, it compiles through the same XLA, and HGQ2's maintainer points to JAX.

## 3. Target architecture

### 3.1 Package and rules

- New package `src/hepattn/keras_native/`, written with `keras.ops` only. **No `import torch`** anywhere
  in it (a test enforces this). The hybrid `src/hepattn/keras/` stays unchanged. It remains the production
  path and the parity oracle until gate G4.
- `factory.py` moves to `keras_native/factory.py` unchanged, with `import keras` in place of
  `from hepattn.keras import keras` (`src/hepattn/keras/factory.py:23`). `hepattn/keras/factory.py`
  becomes a re-export shim. The move is needed because `hepattn/keras/__init__.py:13-30` raises under any
  backend except torch.
- Pure-Python helpers are copied into `keras_native`, because their modules import torch:
  `get_hybrid_norm_config` (`src/hepattn/models/norm.py:66-85`, module imports torch at line 1) and the
  `FeatureScaler` transforms (`src/hepattn/utils/scaling.py:4` imports torch).
- Keep the deterministic layer names (`encoder_l{i}_attn_*`, `decoder_l{i}_*`, `task{i}_*`, `innet{i}_*`).
  Weight paths, the converter and `.weights.h5` all depend on them.
- Each container becomes a `keras.layers.Layer`, and the top level becomes a `keras.Model`. Build the model
  eagerly with one real batch before the first `jax.jit` call. Under tracing, HGQ2's build check hits
  `TracerBoolConversionError`. Keras then falls back to an eager build (measured in the spike, harmless),
  but every layer must be built before tracing.
- Configurations we do not use **raise** instead of being ported: dynamic queries, kmeans, flex/local
  strided attention, and MTL (`mtl: false`).

### 3.2 Model (all `keras.ops`, static shapes: 160 nodes, 8 registers, 150 queries, 150 objects)

| Component | Torch construct today | Keras-native replacement |
|---|---|---|
| Dense / FFN / heads | `KerasDense(nn.Module)`, built lazily at the first forward (`src/hepattn/keras/dense.py:63,174-190`). SwiGLU uses `torch.chunk` + `F.silu` (198-199). | A `Layer` that builds its HGQ2 sublayers in `build(input_shape)` from the static shape. SwiGLU becomes `keras.ops.split` + `keras.activations.silu`. `training` is passed through `call`. |
| Linformer (org/main PR #9: masked Linformer in every attention, k = 64; RETARGET_ORG_MAIN.md section 5.6) | `Attention(attn_type="linformer")` with `proj_k`/`proj_v` (seq_len, k) `nn.Parameter`s; `masked_linformer.project` (encoder, decoder query self-attention) and `attend` (decoder cross-attentions), keys gathered by `kv_sort_idx` first (`src/hepattn/models/attention.py:317-331`) | A branch of native `Attention` on the quantizable core of `keras_native/masked_linformer.py` (`AttendCore`, `ProjectCore`: `QEinsumDense` sequence projections, renorm by a 1-bit-mask `QEinsum` and a rank-2 rsqrt `QUnaryFunctionLUT`, unmasked `QSoftmax` over the k slots). The core kind is fixed per call site (static graph). The hybrid's `KerasAttention` runs the same core under the torch backend (gate H0, [R4b.md](port_log/R4b.md)). *Before the retarget this row kept Helen's layer (`kv_mask`, `q_mask`, `qkv_norm` ignored); that layer is deleted.* |
| Dense attention (`attn_type: torch`; used by parity tests) | `unflatten`/`transpose`, torch norms, `merge_masks` with `torch.full` (`src/hepattn/models/attention.py:62,64`) | `keras.ops.reshape`/`transpose`; `merge_masks` via `keras.ops.logical_and`; `keras.layers.LayerNormalization(epsilon=1e-5)` (the Keras default is 1e-3). The value residual is returned explicitly instead of written into a dict. |
| Encoder, Residual, register tokens | torch `Residual`/LayerNorm/LayerScale; `register_tokens = nn.Parameter(torch.randn(1,8,dim))` (`src/hepattn/keras/encoder.py:111`) | A Keras `Residual` (pre/post/hybrid norm). Register tokens use `add_weight(RandomNormal(stddev=1))` + `keras.ops.broadcast_to` + `keras.ops.concatenate`, then a static slice. DropPath stays off (`drop_path=0`). If it is ever enabled, it needs `keras.random.SeedGenerator` state. |
| Input net + Fourier posenc | in-place `x += posenc` (`src/hepattn/models/input.py:57`). `B` is a `Buffer` (`src/hepattn/models/posenc.py:131`); in-place `xs @= B` (137). | Functional. `B` becomes a **non-trainable** `add_weight`, so it is saved and restored. Its value is ported from torch for parity. |
| Decoder (4 layers) | Inherited torch forward (`src/hepattn/keras/decoder.py:23,82`). Tasks OR their masks in place (`src/hepattn/models/decoder.py:276`). The boolean-mask scatter at line 291 and the gather at `src/hepattn/utils/model_utils.py:17` have data-dependent shapes. | A static Python loop over layers and tasks; `keras.ops.logical_or`. There is a single input type (`node`, `pflow_hgq.yaml:105,116`), so `key_is_node` is all True: the scatter and the `unmerge_inputs` gather become identity / static slices. Use `keras.ops.stop_gradient` for `.detach()`. **All-False-row restore** (`unmask_all_false`, default True, `decoder.py:40,296`): `keras.ops.where(keras.ops.all(~m, axis=-1, keepdims=True), True, m)`. `initial_queries` becomes `add_weight`. Outputs are a nested dict pytree per layer. |
| Classification task | `torch.softmax`; `loss_weights` buffer (`src/hepattn/models/task.py:187-193,212`) | `keras.ops.softmax`; class weights are a constant. |
| Hit-mask task | `torch.einsum`, then the in-place `logit[~valid] = finfo.min` (`task.py:557,562`); the quantized variant uses `masked_fill`; `attn_mask = sigmoid(.detach()) >= thresh` (571) | A `QEinsum`/einsum built through `LayerFactory`, then `keras.ops.where(valid, logit, finfo.min)`. **Keep the `finfo.min` sentinel for parity** (torch writes it too). Since org `a4263c1` (bug 1) padded nodes are multiplied by the event's pad mask, so `mask_bce` stays finite at random init and on dummy targets ([R1.md](port_log/R1.md)); before, it was `inf` in both frameworks. |
| Incidence task | `QEinsum` + `QSoftmax(axis=1)` via a class swap (`src/hepattn/keras/tasks.py:51-60,99`) | Built directly through `LayerFactory`. The class-swap mechanism is dropped. |
| Incidence-based regression | `torch.cat`/`bmm`; `get_proxy_feats` uses topk on axis -2 + `torch.scatter` (`src/hepattn/utils/masks.py:304-307`), and in-place column writes (`task.py:1623-1632`) | Transpose, then `keras.ops.top_k`, then a `keras.ops.one_hot` sum to build the boolean mask. Columns are rebuilt with `keras.ops.stack`/`concatenate` (or `keras.ops.slice_update`). Uses `keras.ops.cosh`, `keras.ops.einsum`, and `FeatureScaler` rewritten in `keras.ops` (`log`/`log1p`/`sqrt`). |
| Quantization losses | `quant_losses()` walks the torch module tree (`src/hepattn/keras/maskformer.py:96-106`) | The `losses` list returned by `stateless_call(..., return_losses=True)` (`SP/keras/src/layers/layer.py:1089-1189`). |
| Removed outright | `register_keras_parameters`, `move_keras_variables_to`, device sync, the state-dict key check, and `clear_session` key determinism (`src/hepattn/keras/maskformer.py:53-60,146-228`; `lightning_module_hgq.py:35-136`) | Not needed. JAX state is an explicit pytree. |

### 3.3 Costs and losses (inside `jit`)

The torch versions use boolean indexing (`src/hepattn/models/loss.py:78-79,212-214,263`;
`task.py:828-829,1589-1590`), which gives data-dependent shapes. Every masked reduction becomes
`sum(where(m, l, 0)) / maximum(sum(m), 1)` with `keras.ops.where`/`sum`/`maximum`, keeping each torch
normalisation exactly:
- `mask_bce`, `mask_dice`: per object over its OWN event's valid nodes (count clamped at 1), then the mean over the valid
  objects (`_mean_over_valid_objects`, org `a4263c1`, bug 1; R1). Before R1 the torch formula averaged across events.
- Weighted `object_ce`: `F.cross_entropy(weight=...)` means sum(w_i l_i)/sum(w_i) (`loss.py:37`). It ignores the mask.
- `mask_dice`, `kl_div` (`loss[mask].mean()`, `loss.py:263`), regression `l1`: as in torch.
- BCE with logits in the stable `keras.ops.softplus` form. Costs are cast to fp32 explicitly; this replaces
  the `torch.autocast(enabled=False)` blocks (`loss.py:126,194,254,330`).
- **NaN-padded regression targets** (`src/hepattn/experiments/clic/pflow_data.py:564`): apply
  `keras.ops.nan_to_num` **before** the loss. `where(mask, |out - nan|, 0)` still gives a NaN gradient.
- The `torch.compile(dynamic=True)` wrappers (`loss.py:351-370`) are dropped. They only served performance.

### 3.4 Hungarian matching

Today, every step solves (4 decoder layers + final) x B = 5B assignment problems with scipy, on up to
150 x 150 costs, in a 16-thread pool (`src/hepattn/models/matcher.py:107-123,259-270`;
`pflow_hgq.yaml:145-151`; `src/hepattn/models/maskformer.py:315-359`).

**Decision: (A) host scipy inside the single jitted step through `jax.pure_callback`, on
`jax.lax.stop_gradient(costs)`.** The callback runs the *existing* matcher logic on the host: slice to
the valid length, solve, then append the unmatched default indices (`matcher.py:96-104`). So index
semantics equal torch's except for ties. Outputs are permuted with `keras.ops.take_along_axis(out, idx[..., None], axis=1)`.

Measured locally (CPU, scratch probes `match_probe.py` and `match_probe_calls.py`) on a toy model:
- **(A) `pure_callback` inside `jax.jit(jax.value_and_grad(...))`:** the loss matches the
  "match outside, then differentiate" reference to 4.8e-7, with a maximum gradient difference of **0.0**.
  An identity matching (negative control) differs by 7.6 in loss and 0.67 in gradient, so the check
  discriminates. The callback ran **once per step** (5 calls in 5 steps), with 1 compiled variant.
- **(B) split step:** jit #1 returns outputs, costs and the `jax.vjp` pullback. The host matches. jit #2
  computes the loss and applies the pullback. It gave the same gradients (difference 0.0). This is the
  fallback if (A) fails under multi-device sharding.

Measured host solve cost (`lsap_bench.py`; M2 Pro; scipy 1.18.1; random fp32 costs; 16 threads as in
production; median of 7). The Polaris host CPU is not measured.

| batch | LSAPs/step | full length 150, 16 threads | full length, 1 thread | lengths U[30,100], 16 threads |
|---|---|---|---|---|
| 32 | 160 | 16.1 ms | 104.5 ms | 4.0 ms |
| 128 | 640 | 58.0 ms | 422.2 ms | 16.2 ms |

Device-to-host copy per step: 5 x B x 150 x 150 x 4 B, which is 14.4 MB at B=32 and 57.6 MB at B=128
(arithmetic; the transfer time is not measured).

**(C) On-device matching** is a later option (S9), used only if G3 shows matching above 15% of the step.
`optax.assignment.hungarian_algorithm` is pure JAX with `lax.while_loop`, and it is tested under `jit`
and `vmap` against scipy
(https://github.com/google-deepmind/optax/blob/main/optax/assignment/_hungarian_algorithm.py). It is not
installed and not measured. It needs fixed shapes: padded target rows get a **constant** cost. Checked
locally: on 200 random problems, constant padding left the optimal valid-row cost unchanged (0/200
mismatches). Random padding (negative control) changed it in 197/200.
Scenic's `cpu_matcher` (`jax.custom_vjp` around `jax.pure_callback`) is not needed, because
`stop_gradient` already gives correct gradients (measured above).

### 3.5 Train step

The pattern is the measured `jax-stateless` mode (`polaris/14_backend_spike.py:484-500`). Names in
`snake_case` without a module prefix are ours.

```python
def loss_fn(tv, ntv, batch, ebops_on):          # ebops_on: static Python bool (section 3.6)
    out, ntv, qlosses = model.stateless_call(tv, ntv, batch["inputs"], training=True, return_losses=True)
    costs = jax.lax.stop_gradient(model_costs(out, batch["targets"]))           # fp32, (5B, P, T)
    idx = jax.pure_callback(host_match, jax.ShapeDtypeStruct(idx_shape, "int32"), costs, batch["target_valid"])
    parts = model_losses(permute(out, idx), batch["targets"])                    # nested dict, functional
    return sum_tree(parts) + sum(qlosses), (ntv, parts, qlosses)

@partial(jax.jit, static_argnames="ebops_on")
def train_step(tv, ntv, ov, batch, ebops_on):
    (loss, (ntv, parts, qlosses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(tv, ntv, batch, ebops_on)
    tv, ov = optimizer.stateless_apply(ov, grads, tv)
    return tv, ntv, ov, loss, parts, qlosses
```

- Logging, metrics, beta control and checkpointing run on the host between steps, every
  `log_every_n_steps` (200 in the Polaris config, `polaris/03_make_config.py:148`; 1000 in `pflow_hgq.yaml:50`). The `batch_idx` branch (`src/hepattn/models/wrapper.py:94`)
  and the `self.log` calls (54-62) leave the traced code.
- Why not `Model.fit`: the JAX trainer purges variables during an epoch and only re-reads them after a
  callback sync (`SP/keras/src/backend/jax/trainer.py:440-460,486-490,880-904`). Per-batch beta writes
  would be dropped unless every batch calls `jax_state_sync`, and the PeriodicEBOPs toggle needs two
  executables. The custom loop gives the same speed (`jax-stateless` ~ `jax-jit`) and direct control.

### 3.6 EBOPs and beta in a stateless step

- **EBOPs read-out (measured bug on JAX).** HGQ2 stores each layer's EBOPs in a `uint32` variable
  (`SP/hgq/layers/core/base.py:223-228`), cast at line 105. The 6 softmax exp-table layers saturate at
  4294967295 on JAX, against a true 3.72e13. On torch the value is exact only because the Keras torch
  backend stores `uint32` as `torch.int64` (`SP/keras/src/backend/torch/core.py:48`). Both
  `EBOPsMonitor` (`src/hepattn/keras/callbacks.py:74-78`) and BetaPID would silently read the saturated
  number. **Fix:** compute EBOPs = sum(qlosses) / beta. Each term is `ebops * beta` (`base.py:106`), and
  all layers share one beta. This needs beta > 0, so set `min_beta > 0` (the default is 0.0,
  `callbacks.py:260`) and start the `BetaScheduler` ramp above 0 (`beta_start` defaults to 0.0, line 171).
  The Polaris config builds the layers with `beta0: 0.0` (`polaris/03_make_config.py:206`); in pid mode
  BetaPID writes `init_beta` at the start of epoch 0 (`callbacks.py:336-340`), so beta > 0 from the first
  step, but the schedule mode starts at 0 (`03_make_config.py:229`, `beta_start: 0.0`). The read-out is only
  valid on EBOPs-active steps. Re-measured for this review: `test_periodic_ebops.py` gives
  `quant_losses` 804.96 (active) vs 0.381 (skipped), so 0.05% of the sum is not EBOPs x beta.
- **Beta.** `_beta` is a non-trainable variable (`base.py:217-221`), so it is a leaf of `ntv`. The host
  replaces those leaves (paths ending in `beta`) with a new scalar of the same shape and dtype. A value
  change does not retrace. The control law of `BetaPID.step` (`callbacks.py:311-334`) is reused, but
  **not by import**: `hepattn/keras/callbacks.py` imports torch and lightning (lines 7-8) and lives in
  `hepattn.keras`, whose `__init__.py:15-30` raises under `KERAS_BACKEND=jax`. Move the pure-Python
  controller (`__init__` validation + `step`) into a torch-free module in `keras_native`, and make the
  Lightning `BetaPID` wrap it, as for `factory.py`. The per-step `BetaScheduler` ramp becomes a host
  function of the step.
- **PeriodicEBOPs.** HGQ2 reads the Python bool `enable_ebops` at trace time (`base.py:97,103`). So
  `ebops_on` is a static argument, and the step function sets `layer._enable_ebops` while tracing: two
  cached executables, no retrace afterwards. First measure the EBOPs share of the *compiled* step in S5.
  If it is below 5%, delete PeriodicEBOPs (it saved 16% in eager torch, `093757c`).
- `_ebops.assign` and the WRAP quantizer range `assign`s run inside `stateless_call` as state updates
  (`SP/keras/src/backend/common/variables.py:293-295`). They come back in the returned `ntv`, which the
  loop must keep. The spike measured that the bitwidths train this way.

### 3.7 Optimizer, schedule, clipping, accumulation, precision

- `keras.optimizers.AdamW(weight_decay=1e-4, global_clipnorm=0.1)`. Exclude quantizer variables (paths
  containing `quantizer`, as `src/hepattn/keras/maskformer.py:132-136` does today) with
  `exclude_from_weight_decay(var_list=[v for v in model.trainable_variables if "quantizer" in v.path])`.
  **Do not use `var_names=["quantizer"]`:** it matches `variable.name`, not the path
  (`SP/keras/src/optimizers/base_optimizer.py:942-948`), and HGQ2 quantizer variables are named `f`, `i`,
  `b`. Measured on a `QDense`: `var_names` excluded 0 of 4 quantizer variables, `var_list` excluded 4.
  Beta is non-trainable, so `stateless_apply` never updates it. `keras.optimizers.Lion` covers the Lion option.
- OneCycle: a `keras.optimizers.schedules.LearningRateSchedule` subclass that reproduces torch
  `OneCycleLR(max_lr, total_steps, div_factor, final_div_factor, pct_start)` (`lightning_module_hgq.py:167-174`).
  **Known deviation:** torch also cycles Adam beta1 by default (`cycle_momentum=True`, 0.85-0.95; local
  torch `.venv-torch/.../lr_scheduler.py:2394-2396`). Keras Adam's docstring allows a callable `beta_1`
  (`SP/keras/src/optimizers/adam.py:25`), but the update casts `self.beta_1` directly (`adam.py:107-120`).
  **Measured: a callable `beta_1` fails** under `jax.jit(optimizer.stateless_apply)` with
  `TypeError: float() argument must be ... not 'function'`. So either accept a fixed beta1 = 0.9 (recorded
  as a deviation), or subclass the optimizer to read beta1 from a schedule of `iterations`.
- Gradient accumulation: `gradient_accumulation_steps=k` **averages** the gradients and then clips the
  average (`SP/keras/src/optimizers/base_optimizer.py:563-571,1168-1175`). **Lightning also averages:**
  in automatic optimization it divides each micro-batch loss by `accumulate_grad_batches`
  (lightning 2.5.2, `lightning/pytorch/loops/optimization/automatic.py:83,329`), and clips at the optimizer
  step. The two have the same semantics. The hybrid's old LR / accum and clip x accum compensation
  (it assumed Lightning *sums*) ran every job before 2026-10-14 at max LR 1.25e-5 and clip 0.4 instead
  of 5e-5 and 0.1. **Fixed** on `keras-jax`: `03_make_config.py` no longer rescales, pinned by
  `tests/polaris/test_grad_accumulation.py`. The JAX driver uses 5e-5 / 0.1 with no rescaling; a G4
  comparison must be against a hybrid run generated after the fix.
- Precision: **fp32 first** (the spike ran fp32). bf16 under HGQ2 on JAX is unverified. Decide in S8
  with `keras.mixed_precision.set_dtype_policy("mixed_bfloat16")`, gated on the quantized parity and
  training checks. The Polaris production config is micro-batch 32 x accum 4 in bf16-mixed
  (`polaris/03_make_config.py:44-46`); batch 128 OOM'd an 80 GB A100 (`polaris/README.md:92`), so
  `pflow_hgq.yaml:40,47` (batch 128) is not what Polaris runs. Whether XLA needs bf16 depends on G1's `peak_dev_gb`.
- The production config is **generated** by `polaris/03_make_config.py` from `base.yaml`, not taken from
  `pflow_hgq.yaml`. It differs in: batch/accum/precision (above), `log_every_n_steps` 200, `num_workers`
  8, PeriodicEBOPs every 8 with beta values scaled x8, a step `ModelCheckpoint` every 1000 steps, and
  `quant.ebops.beta0: 0.0` (`03_make_config.py:44-46,79-90,102-106,121,148,155-166,206,244-248`). The JAX driver
  and the S6/S8 gates must use that generated `polaris_hgq.yaml`.

### 3.8 Data

- Keep `CLICDataset` + torch `DataLoader` (8 workers on Polaris, `03_make_config.py:121`; 16 in `pflow_hgq.yaml:36`) as a host iterator. Convert each collated batch
  with `np.asarray`, then `jax.device_put`. Keras's own torch-DataLoader adapter also goes through numpy
  (`SP/keras/src/trainers/data_adapters/torch_data_loader_adapter.py:43-45`). Torch becomes a CPU-only
  data dependency of the JAX env. A numpy `load_event` (~300 lines) is optional (S9).
- Drop the `LightningDataModule` wrapper (`pflow_data.py:742`).
- Train with `drop_last=True`, so the last batch cannot trigger a new compile. Pad validation and test
  batches with invalid events instead.
- Cast int64 to int32 on purpose (JAX x64 is off).

### 3.9 Checkpointing

- The state is the host copy of `(tv, ntv, ov)` plus the step, epoch, BetaPID state, data position and a
  config hash. Write it as an `.npz` keyed by `variable.path`, plus a JSON sidecar. Keep the production
  policy: every 1000 steps, `last`, and best on `val/loss`.
- For export and interoperability, assign the state into the variables, then call `Model.save_weights(".weights.h5")`.
- Existing Polaris `.ckpt` files are loaded through the S7 converter (torch `state_dict` -> Keras paths).

### 3.10 Metrics, logging, callbacks

- `torchmetrics` (`lightning_module.py:21-104`) is replaced by host numpy count sums with the same
  micro/macro semantics. Metrics run every `log_every_n_steps` and at validation, outside `jit`.
- `MetricsCsvWriter`, `EBOPsMonitor` and `BitwidthMonitor` become host hooks of the loop. They read
  `ntv`/`tv` by path (`/b`, `/f`, `/i`) and use the EBOPs read-out of section 3.6. `keras.callbacks.CSVLogger`
  can serve if the loop exposes the Keras callback interface. Comet goes through its SDK directly.
- `PflowPredictionWriter` (`predictionwriter.py:65-238`) becomes `jit(predict)`, then numpy, then the
  **same h5 layout**, so `load_convert_h5` consumers stay unchanged.

### 3.11 Multi-GPU

- **Phase 1 (no code):** the production sweep runs independent beta points, one per GPU
  (`polaris/README.md:134`). Run one JAX process per GPU with `CUDA_VISIBLE_DEVICES`. Single-GPU
  throughput is the number that matters.
- **Phase 2 (S9, optional):** single-process data parallelism on the 4 A100s of a node. Use a
  `jax.make_mesh` / `jax.sharding.Mesh` over `jax.devices()`, put the batch on a
  `NamedSharding(mesh, PartitionSpec("batch"))`, and replicate the state; XLA then all-reduces the
  gradients. `keras.distribution.DataParallel` is the Keras-level equivalent for `fit`. **Unverified:**
  how `pure_callback` behaves with batch-sharded costs. If it fails, use matching option (B) with
  `jax.device_get` of the costs. The current torch DDP reference is 3.72x on 4 GPUs (`polaris/README.md:96`).
  Multi-node (`jax.distributed.initialize`) is out of scope.

### 3.12 Export to hls4ml

- hls4ml 1.3.0 reads values with `keras.ops.convert_to_numpy` (`hepattn/.venv/.../hls4ml/converters/keras_v3/_base.py:192`;
  `keras_v3/hgq2/_base.py:113,203-204`), which works on any backend. hls4ml (and da4ml) are not yet in
  `.venv-jax`; S7 adds them. The `torch.no_grad` wrapper in
  `src/hepattn/keras/export.py:70-80` is only needed under torch.
- Keep `export.py`'s functional assembly from live layers (`build_functional_dense`, lines 17-39). It is
  plain Keras, and the layers are now the same objects that trained.
- Saved `.keras` files have "no backend specialization" (https://keras.io/keras_3/; the page says this of
  `.keras` files, not of `.weights.h5`), so a JAX-trained model saved as `.keras` can be converted in any
  environment; a `.weights.h5` file needs the same layer paths on the loading side. The csim bit-exactness gates still run only on linux (`WORKLOG.hgq.md`, Stage 6).

## 4. Staged work breakdown

Each stage lands as its own commit series on a branch from `keras-jax`, and each stage's gate passes
before the dependent stages merge. Parity references:
- Float: the pure-torch hepattn model, **in the same process** as keras-jax (only the Keras backend is
  process-global). Inputs are numpy.
- Quantized: **golden files** from the hybrid keras-torch model, written by a script in the torch env:
  weights by path, inputs and outputs in `.npz`.

The tolerances are the `docs/hgq/PARITY.md` rows. LOC estimates include the tests.

| Stage | Content | Depends on | LOC |
|---|---|---|---|
| S0 | Polaris spike G1 (user) | — | 0 |
| S1 | Env, package skeleton, factory move, import hygiene | S0 = GO | ~100 |
| S2 | Layers: Dense/SwiGLU, Linformer, dense attention, Residual/norm, Encoder + registers, InputNet + posenc | S1 | ~760 |
| S3 | Decoder (mask attention, all-False restore), 4 task forwards, MaskFormer `call`; gate G2 | S2 | ~740 |
| S4 | Costs, losses, matching callback, permute | S1 | ~670 |
| S5 | Train step, optimizer/schedule/clip/accum, EBOPs/beta/PeriodicEBOPs, BetaPID hooks; gate G3 | S3, S4 | ~510 |
| S6 | Driver + YAML loader, data adapter, checkpoint/resume, metrics, validation loop, prediction writer, eval | S1 (stub model), then S5 | ~980 |
| S7 | Torch-ckpt -> Keras converter, export under JAX | S2 (naming), then S3 | ~250 |
| S8 | Polaris validation: production config, bf16 decision, sweep launcher; gate G4 | S5, S6, S7 | ~100 (scripts) |
| S9 | Optional: data parallelism, on-device optax matching, numpy loader, retire the hybrid | S8 | not estimated |

Total S1-S8: **about 4,100 LOC** (the inventory's per-component sum is 3,945, including about 600 LOC of tests).
**Treat this as a lower bound.** The inventory is a workflow output, not a file in the repo, so it cannot be
re-checked. For scale (`wc -l`): the hybrid itself is 2,093 LOC in `src/hepattn/keras/` plus 3,056 LOC in
`tests/keras/` (22 files), and it reuses, unchanged, about 4,000 LOC of torch code that this port rewrites
(`task.py` 1,637, `decoder.py` 475, `attention.py` 441, `maskformer.py` 429, `loss.py` 370, `encoder.py` 303,
`matcher.py` 296). The gates below name more than 35 checks, several of which port existing test files;
600 LOC of tests is not enough for them.

**What can run in parallel after S1:**
- Lane A: S2, then S3.
- Lane B: S4. The loss and cost functions are pure array functions, tested against the torch functions in-process.
- Lane C: S6 with a stub model.
- Lane D: S7 converter, once the S2 naming is fixed.

S5 needs lanes A and B. S8 needs all four lanes.

### Gates per stage (every gate is discriminating: each names what would make it fail)

- **S1:**
  - A fresh subprocess imports `hepattn.keras_native` under `KERAS_BACKEND=jax`, and `torch` is not in
    `sys.modules`. Control: the same check on `hepattn.keras` reports torch.
  - The hybrid suite `tests/keras` still passes in the torch env (factory shim).
- **S2:**
  - Float parity at the PARITY.md tolerances: dense 1e-6/1e-5; attention 1e-6/1e-5; `encoder.clic_shaped`
    2e-6/2e-5; hybrid-norm and register-token rows; Linformer 2e-5/2e-4 (`tests/keras/test_linformer_parity.py:39,52`).
    *Retarget (R2): the masked-Linformer rows are against `masked_linformer.attend`/`project` and org `Attention` at 2x
    measured (`tests/keras_native/test_masked_linformer.py`).*
  - The fully-masked-query key-independence test, with its positive control (`test_linformer_parity.py:72-73`). *Since R2
    the masked attend makes it exact for the Linformer too.*
  - Quantized parity against the golden files: first record the measured maxima, then fix each tolerance
    at 2x that. The paired delta criterion must reproduce: high bitwidth ~0.30 vs low bitwidth ~31, with
    at least 10x separation (PARITY.md, "Quantized deltas"). If the quantizers were inert, the two would not separate.
  - JIT: each layer inside `jax.jit(jax.value_and_grad(...stateless_call...))` keeps `_cache_size() == 1`
    over 3 calls, every trainable variable gets a nonzero gradient, and the EBOPs loss is present and > 0.
- **S3:**
  - `maskformer.e2e` rows at 5e-5/1e-4.
  - The per-layer mask-attention booleans are **exactly** equal.
  - All-False-row test: a query with an all-False mask attends everywhere. Turning the restore off must change the output.
  - The forward compiles once over 3 batches. Unsupported configs raise.
  - **G2 (Polaris):** a full forward + backward with a surrogate loss, at production shape and batch 32,
    jitted, must take ≤ 0.5x the hybrid's bare-loop step (`polaris/07_profile.py`, same node and batch,
    EBOPs every step in both, and the same precision: `07_profile.py:134` runs bf16 autocast, so either
    run the JAX arm in `mixed_bfloat16` or rerun 07 in fp32). If not, stop before S4-S8 merge.
- **S4:**
  - `maskformer.loss` rows at 1e-4/1e-3, including the `mask_bce` `inf` semantics (*since R1 every row is finite: bug 1*).
  - Each task's cost matrix matches the torch `task.cost` to 1e-5.
  - Over 100 random batches, the **matched total cost** equals the torch `Matcher`'s (rel 1e-6).
    Indices are not compared, because ties can differ.
  - The `pure_callback` step gradients equal the fixed-index reference (measured 0.0 on the toy; the
    identity-matching control gives 0.67), with one callback per step.
  - NaN-target test: gradients are finite with `nan_to_num` and NaN without it.
  - Weighted CE with non-uniform weights differs from the plain mean (this shows the test discriminates)
    and equals torch.
  - The permutation matches torch advanced indexing exactly.
- **S5:**
  - Port of `tests/keras/test_training.py`: over 30 steps the loss falls and the quantizers move.
  - Every trainable variable gets a nonzero gradient. This replaces `test_optimizer_covers_weights.py`.
  - The spike's penalty check: step-0 loss minus task loss equals sum(beta x EBOPs), rel err ≤ 1e-5.
  - The EBOPs read-out equals the exact torch-backend value. Control: reading `_ebops` returns 4294967295.
  - Changing beta between steps leaves `_cache_size()` unchanged and scales the penalty by the beta ratio.
  - PeriodicEBOPs reproduces the 804.96 (active) vs 0.381 (skipped) `quant_losses` of
    `tests/keras/test_periodic_ebops.py` (re-measured in this review; the test's own criterion is
    skipped < 1% of active, lines 79-80).
  - The BetaPID control-law tests (`tests/keras/test_beta_pid.py:82-184`) pass against the torch-free
    controller (section 3.6). They cannot run unchanged under `KERAS_BACKEND=jax`: line 18 imports
    `hepattn.keras.callbacks`, which raises under that backend.
  - The OneCycle LR equals torch `OneCycleLR` at 20 sampled steps.
  - k accumulated micro-batches equal one k-times batch.
  - The weight-decay exclusion covers every variable with `quantizer` in its path (control:
    `var_names=["quantizer"]` excludes 0, measured).
  - **G3 (Polaris):** the full train step at batch 32 must be **≥ 1.8x** faster than the hybrid's in-run
    step (`polaris/13_step_time.pbs`, same node), and the matching callback must be ≤ 15% of the step
    (otherwise schedule optax in S9). The in-run step uses the generated config (`13_step_time.pbs:31`):
    bf16-mixed and PeriodicEBOPs every 8. Run the JAX arm with the same EBOPs period and precision, or
    record the difference next to the result.
- **S6:**
  - Save, load and step once: the result is bitwise equal to an uninterrupted run on CPU. GPU
    determinism is not assumed, so on GPU the check uses a tolerance.
  - Over one epoch of dummy data, `_cache_size()` ≤ 2 for train (EBOPs on/off) and ≤ 1 for each of eval and predict.
  - The prediction h5 has the same keys, shapes and dtypes as `PflowPredictionWriter` for the same weights.
  - The metrics equal `torchmetrics` on fixed predictions.
  - The YAML loader reads `pflow_hgq.yaml` and the generated `polaris_hgq.yaml` (`polaris/03_make_config.py`) unchanged.
- **S7:** (*done 2026-10-20: [port_log/S7.md](port_log/S7.md); the gates as run are listed there*)
  - A converted hybrid checkpoint gives forwards equal to the hybrid's golden files at the S3 tolerances.
  - `tests/keras/test_export.py` (9 tests) passes under `KERAS_BACKEND=jax`. The csim gates run on linux.
- **S8 / G4 (Polaris):**
  - Start from the same converted initial weights and the same data order.
  - After K = 5000 steps, the JAX run's validation loss lies within the spread of two hybrid seeds.
  - The EBOPs trajectory under BetaPID tracks the setpoint as in the hybrid.
  - bf16 is adopted only if the S2/S5 quantized gates pass under `mixed_bfloat16`.
  - The final throughput is recorded in `polaris/README.md`.

## 5. Risks, go/no-go, and the Polaris measurement

### 5.1 The measurement the user must run first (G1)

`polaris/14_backend_spike.pbs` runs the same HGQ2 graph at production shape on one A100: dim 256,
16 heads, 6 encoder + 4 decoder Linformer layers, k=256. It sweeps batch 4/8/16/32 over the modes
`torch-loop`, `torch-eager`, `jax-jit` and `jax-stateless`, and records the peak device memory (`peak_dev_gb`).

1. **Get the files onto Polaris.** The two files exist only in local commit `1a94dfe` (not pushed).
   Polaris clones `keras-hgq2` (`polaris/01_setup_env.sh:38-41`). The script imports no hepattn code
   (`14_backend_spike.py:3`). So either copy `polaris/14_backend_spike.py` and
   `polaris/14_backend_spike.pbs` into `$WORK_ROOT/hepattn/polaris/`, or push `keras-jax` and check it out.
   Which one is your decision.
2. **One-time setup on a login node** (written in the `.pbs` header):
   `uv venv --python 3.12.0 .venv-jax && uv pip install --python .venv-jax/bin/python "keras==3.15.0" "hgq2==0.1.9" "quantizers==1.2.2" "numpy==2.3.5" "jax[cuda12]==0.11.2" psutil`.
   The torch modes use the existing `.venv`.
3. **Submit:** `bash polaris/submit.sh 14_backend_spike.pbs`. Read `logs/hgq_backend_spike.log` (it
   prints `nvidia-smi` and the driver version) and `logs/backend_spike_<jobid>.json`.

**GO** only if all of these hold:

| Criterion | Threshold | Why |
|---|---|---|
| Speed | median ms/step of `jax-stateless` (or `jax-jit`) at batch 32 ≤ **0.5x** `torch-loop` | The full port is at least 4,100 LOC. The no-port alternative (`cdc2f3b`) gave 1.65x on a decoder-scope model (not measured on the full model). The spike also leaves out the task heads, the losses and the host matching (16-58 ms, section 3.4). Both spike arms run fp32, while production runs bf16-mixed. |
| Overhead gone | JAX fixed cost (line through batch 4..32) ≤ 25% of its batch-32 step | Torch was 79% fixed on A100 (`e998873`) |
| Correctness | for the JAX modes: `penalty_check` rel err ≤ 1e-5, every bitwidth variable changed, `loss_last` < `loss_first` | CPU measured ≤ 2.8e-6 |
| Memory | `peak_dev_gb` (JAX, b32) ≤ `peak_dev_gb` (`torch-loop`, b32) | On CPU, JAX used 2-6x more host RAM. Batch 64 OOM'd on 40 GB with torch (`e998873`). |
| Compile | `first_step_s` ≤ 300 s | 13 s at dim 64 on CPU. Restarts can reuse a cache via `jax.config.update("jax_compilation_cache_dir", ...)`. |

`n_ebops_var_saturated_uint32 > 0` is expected on JAX. It is a known problem with a planned fix (section 3.6), not a failure.

**NO-GO** (any criterion fails, or JAX fails to compile or run on the GPU): adopt `cdc2f3b` on the hybrid,
after measuring it on the all-quantized production model (its 1.65x is from a decoder-scope model), with
`da29f54` and without `TORCHDYNAMO_DISABLE=1` in `05_sweep.pbs:34`.
Separately, run the matched Lightning A/B from section 1 if the team still wants to drop Lightning.

### 5.2 Risk register

| # | Risk | Evidence | Mitigation / gate |
|---|---|---|---|
| R1 | The CPU speedup does not carry to the A100 | The CPU lines cross near batch 59 at dim 64, cause unknown | G1 |
| R2 | The full model adds cost the spike omits (4 heads, losses, matching) | The spike is encoder/decoder + MSE only | G2 after S3 (before the loss work merges), G3 after S5 |
| R3 | Host matching latency blocks the device | 16.1 / 58.0 ms at B=32/128 locally; 14.4 / 57.6 MB D2H | G3's ≤ 15% share; optax in S9 |
| R4 | Long compiles or silent recompiles | 13 s first step at dim 64; HGQ2 needed up to 13 recompiles per function under `torch.compile` (`cdc2f3b`) | `_cache_size()` assertions in S2-S6; static shapes; `drop_last`; compilation cache |
| R5 | HGQ2-on-JAX quirks | `uint32` EBOPs saturation (measured); trace-time build fallback (measured, harmless); the upstream JAX tests run with `jax_disable_jit=True` (HGQ2 `tests/conftest.py`) | Section 3.6 fix; eager build before `jit`; our `jit` tests are the coverage, so keep them in CPU CI |
| R6 | bf16 + HGQ2 on JAX | Unverified | fp32 first; the bf16 gate in S8 |
| R7 | GPU memory under XLA | Unknown; host RSS was 2-6x higher on CPU | G1 memory criterion |
| R8 | Numerics drift silently | LayerNorm eps (Keras 1e-3 vs torch 1e-5); initializers; beta1 cycling (callable `beta_1` measured to fail); weighted-CE normalisation; weight-decay exclusion by name vs path (measured); LR/clip must match the hybrid run (no accum rescaling after the R15 fix); `finfo.min`/`inf`; NaN targets | The S2-S5 parity gates; the documented deviations in 3.7 |
| R9 | Quantized forward differs across backends at rounding boundaries | Not yet measured | Measure first in S2, then set tolerances; paired delta criterion |
| R10 | Existing `.ckpt` files become unusable | The torch `state_dict` key layout | S7 converter, gated on the golden files |
| R11 | Two code paths during migration | — | `keras_native` is isolated; the hybrid stays untouched until G4 |
| R12 | Polaris JAX environment | The driver and glibc versions for the `cuda12` wheels are not checked | The spike job prints `nvidia-smi`. Fallback: the ALCF `conda` module JAX (0.11.1). |
| R13 | `cdc2f3b` depends on an unmerged HGQ2 fork | `helenllii/HGQ2@4235490` | Only matters on NO-GO; upstream the fork |
| R14 | Baselines are not like-for-like | Spike arms fp32; 07 bare loop bf16 autocast (`07_profile.py:134`); 13 in-run bf16 + PeriodicEBOPs every 8 | State precision and EBOPs period with every G1-G3 number |
| R15 | The hybrid's accumulation compensation is wrong (Lightning averages, `automatic.py:83`) | Production LR 1.25e-5 / clip 0.4 instead of 5e-5 / 0.1 | **Fixed 2026-10-14** in `03_make_config.py` + `tests/polaris/test_grad_accumulation.py`; compare only against post-fix hybrid runs |

### 5.3 Decisions needed from the user

1. Run G1 (section 5.1). Choose between copying the two files and pushing `keras-jax`.
2. On GO, accept: fp32 first, with bf16 decided in S8; beta > 0 always (`min_beta > 0`, ramp start > 0); a
   fixed Adam beta1 (a callable `beta_1` was measured to fail) or a custom optimizer subclass; a torch CPU
   dependency for the data loader until the optional S9.
3. Whether single-process 4-GPU data parallelism is needed (S9), or one process per GPU (the current
   sweep pattern) is enough.
4. ~~Whether to fix the hybrid's grad-accumulation LR/clip compensation (R15)~~ Done 2026-10-14.

## Appendix: evidence index

- Spike: `polaris/14_backend_spike.py`, `polaris/14_backend_spike.pbs` (commit `1a94dfe`). Results are in the
  session scratchpad: `spikeHJpB/sweep_dim64.json`, `sweep_linformer_dim64.json` and `final_check.json`, under
  `/private/tmp/claude-501/-Users-eertorer-Desktop-CMS-Group-LightweightTransformers4RECO-hepattn/344375f2-3963-489b-a70c-e605492b4dce/scratchpad/`.
- Probes written for this plan (same scratchpad):
  - `lsap_bench.py` / `.log`: host LSAP cost and the constant-padding check.
  - `match_probe.py` / `.log` and `match_probe_calls.py` / `.log`: `pure_callback` and the `vjp` split.
  - `api_check.py` / `.log`: the 100 Keras/JAX/HGQ2 names this plan uses all resolve in keras 3.15.0, jax 0.11.2
    and hgq2 0.1.9. `optax.assignment.hungarian_algorithm` is the exception: it is cited from the optax source and is not installed.
- Review probes (2026-10-14, same scratchpad, `review/`): `probe_opt.py` (weight-decay exclusion by
  `var_names` vs `var_list`; callable `beta_1` under jitted `stateless_apply`; LayerNormalization default
  epsilon 1e-3), `test_periodic_ebops_print.py` / `periodic.log` (804.96 vs 0.381), and a rerun of `sat.py`
  (JAX `_ebops` 4294967295 vs torch 37239136452608 for `e0_sa_softmax_exp_table`).
- History: `e998873` (batch scan), `be28dfa` / `093757c` (bare loop, EBOPs every N), `5ebea38` / `6a20659`
  (optimizer and device fixes), `cdc2f3b` (Helen, quantizer compile); `polaris/README.md:90-96`.
- Inventory of torch constructs per component, and the premise check: the workflow outputs that fed this plan.
  `WORKLOG.hgq.md` (in the main `hepattn` checkout, git-excluded) has the hybrid's history.
- External:
  - https://keras.io/getting_started/ and https://keras.io/keras_3/
  - https://github.com/calad0i/HGQ2 (issues #10 and #14)
  - https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html
  - https://github.com/google-research/scenic (`scenic/model_lib/matchers/`)
  - https://github.com/google-deepmind/optax/blob/main/optax/assignment/_hungarian_algorithm.py
  - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
  - https://docs.jax.dev/en/latest/installation.html
  - ALCF JAX: https://github.com/argonne-lcf/user-guides/blob/main/docs/polaris/data-science/frameworks/jax.md
