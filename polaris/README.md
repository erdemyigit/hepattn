# HGQ2 quantization-aware training on ALCF Polaris

Everything here is self-contained: clone the repo on Polaris and run the scripts in
order. Site-specific values (project, queue, filesystems, paths) live in **one** file,
`env.sh`.

## Quick start

```bash
# on a Polaris LOGIN node
git clone --branch keras-hgq2 https://github.com/erdemyigit/hepattn.git
cd hepattn

bash polaris/00_discover.sh                 # prints project / queues / filesystems
cp polaris/env.sh.example polaris/env.sh
$EDITOR polaris/env.sh                      # paste in the values from 00_discover

bash polaris/01_setup_env.sh                # venv + deps + verification (~10 min)
bash polaris/02_stage_data.sh               # ~12 GB of CLIC ROOT files

bash polaris/submit.sh 04_smoke.pbs         # GATE: ~10 min on 1 GPU
# wait for "SMOKE-ALL-PASSED" in logs/hgq_smoke.log, then:
bash polaris/submit.sh 05_sweep.pbs         # 4 GPUs, 4 beta points, in parallel
```

Watch: `qstat -u $USER` · `tail -f logs/beta*.log` · cancel: `qdel <jobid>`

## How the GPUs are used, and why

`05_sweep.pbs` runs **one independent single-GPU training per GPU**, each at a
different `beta_end` (the weight on the EBOPs resource penalty). All GPUs run at
~100% with no inter-GPU communication.

This is deliberate, for two reasons.

**1. DDP over this stack is NOT validated.** Every config ships `devices: 1`. Nobody
has tested multi-GPU DDP over Keras-3-torch-backend modules carrying HGQ2 quantizers,
and there are two concrete hazards: quantized layers **build lazily at first forward**
(ranks can diverge in parameter registration), and quantizer layer names land in
`state_dict` keys (they must agree **across ranks**, not just across processes). A
silent failure here produces wrong science, not a crash. See `06_ddp_validate.pbs` if
you want to unlock it.

**2. The sweep maps the accuracy-vs-resources curve** — but **`BETA_VALUES` must be
calibrated first, and the obvious way to do it is wrong.** See the warning below.

> ### ⚠️ Do not calibrate beta from `total_ebops`
>
> The first Polaris sweep (`BETA_VALUES="4.0e-15 … 4.0e-12"`, 4 GPUs, 4 days) was
> **inert**. Measured from the checkpoints, mean learned bitwidth after 4 epochs:
>
> | beta | 4.0e-15 | 4.0e-14 | 4.0e-13 | 4.0e-12 |
> |---|---|---|---|---|
> | mean bits (from 8.0000) | 7.9815 | 7.9815 | 7.9813 | 7.9807 |
>
> **Four decades of beta produced a 0.0008-bit spread (0.01%)**, and all four runs had
> near-identical val-loss curves. Cause: that range was derived from `total_ebops`
> (~1.7e15), but `quant_losses()` is affine in beta —
> `quant_loss = floor + beta * E_eff` — and **`E_eff` is ~5 orders of magnitude smaller
> than `total_ebops`**, because ~94% of the reported total comes from `QSoftmax` layers
> whose cost never enters the regularization loss. So `beta*E_eff` was ~1e-1 against a
> task loss of ~30, not the ~6.8 the arithmetic predicted.
>
> **Calibrate against `E_eff`:**
> ```python
> from hepattn.keras.evaluate import effective_ebops
> e_eff, floor = effective_ebops(model, batch)   # measures the slope, restores beta
> beta = target_fraction * task_loss / e_eff     # e.g. 0.25 * 30 / e_eff
> ```
> Covered by `tests/keras/test_effective_ebops.py` (7 tests; 3 fail if `E_eff` is
> aliased back to `total_ebops`). `BitwidthMonitor` now logs `val/bits_mean` so an inert
> sweep is visible on day one instead of day four.

**More parallelism:** submit `05_sweep.pbs` again — each submission is another node
(4 more GPUs). Add points by editing `BETA_VALUES` in `env.sh`.

## Facts baked into these scripts (measured, don't re-derive)

| | |
|---|---|
| `flash-attn` | **not needed.** The Keras/HGQ2 path never imports it; `flash`/`flash-varlen` are coerced to torch SDPA and the config uses the in-repo linformer. Skips the hardest build step. |
| `~/.local` shadowing | A venv disables user-site by default, and `PYTHONNOUSERSITE=1` is exported everywhere — the torch/CUDA-mismatch trap cannot bite. |
| `torch.compile` | **Do not enable.** Measured 0.77× (slower): Keras-torch dispatch breaks the graph at every layer so inductor can't fuse. The `Compile` callback is stripped in `03_make_config.py`. |
| `bf16-mixed` | Validated on CUDA — finite loss, quantizer bitwidth gradients flow. |
| Batch size | Quantizers hold fp32 internal state. micro-batch 32 fits 40 GB; 64 needed ~78 GB; **128 OOM'd an 80 GB A100.** |
| Grad accumulation | Losses are mean-reduced → Lightning **sums** accumulated grads. `03_make_config.py` sets LR ÷ accum and clip × accum so an optimizer step is invariant. |
| Checkpointing | The repo's `Checkpoint` monitors `val/loss`, which never fires mid-epoch — a walltime kill would lose the whole epoch. A plain `ModelCheckpoint` (`every_n_train_steps=1000`, `save_last`) is added, and `05_sweep.pbs` auto-resumes from `last.ckpt`. |
| Test data | `test_clic_common_infer.root` has `-9999` sentinel indices that crash the loader. Configs evaluate on `val_clic_fix.root`. |
| QAT cost | ~2.3× the float per-step time on an A100. Budget accordingly. |

## Queue reality on Polaris (measured from `00_discover.sh`)

`prod`, `small`, and `medium` all have **`resources_min.nodect` >= 10** — they cannot
run a 1-node job. For this work only two queues apply:

| queue | nodes | max walltime | use |
|---|---|---|---|
| `debug` | 1-2 | 1 h | `04_smoke.pbs`, `06_ddp_validate.pbs` |
| `preemptable` | 1-10 | **72 h** | `05_sweep.pbs` (production) |

**`preemptable` jobs can be killed at any moment** to make room for higher-priority
work. That is fine here and is why the sweep checkpoints every 1000 steps, chains a
successor *before* starting, and auto-resumes each beta point from its own
`last.ckpt`. Expect restarts; they cost at most ~1000 steps of progress each.

## Files

| file | what it does |
|---|---|
| `00_discover.sh` | read-only; prints project, queues, filesystems, GPUs, user-site hazards |
| `env.sh.example` | copy to `env.sh`; the only file with site-specific values |
| `01_setup_env.sh` | uv venv (own Python 3.12) + deps + import/GPU verification |
| `02_stage_data.sh` | rsync CLIC ROOT files into project space, with size verification |
| `03_make_config.py` | generates `polaris_hgq.yaml` from `base.yaml` with all the above baked in |
| `04_smoke.pbs` | **gate**: real GPU, real data, bf16 steps, asserts quantizers exist and train |
| `05_sweep.pbs` | **main**: N GPUs × independent beta points, self-chaining across walltime |
| `06_ddp_validate.pbs` | optional: tests whether DDP is safe on this stack before trusting it |
| `submit.sh` | injects `-A` / `-l filesystems=` / `-q` from `env.sh` |

## Troubleshooting

**`detected CUDA version mismatches the version used to compile PyTorch`** — a stale
`~/.local/lib/python3.12/site-packages` is shadowing the venv. The scripts export
`PYTHONNOUSERSITE=1`; if it persists, `mv ~/.local/lib ~/.local/lib.disabled`.

**`resolved Python interpreter ... is incompatible with the project's Python
requirement: == 3.12`** — `pyproject.toml` pins `== 3.12`, which in PEP 440 means
exactly **3.12.0**, so a system 3.12.x interpreter is rejected. `01_setup_env.sh` now
fetches a uv-managed 3.12.0. To fix an existing broken venv by hand:
`rm -rf .venv && uv python install 3.12.0 && uv venv --python 3.12.0 .venv && uv sync --group hgq --no-install-project`

**`no pip` inside the environment** — `python -m ensurepip` then always call
`python -m pip` (never bare `pip`, which may resolve to the system one).

**Job held or killed immediately** — the `-l filesystems=` list must include every
filesystem the job touches. Set `PBS_FILESYSTEMS` in `env.sh` to cover both your data
and work paths.

**`SMOKE-ALL-PASSED` missing** — read `logs/hgq_smoke.log` top-down; the four stages
print their own headers. Do not launch `05_sweep.pbs` until it passes.
