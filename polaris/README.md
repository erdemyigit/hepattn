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

**2. The sweep is the experiment worth running.** On the previous cluster we measured
that **EBOPs did not move** — 1.676e15 at step 15k vs 1.6755e15 at step 29k (0.03%)
while beta nearly doubled. Even at full ramp `beta*EBOPs ≈ 6.7` against a task loss of
~25, so the resource gradient is too weak to compress the learned bitwidths. Repeating
one long run at that beta would re-measure nothing. Sweeping it maps the
accuracy-vs-resources curve, which is the open question.

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

**`no pip` inside the environment** — `python -m ensurepip` then always call
`python -m pip` (never bare `pip`, which may resolve to the system one).

**Job held or killed immediately** — the `-l filesystems=` list must include every
filesystem the job touches. Set `PBS_FILESYSTEMS` in `env.sh` to cover both your data
and work paths.

**`SMOKE-ALL-PASSED` missing** — read `logs/hgq_smoke.log` top-down; the four stages
print their own headers. Do not launch `05_sweep.pbs` until it passes.
