# Running CLIC on HiPerGator (HPG)

HPG-specific instructions for training and evaluating the CLIC particle-flow model
on UF's HiPerGator cluster. For the model/paper overview and data-format details see
[`README.md`](./README.md).

## Prerequisites

- **GPU partitions:**
  - `hpg-b200` — B200 192 GB (up to 4/node)
  - `hpg-turin` — L4 24 GB (up to 3/node)
- **Container + env:** the repo ships a pixi Apptainer image at the repo root
  (`pixi.sif`). Training runs inside it via `apptainer run --nv ... pixi run -e clic ...`;
  the submit scripts already do this — you don't need to enter it by hand to submit.
- **CUDA module:** `module load cuda/12.8.1` (matches the container build; already in
  the submit scripts).

## Data

The CLIC ROOT files can live anywhere the compute nodes can read. On UF HiPerGator we
keep them on the CMS store (`/cmsuf`) rather than `/blue`, to stay under the `/blue`
quota (the submit scripts bind both `/blue` and `/cmsuf` into the container):

```
/cmsuf/data/store/user/<user>/hepattn_clic_data/
├── train_clic_fix.root          # 12 GB
├── val_clic_fix.root            # 309 MB
└── test_clic_common_infer.root  # 250 MB
```

Configs point at these paths directly (`data.train_path` / `valid_path` / `test_path`).
If you write a new config, make sure its paths point here and **not** at a Jupyter/
container path like `/home/jovyan/...`, which does not exist on the compute nodes.

## Submitting a training

All commands run from the experiment directory:

```shell
cd /path/to/hepattn/src/hepattn/experiments/clic
```

### Submit scripts

| Script | Partition | Hardware | Nodes × GPU | `--mem` | Config source |
|---|---|---|---|---|---|
| `submit_training_hpg.sh` | `hpg-b200` | B200 | 1 × 4 | 300G | hardcoded `configs/base.yaml` |
| `submit_training_hpg_1gpu.sh` | `hpg-b200` | B200 | 1 × 1 | 60G | hardcoded `configs/base.yaml` |
| `submit_training_hpg_l4.sh` | `hpg-turin` | L4 | 1 × 3 (= **3 L4**) | 150G | **argument** (`$1`) |
| `submit_training_hpg_l4_2nodes.sh` | `hpg-turin` | L4 | 2 × 3 (= **6 L4**) | 150G | **argument** (`$1`) |

The L4 scripts take the config as their first argument, so you can reuse them for any
config without editing the file:

```shell
sbatch submit_training_hpg_l4.sh configs/base.yaml          # 3 L4
sbatch submit_training_hpg_l4_2nodes.sh configs/base.yaml   # 6 L4
```

The B200 scripts still hardcode `configs/base.yaml` near the bottom
(`CONFIG_PATH="..."`) — edit that line to change config, or copy the pattern above to
make them argument-driven.

### Key rule: devices must match the allocation

`--ntasks-per-node` (SBATCH) **must equal** `--trainer.devices`, and
`--nodes` **must equal** `--trainer.num_nodes`. The scripts set the `trainer.*`
overrides on the `python main.py` line to match their SBATCH directives — if you
change the GPU count, change both. Total GPUs = `num_nodes × devices`; global batch =
`num_nodes × devices × data.batch_size`.

`batch_size` comes from the config unless a script overrides it. The 6×L4 script does
**not** override it (e.g. `512`/GPU → global 3072). On 24 GB L4s a large model may
OOM at epoch 0 — if so, append an override, which passes straight through to
`main.py`:

```shell
sbatch submit_training_hpg_l4_2nodes.sh configs/base.yaml --data.batch_size=256
```

### Interactive (debug) run

```shell
apptainer shell --nv --bind /blue/ /path/to/hepattn/pixi.sif
pixi shell -e clic
python main.py fit --config configs/base.yaml --trainer.devices=1
```

Everything CLIC on HPG uses the `clic` environment: the submit scripts run
`pixi run -e clic`, and it is the environment to use for the performance notebooks
too. It is the GPU `default` environment plus the analysis packages, so there is no
need to install `default` as well (each environment is ~15 GB).

### Optional: solve the matching on the GPU

A single B200 is not saturated by the CLIC model, so the training step there is
host-bound and the Hungarian matching (device-to-host copy plus a threaded solve)
dominates it. The `Matcher` has an opt-in GPU solver for exactly that case: build it
once with `pixi run -e clic bash setup/build_torch_linear_assignment.sh`, export the
`PYTHONPATH` and `LD_LIBRARY_PATH` it prints in the submit script, and set
`device_solver: jv` on the matcher in the config. See the
[top-level README](../../../../README.md#optional-solving-the-matching-on-the-gpu).
Leave it off on the L4 nodes, which the model already keeps busy.

## Outputs & monitoring

- **Training output folder:** `logs/<config-name>_<YYYYMMDD>-T<HHMMSS>/` (checkpoints in
  `ckpts/`, resolved `config.yaml`, `metadata.yaml`). The `<config-name>` is the
  config's `name:` field (e.g. `base`).
- **SLURM stdout/stderr:** `slurm_logs/slurm-<jobid>.<jobname>.out`.

```shell
squeue -u $USER                          # your queued/running jobs
tail -f slurm_logs/slurm-<jobid>.*.out     # live training log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it finishes
scancel <jobid>                            # cancel
```

## Evaluation

Run on a single GPU (see [`README.md`](./README.md) for the full flag rationale). Note
the required overrides — inference data mode, fp32, and switching attention to `torch`:

```shell
python main.py test \
    --config logs/<run-folder>/config.yaml \
    --data.test_path /cmsuf/data/store/user/<user>/hepattn_clic_data/test_clic_common_infer.root \
    --data.is_inference true \
    --trainer.precision 32-true \
    --matmul_precision highest
```

Also remove the `Compile` callback and set attention to `torch` in the config before
testing. There are ready-made eval scripts: `submit_eval_hpg_l4.sh` and
`submit_eval_test_hpg_l4.sh`. Produce performance plots with
[`notebooks/performance.ipynb`](./notebooks/performance.ipynb).
