#!/bin/bash

#SBATCH --job-name=clic-eval-test
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=/path/to/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /path/to/hepattn/src/hepattn/experiments/clic/
echo "Working dir: ${PWD}"

export TMPDIR=/var/tmp/

CKPT="logs/clic_v6_20260605-T113014/ckpts/epoch=195-val_loss=3.78800.ckpt"

PYTORCH_CMD="python main.py test \
  --config logs/clic_v6_20260605-T113014/config.yaml \
  --config configs/eval.yaml \
  --trainer.devices=1 \
  --trainer.fast_dev_run=1 \
  --ckpt_path $CKPT"

echo "Running: $PYTORCH_CMD"
apptainer run --nv --bind /blue/,/cmsuf/ /path/to/hepattn/pixi.sif \
  pixi run -e clic $PYTORCH_CMD
echo "Done!"
