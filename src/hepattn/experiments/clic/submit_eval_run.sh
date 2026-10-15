#!/bin/bash
# Parameterized CLIC eval: run inference from a run's best checkpoint to produce
# <ckpt>__test.h5 and (via PflowPredictionWriter) <ckpt>__test.root next to the ckpt.
#
# Submit with:
#   sbatch --job-name=clic-eval-<tag> \
#          --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file> \
#          submit_eval_run.sh

#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=/path/to/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "RUN_DIR: ${RUN_DIR}"
echo "CKPT_NAME: ${CKPT_NAME}"
nvidia-smi

cd /path/to/hepattn/src/hepattn/experiments/clic/
echo "Working dir: ${PWD}"

export TMPDIR=/var/tmp/

CKPT="${RUN_DIR}/ckpts/${CKPT_NAME}"

PYTORCH_CMD="python main.py test \
  --config ${RUN_DIR}/config.yaml \
  --config configs/eval.yaml \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  --ckpt_path $CKPT"

PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

APPTAINER_CMD="apptainer run --nv --bind /blue/,/cmsuf/ /path/to/hepattn/pixi.sif $PIXI_CMD"

echo "Running: $PYTORCH_CMD"
$APPTAINER_CMD
echo "Done!"
