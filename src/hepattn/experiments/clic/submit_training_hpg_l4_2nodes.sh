#!/bin/bash

#SBATCH --job-name=clic-train-l4-2nodes
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=2
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=/path/to/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

# Load CUDA matching the container build
module load cuda/12.8.1

# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd /path/to/hepattn/src/hepattn/experiments/clic/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the training
echo "Running training script..."

# Config is passed as the first argument, e.g.
#   sbatch submit_training_hpg_l4_2nodes.sh configs/clic_v7.yaml
CONFIG_PATH="${1:?Usage: sbatch submit_training_hpg_l4_2nodes.sh <config.yaml>}"
echo "Using config: $CONFIG_PATH"

# Python command that will be run.
# devices=3 x num_nodes=2 = 6 L4 GPUs, matching the SBATCH allocation above.
# batch_size is taken from the config (override here with --data.batch_size=N if needed).
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH --trainer.devices=3 --trainer.num_nodes=2"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
# srun in front for multi-node multi-GPU DDP
APPTAINER_CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ /path/to/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
