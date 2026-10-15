#!/bin/bash

#SBATCH --job-name=clic-train
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:4
#SBATCH --ntasks-per-node=4        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=48:00:00
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

# Python command that will be run
CONFIG_PATH="configs/base.yaml"
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
# srun in front for multi-GPU DDP
APPTAINER_CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ /path/to/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
