#!/bin/bash

#SBATCH --job-name=pg-vla
#SBATCH --output=logs/%A.log
#SBATCH --error=logs/%A.log
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --partition=pli
#SBATCH --account=ensemblellm

source /home/wl5915/code/vla/openpi/.venv/bin/activate

module purge
module load cudatoolkit/12.4

export WANDB_MODE=offline

# run script with selected configuration using torchrun
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_guided --exp-name=my_experiment --overwrite