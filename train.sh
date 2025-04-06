#!/bin/bash

#SBATCH --job-name=unet-robot    # Job name
#SBATCH --output=logs/%A.out   # Output file
#SBATCH --error=logs/%A.err    # Error file
#SBATCH --time=72:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:2            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=8       # Reduced CPU per task
#SBATCH --mem=200G                   # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if needed

CONFIGS=("$@")

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py "${CONFIGS[@]}" 