#!/bin/bash

#SBATCH --job-name=pi0_data    # Job name
#SBATCH --output=logs/%A_pi0data.out   # Output file
#SBATCH --error=logs/%A_pi0data.err    # Error file
#SBATCH --time=24:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:0            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=6       # Reduced CPU per task
#SBATCH --mem=60G                    # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if needed

# Parameter configurations
CONFIGS=("$@")

uv run examples/guided/process_pi_dataset.py "${CONFIGS[@]}"