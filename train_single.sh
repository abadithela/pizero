#!/bin/bash

#SBATCH --job-name=unet-robot    # Job name
#SBATCH --output=logs/%A.out   # Output file
#SBATCH --error=logs/%A.err    # Error file
#SBATCH --time=72:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:4            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=8       # Reduced CPU per task
#SBATCH --mem=200G                    # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if needed

sample_real_strat="uniform"
num_base_demos=120
additional_name="base_demo_${num_base_demos}_${sample_real_strat}_new_cp"
seed=0
sim_data_num=0
real_data_num=30
path_name="cp"

batch_size=64
freeze_llm=0
freeze_img=1
use_droid=1
fsdp_devices=4
exp_name=${additional_name}_${path_name}_pi_seed${seed}_sim${sim_data_num}_real$((num_base_demos + real_data_num))_${batch_size}_noema_noimg_droid

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
    --exp_name $exp_name \
    --repo_id ${additional_name}_${path_name}_pi_seed${seed}_sim${sim_data_num}_real$((num_base_demos + real_data_num)) \
    --use_droid $use_droid \
    --num_train_steps 50000 \
    --keep_period 100 \
    --batch_size $batch_size \
    --freeze_llm $freeze_llm \
    --freeze_img $freeze_img \
    --fsdp_devices $fsdp_devices \
    --ema_decay None \
    --overwrite \
    # --resume \

