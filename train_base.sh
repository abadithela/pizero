#!/bin/bash

sample_real_strat="uniform"
num_base_demos=120
additional_name="base_demo_${num_base_demos}_${sample_real_strat}_new_cp"
seed=0
sim_data_num=0
real_data_num=30
path_name="cp"

batch_size=40
fsdp_devices=2

for ema_decay in "None"; do
    for freeze_img in 0; do
        for freeze_llm in 0; do
            for use_droid in 0; do
                exp_name=${additional_name}_${path_name}_pi_seed${seed}_sim${sim_data_num}_real$((num_base_demos + real_data_num))_${batch_size}_base_2gpu_fsdp
                if [ "$freeze_img" -eq 1 ]; then
                    exp_name=${exp_name}_freeze_img
                fi
                if [ "$freeze_llm" -eq 1 ]; then
                    exp_name=${exp_name}_freeze_llm
                fi
                if [ "$use_droid" -eq 1 ]; then
                    exp_name=${exp_name}_droid
                fi
                if [ "$ema_decay" = "None" ]; then
                    exp_name=${exp_name}_noema
                fi
                sbatch train.sh \
                    --exp_name $exp_name \
                    --repo_id ${additional_name}_${path_name}_pi_seed${seed}_sim${sim_data_num}_real$((num_base_demos + real_data_num)) \
                    --use_droid $use_droid \
                    --num_train_steps 50000 \
                    --keep_period 100 \
                    --batch_size $batch_size \
                    --freeze_llm $freeze_llm \
                    --freeze_img $freeze_img \
                    --fsdp_devices $fsdp_devices \
                    --ema_decay $ema_decay \
                    --overwrite
            done
        done
    done
done