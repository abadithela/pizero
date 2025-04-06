#!/bin/bash

# Define parameters
seeds=(0)
sim_data_nums=(0 10 15 20 30 40)
sample_sim_strat="sim_uniform"
num_base_demos_per_factor=20

# Define path groups
background_paths=("/n/fs/robot-data/guided-data-collection/data/sim_new_variation/background")
distractor_paths=("/n/fs/robot-data/guided-data-collection/data/sim_new_variation/distractor")
table_texture_paths=("/n/fs/robot-data/guided-data-collection/data/sim_new_variation/table_texture")
lighting_paths=("/n/fs/robot-data/guided-data-collection/data/sim_new_variation/directional")
camera_pose_paths=("/n/fs/robot-data/guided-data-collection/data/sim_new_variation/camera_pose")

path_groups=(
    camera_pose_paths
    table_texture_paths
    lighting_paths
    background_paths
    distractor_paths
)

delta=$((sim_data_nums[1] - sim_data_nums[0]))
num_base_demos=$((num_base_demos_per_factor * ( ${#path_groups[@]} - 1 )))
additional_name="base_demo_${num_base_demos}_${sample_sim_strat}"

echo $delta


# Loop over each group as the current group
for path_group in "${path_groups[@]}"; do
    echo "PATHS_GROUP: $path_group"
    if [[ "$path_group" == "camera_pose_paths" ]]; then
        continue
    fi

    # Build the input string: all groups not current first, then the current group at the end
    FINAL_INPUT_PATHS=()
    for group in "${path_groups[@]}"; do
        if [[ "$group" != "$path_group" ]]; then
            declare -n arr="$group"
            FINAL_INPUT_PATHS+=("${arr[@]}")
        fi
    done
    # Append the current group at the end
    declare -n current_arr="$path_group"
    FINAL_INPUT_PATHS+=("${current_arr[@]}")
    INPUT_STRING="${FINAL_INPUT_PATHS[*]}"

    job_ids=()

    if [[ "$path_group" == "background_paths" ]]; then
        path_name="bg"
    fi

    if [[ "$path_group" == "camera_pose_paths" ]]; then
        path_name="cp"
    fi

    if [[ "$path_group" == "lighting_paths" ]]; then
        path_name="lt"
    fi

    if [[ "$path_group" == "table_texture_paths" ]]; then
        path_name="tt"
    fi

    if [[ "$path_group" == "distractor_paths" ]]; then
        path_name="dis"
    fi

    # For each combination of seed and real_data_num, compute the distribution
    for seed in "${seeds[@]}"; do
        for sim_data_num in "${sim_data_nums[@]}"; do
            distributed_sim_traj=()
            # First, process all non-current groups (each gets a fixed total of 25)
            for group in "${path_groups[@]}"; do
                if [[ "$group" != "$path_group" ]]; then
                    declare -n arr="$group"
                    total=$num_base_demos_per_factor
                    num_paths=${#arr[@]}
                    base=$(( total / num_paths ))
                    rem=$(( total % num_paths ))
                    for (( i=0; i<num_paths-1; i++ )); do
                        distributed_sim_traj+=("$base")
                    done
                    distributed_sim_traj+=("$(( base + rem ))")
                fi
            done
            
            total=$sim_data_num
            num_paths=${#current_arr[@]}
            base=$(( total / num_paths ))
            rem=$(( total % num_paths ))
            for (( i=0; i<num_paths-1; i++ )); do
                distributed_sim_traj+=("$base")
            done
            distributed_sim_traj+=("$(( base + rem ))")

            SIM_TRAJ_STRING="${distributed_sim_traj[*]}"
            echo "SIM_TRAJ_STRING: $SIM_TRAJ_STRING"
            echo ${path_name}_$((num_base_demos + sim_data_num)) 

            job_id=$(sbatch --parsable process_data.sh \
                -in $INPUT_STRING \
                --num_sim_traj ${distributed_sim_traj[*]} \
                --num_real_traj 0 \
                --seed "$seed" \
                --additional_name ${additional_name}_${path_name}_pi \
                --sample_sim_strat $sample_sim_strat \
                --delta $delta \
                --num_per_instance 30 \
                --num_instances 4 \
                )

            job_id2=$(sbatch --parsable --dependency=afterok:$job_id compute_norm_stats.sh \
            --repo_id ${additional_name}_${path_name}_pi_seed${seed}_sim$((num_base_demos + sim_data_num))_real0 \
            )

            job_id3=$(sbatch --parsable --dependency=afterok:$job_id2 train.sh \
                --exp_name ${path_name}_$((num_base_demos + sim_data_num )) \
                --repo_id ${additional_name}_${path_name}_pi_seed${seed}_sim$((num_base_demos + sim_data_num))_real0 \
                --use_droid 0 \
                --num_train_steps 5001 \
                --keep_period 100 \
                --batch_size 32 \
                --freeze_llm 1 \
                --freeze_img 1 \
                --fsdp_devices 2 \
                --ema_decay None \
                --overwrite)

            cd ..
            sbatch --parsable --dependency=afterok:$job_id3 eval_sim_pi0.sh \
                checkpoints/pi0_base_guided/${path_name}_$((num_base_demos + sim_data_num))/5000 \
                --eval_name numenv10 \
                simulation/randomization=eval_sim_pick_pi \
                simulation.num_envs=10 \
                simulation.eval_manip_pose_file=/n/fs/robot-data/guided-data-collection/data/eval_grid_pick_poses_30.npy \
                simulation.eval_goal_pose_file=/n/fs/robot-data/guided-data-collection/data/eval_grid_place_poses_30.npy
            cd pi-zero
        done
    done
done