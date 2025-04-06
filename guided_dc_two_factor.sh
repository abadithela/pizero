#!/bin/bash

# Define parameters
seeds=(0)
sim_data_nums=(0)
real_data_nums=(0 10 20 30 40)
sample_real_strat="uniform"
num_base_demos_per_factor=30

# Define path groups
background_paths=("/n/fs/robot-data/guided-data-collection/data/background_new")
distractor_paths=("/n/fs/robot-data/guided-data-collection/data/distractor_new")
table_texture_paths=("/n/fs/robot-data/guided-data-collection/data/table_texture_new")
lighting_paths=("/n/fs/robot-data/guided-data-collection/data/lighting_new")
camera_pose_paths=("/n/fs/robot-data/guided-data-collection/data/camera_pose_new_new")

path_groups=(
    camera_pose_paths
    table_texture_paths
    lighting_paths
    background_paths
    distractor_paths
)

num_base_demos=$((num_base_demos_per_factor * ( ${#path_groups[@]} - 2 )))
additional_name="base_demo_${num_base_demos}_${sample_real_strat}_new_cp"

echo $delta

for ((i = 0; i < ${#path_groups[@]}; i++)); do
    for ((j = i + 1; j < ${#path_groups[@]}; j++)); do
        path_group_1="${path_groups[i]}"
        path_group_2="${path_groups[j]}"

        # # Skip if either path_group_1 or path_group_2 is background_paths or distractor_paths
        # if [[ "$path_group_1" == "background_paths" || "$path_group_1" == "distractor_paths" || \
        #       "$path_group_2" == "background_paths" || "$path_group_2" == "distractor_paths" ]]; then
        #     continue
        # fi

        echo "Selected Path Groups: $path_group_1, $path_group_2"

        # Build the input string: all groups except selected two first, then append selected two at the end
        FINAL_INPUT_PATHS=()
        for group in "${path_groups[@]}"; do
            if [[ "$group" != "$path_group_1" && "$group" != "$path_group_2" ]]; then
                declare -n arr="$group"
                FINAL_INPUT_PATHS+=("${arr[@]}")
            fi
        done
        # Append the selected two groups at the end
        declare -n arr1="$path_group_1"
        declare -n arr2="$path_group_2"
        FINAL_INPUT_PATHS+=("${arr1[@]}")
        FINAL_INPUT_PATHS+=("${arr2[@]}")
        INPUT_STRING="${FINAL_INPUT_PATHS[*]}"
        

        job_ids=()
        job_ids_exp=()

        path_name=""

        if [[ "$path_group" == "background_paths" || "$path_group_1" == "background_paths" || "$path_group_2" == "background_paths" ]]; then
            path_name+="bg"
        fi

        if [[ "$path_group" == "camera_pose_paths" || "$path_group_1" == "camera_pose_paths" || "$path_group_2" == "camera_pose_paths" ]]; then
            path_name+="cp"
        fi

        if [[ "$path_group" == "lighting_paths" || "$path_group_1" == "lighting_paths" || "$path_group_2" == "lighting_paths" ]]; then
            path_name+="lt"
        fi

        if [[ "$path_group" == "table_texture_paths" || "$path_group_1" == "table_texture_paths" || "$path_group_2" == "table_texture_paths" ]]; then
            path_name+="tt"
        fi

        if [[ "$path_group" == "distractor_paths" || "$path_group_1" == "distractor_paths" || "$path_group_2" == "distractor_paths" ]]; then
            path_name+="dis"
        fi

        echo "$path_name"


        # For each combination of seed and real_data_num, compute the distribution
        for seed in "${seeds[@]}"; do
            for sim_data_num in "${sim_data_nums[@]}"; do
                for real_data_num in "${real_data_nums[@]}"; do
                    distributed_real_traj=()

                    # First, process all non-current groups (each gets a fixed total of 25)
                    for group in "${path_groups[@]}"; do
                        if [[ "$group" != "$path_group_1" && "$group" != "$path_group_2" ]]; then
                            declare -n arr="$group"
                            total=$num_base_demos_per_factor
                            num_paths=${#arr[@]}
                            base=$(( total / num_paths ))
                            rem=$(( total % num_paths ))
                            for (( t=0; t<num_paths-1; t++ )); do
                                distributed_real_traj+=("$base")
                            done
                            distributed_real_traj+=("$(( base + rem ))")
                        fi
                    done
                    
                    distributed_real_traj+=("$((real_data_num))")
                    distributed_real_traj+=("$((real_data_num))")
                    
                    REAL_TRAJ_STRING="${distributed_real_traj[*]}"
                    echo "REAL_TRAJ_STRING: $REAL_TRAJ_STRING"
                    echo ${path_name}_$((num_base_demos + real_data_num + real_data_num)) 

                    job_id=$(sbatch --parsable process_data.sh \
                        -in $INPUT_STRING \
                        --num_sim_traj $sim_data_num \
                        --num_real_traj ${distributed_real_traj[*]} \
                        --seed "$seed" \
                        --additional_name ${additional_name}_${path_name}_pi \
                        --visualize_image \
                        --sample_sim_strat "ordered"\
                        --sample_real_strat $sample_real_strat \
                        )

                    job_id2=$(sbatch --parsable --dependency=afterok:$job_id compute_norm_stats.sh \
                    --repo_id ${additional_name}_${path_name}_pi_seed${seed}_sim${sim_data_num}_real$((num_base_demos + real_data_num + real_data_num)) \
                    )

                    job_id3=$(sbatch --parsable --dependency=afterok:$job_id2 train.sh \
                        --exp_name ${path_name}_$((num_base_demos + real_data_num + real_data_num)) \
                        --repo_id ${additional_name}_${path_name}_pi_seed${seed}_sim${sim_data_num}_real$((num_base_demos + real_data_num + real_data_num)) \
                        --use_droid 0 \
                        --num_train_steps 5001 \
                        --keep_period 100 \
                        --batch_size 32 \
                        --freeze_llm 1 \
                        --freeze_img 1 \
                        --fsdp_devices 2 \
                        --ema_decay None \
                        --overwrite)

                    sbatch --dependency=afterok:$job_id3 eval_sim_pi0.sh \
                        "checkpoints/pi0_base_guided/${path_name}_$((num_base_demos + real_data_num + real_data_num))/5000" \
                        simulation.num_envs=30
                done
            done
        done
    done
done