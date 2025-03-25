#!/bin/bash

seeds=(0)
sim_data_nums=(0)
real_data_nums=(0 10)
sample_real_strat="uniform"
num_base_demos_per_factor=30

# Define path groups
background_paths=("../guided-data-collection/data/background_new")
distractor_paths=("../guided-data-collection/data/distractor_new")
table_texture_paths=("../guided-data-collection/data/table_texture_new")
lighting_paths=("../guided-data-collection/data/lighting_new")
camera_pose_paths=("../guided-data-collection/data/camera_pose_new_new")

path_groups=(
    background_paths
    distractor_paths
    table_texture_paths
    lighting_paths
    camera_pose_paths
)

num_base_demos=$((num_base_demos_per_factor * ( ${#path_groups[@]} - 1 )))
additional_name="base_demo_${num_base_demos}_${sample_real_strat}_new_cp"

# Loop over each group as the current group
for path_group in "${path_groups[@]}"; do
    echo "PATHS_GROUP: $path_group"

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

    # For each combination of seed and real_data_num, compute the distribution
    for seed in "${seeds[@]}"; do
        for sim_data_num in "${sim_data_nums[@]}"; do
            for real_data_num in "${real_data_nums[@]}"; do
                distributed_real_traj=()

                # First, process all non-current groups (each gets a fixed total of 25)
                for group in "${path_groups[@]}"; do
                    if [[ "$group" != "$path_group" ]]; then
                        declare -n arr="$group"
                        total=$num_base_demos_per_factor
                        num_paths=${#arr[@]}
                        base=$(( total / num_paths ))
                        rem=$(( total % num_paths ))
                        for (( i=0; i<num_paths-1; i++ )); do
                            distributed_real_traj+=("$base")
                        done
                        distributed_real_traj+=("$(( base + rem ))")
                    fi
                done

                # Then process the current group using real_data_num as total
                total=$real_data_num
                num_paths=${#current_arr[@]}
                base=$(( total / num_paths ))
                rem=$(( total % num_paths ))
                for (( i=0; i<num_paths-1; i++ )); do
                    distributed_real_traj+=("$base")
                done
                distributed_real_traj+=("$(( base + rem ))")

                REAL_TRAJ_STRING="${distributed_real_traj[*]}"
                echo "REAL_TRAJ_STRING: $REAL_TRAJ_STRING"

                uv run examples/guided/process_pi_dataset.py \
                    -in $INPUT_STRING \
                    -out data \
                    --num_thread 5 \
                    --num_sim_traj $sim_data_num \
                    --num_real_traj ${distributed_real_traj[*]} \
                    --seed "$seed" \
                    --additional_name ${additional_name}_${path_group} \
                    --visualize_image \
                    --sample_sim_strat "cluster_pick"\
                    --sample_real_strat $sample_real_strat

        done
    done
done