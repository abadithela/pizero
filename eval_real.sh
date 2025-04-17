#!/bin/bash

policy_name="pi0_towel_4_visual_factor_2_combos_sim_uniform_cpdis_seed0_sim0_real"

for i in {100..200..20}; do
    session_name="sim${i}"
    checkpoint_path="checkpoints/pi0_base_guided/${policy_name}${i}/5000"

    echo "Starting tmux session: $session_name"

    tmux new-session -d -s "$session_name" "
        echo 'Requesting salloc for index $i on session $session_name';
        salloc --nodes=1 --ntasks=6 --mem=40G --time=02:00:00 --gres=gpu:1 bash -c \"
            echo Running on node: \$(hostname);
            uv run scripts/serve_policy.py policy:checkpoint \
                --policy.config=pi0_base_guided \
                --policy.dir=$checkpoint_path
        \";
        echo 'Finished run for $i';
        exec bash
    "
done
