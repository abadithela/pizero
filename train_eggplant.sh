job_id=$(sbatch --parsable ./process_data.sh \
    -in /n/fs/robot-data/guided-data-collection/data/raw_data/pick_eggplant \
    --num_sim_traj 0 \
    --num_real_traj 200 \
    --seed 0 \
    --additional_name eggplant_pi \
    --sample_real_strat ordered \
    )

job_id2=$(sbatch --parsable --dependency=afterok:$job_id compute_norm_stats.sh \
    --repo_id eggplant_pi_seed0_sim0_real200 \
)

job_id3=$(sbatch --parsable --dependency=afterok:$job_id2 train.sh \
    --exp_name eggplant_200 \
    --repo_id eggplant_pi_seed0_sim0_real200 \
    --use_droid 0 \
    --num_train_steps 5001 \
    --keep_period 100 \
    --batch_size 32 \
    --freeze_llm 1 \
    --freeze_img 1 \
    --fsdp_devices 2 \
    --ema_decay None \
    --overwrite)