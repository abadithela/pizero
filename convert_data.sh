source .venv/bin/activate
# uv run examples/guided/convert_guided_data_to_lerobot.py --data_dir /home/wl5915/code/vla/datasets/base_demos

uv run scripts/compute_norm_stats.py --config-name pi0_fast_guided
