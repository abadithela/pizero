#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/jupyter-notebook-%j.log
#SBATCH --job-name=jupyter-notebook
#SBATCH --partition=pli
#SBATCH --account=ensemblellm

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="della-gpu"
port=8888
# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.princeton.edu
Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# uv run --with jupyter jupyter lab --no-browser --port=${port} --ip=${node}
# uv run jupyter lab --no-browser --port=${port} --ip=${node}

nohup uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=checkpoints/pi0_fast_libero/my_experiment/20000 >& server.log &
# nohup uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=/scratch/gpfs/wl5915/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base >& server.log &

#SBATCH --partition=pli
#SBATCH --account=ensemblellm