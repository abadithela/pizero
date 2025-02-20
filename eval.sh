source examples/libero/.venv/bin/activate
# uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
# uv pip install -e packages/openpi-client
# uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

module purge
module load cudatoolkit/11.3

ssh -N -f -L 8000:della-l07g6:8000 wl5915@della-gpu.princeton.edu
# kill $(lsof -t -i:8000)

# Run the simulation
save_path="data/libero/pi0_fast_libero"
python examples/libero/main.py --args.port=8000 --args.num-trials-per-task=10 --args.video-out-path="${save_path}" >& "${save_path}.log"