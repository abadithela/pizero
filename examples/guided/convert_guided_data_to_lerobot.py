"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import gc
import h5py
import numpy as np
import os
import shutil
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "lihzha/guided"  # Name of the output dataset, also used for the Hugging Face Hub


def load_hdf5(
    file_path,
    action_keys=["joint_positions", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    load_image=True,
    obs_gripper_threshold=0.2,
    binary_gripper=True,
):
    """also get the raw indices of camera images"""
    keys_to_load = ["observation/timestamp/skip_action"]
    for key in action_keys:
        keys_to_load.append(f"action/{key}")
    for key in observation_keys:
        keys_to_load.append(f"observation/robot_state/{key}")
    if load_image:
        keys_to_load.append("observation/image")

    output = {}
    camera_indices_raw = []
    h5_file = h5py.File(file_path, "r")
    for key in keys_to_load:
        if key in h5_file:
            if "image" in key:
                for cam in h5_file[key].keys():
                    output[f"{key}/{cam}"] = h5_file[f"{key}/{cam}"][()]
                    camera_indices_raw.append(int(cam))
            else:
                output[key] = h5_file[key][()]
        else:
            print(f"Key '{key}' not found in the HDF5 file.")

    # make sure to close h5 file
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            try:
                obj.close()
            except Exception:
                pass
    h5_file.close()

    output["state"] = np.concatenate((output["observation/robot_state/joint_positions"], output["observation/robot_state/gripper_position"][:, None]), axis=1)
    output["action"] = np.concatenate((output["action/joint_positions"], output["action/gripper_position"][:, None]), axis=1)
    keep_idx = ~output["observation/timestamp/skip_action"]
    if np.sum(keep_idx) < len(keep_idx):
        print(f"Kept {np.sum(keep_idx)}/{len(keep_idx)} samples")
    for key in output.keys():
        output[key] = output[key][keep_idx]
    
    if binary_gripper:
        output["action/gripper_position"] = (
            output["action/gripper_position"] > 0.5
        ).astype(np.float32)
        output["observation/robot_state/gripper_position"] = (
            output["observation/robot_state/gripper_position"] > obs_gripper_threshold
        ).astype(np.float32)

    return output, camera_indices_raw

def get_traj_paths(data_dir):
    traj_paths = [os.path.join(data_dir, traj_name) for traj_name in os.listdir(data_dir) if traj_name.endswith(".h5")]
    return traj_paths


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    traj_paths = get_traj_paths(data_dir)
    for traj_path in tqdm(traj_paths):
        data, _ = load_hdf5(traj_path)
        for step in range(len(data["state"])):
            dataset.add_frame(
                {
                    "image": data["observation/image/0"][step],
                    "wrist_image": data["observation/image/1"][step],
                    "state": data["state"][step],
                    "actions": data["action"][step],
                }
            )
        dataset.save_episode(task="pick up the tomato and put it into the metal plate")

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
