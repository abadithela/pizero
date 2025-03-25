"""
Script for processing raw teleop data for policy training.

Create a new folder and then save dataset and normalization values in the folder. Also save the config in txt.

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
from multiprocessing import cpu_count
import os
import time

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm

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

    output["state"] = np.concatenate(
        (
            output["observation/robot_state/joint_positions"],
            output["observation/robot_state/gripper_position"][:, None],
        ),
        axis=1,
    )
    output["action"] = np.concatenate(
        (output["action/joint_positions"], output["action/gripper_position"][:, None]),
        axis=1,
    )
    keep_idx = ~output["observation/timestamp/skip_action"]
    if np.sum(keep_idx) < len(keep_idx):
        print(f"Kept {np.sum(keep_idx)}/{len(keep_idx)} samples")
    for key in output:
        output[key] = output[key][keep_idx]

    if binary_gripper:
        output["action/gripper_position"] = (output["action/gripper_position"] > 0.5).astype(np.float32)
        output["observation/robot_state/gripper_position"] = (
            output["observation/robot_state/gripper_position"] > obs_gripper_threshold
        ).astype(np.float32)

    return output, camera_indices_raw


def load_sim_hdf5_for_training(
    file_path,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    load_image=True,
    obs_gripper_threshold=0.2,
    binary_gripper=True,
):
    """also get the raw indices of camera images"""
    keys_to_load = []
    for key in action_keys:
        keys_to_load.append(f"action/{key}")
    for key in observation_keys:
        keys_to_load.append(f"observation/robot_state/{key}")
    if load_image:
        keys_to_load.append("observation/image")

    output = {}
    h5_file = h5py.File(file_path, "r")
    for key in keys_to_load:
        if key in h5_file:
            if "image" in key:
                for cam in h5_file[key].keys():
                    output[f"{key}/{cam}"] = h5_file[f"{key}/{cam}"][()]
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

    output["action/gripper_position"] = 1 - (output["action/gripper_position"] + 1) / 2
    output["observation/robot_state/gripper_position"] = 1 - (output["observation/robot_state/gripper_position"] / 0.04)

    output["state"] = np.concatenate(
        (
            output["observation/robot_state/joint_positions"],
            output["observation/robot_state/gripper_position"][:, None],
        ),
        axis=1,
    )
    output["action"] = np.concatenate(
        (output["action/joint_positions"], output["action/gripper_position"][:, None]),
        axis=1,
    )

    if binary_gripper:
        output["action/gripper_position"] = (output["action/gripper_position"] > 0.5).astype(np.float32)
        output["observation/robot_state/gripper_position"] = (
            output["observation/robot_state/gripper_position"] > obs_gripper_threshold
        ).astype(np.float32)

    camera_indices_raw = [0, 1, 2]

    return output, camera_indices_raw


def get_traj_paths(input_paths):
    # concatenate all paths
    real_traj_paths = []
    sim_traj_paths = []
    num_traj_available = 0
    for path in input_paths:
        if "sim" in path:
            sim_traj_paths.append(
                [
                    os.path.join(path, traj_name)
                    for traj_name in os.listdir(path)
                    # if traj_name.endswith(".h5") and "failed" not in traj_name
                    if traj_name.endswith(".h5")
                ]
            )
            num_traj_available += len(sim_traj_paths[-1])
        else:
            real_traj_paths.append(
                [os.path.join(path, traj_name) for traj_name in os.listdir(path) if traj_name.endswith(".h5")]
            )
            num_traj_available += len(real_traj_paths[-1])
    return real_traj_paths, sim_traj_paths, num_traj_available


# def resize_image(args):
#     img, img_resolution = args
#     return cv2.resize(img, (img_resolution[1], img_resolution[0]))  # (W, H)


# def resize_images_multiprocessing(raw_img, img_resolution, num_thread=10):
#     args = [(raw_img[i], img_resolution) for i in range(raw_img.shape[0])]

#     # Use Pool for multiprocessing
#     with Pool(processes=num_thread) as pool:
#         resized_images = pool.map(resize_image, args)

#     resized_img = np.array(resized_images, dtype=np.uint8)
#     return resized_img


def sample_paths_rand(paths, num_samples, ratio=None):
    """Helper function to sample paths based on either count or ratio."""
    sampled_paths = []
    if num_samples is None and ratio is None:
        sampled_paths = [path for sublist in sampled_paths for path in sublist]
        return sampled_paths

    if len(num_samples) > 1:
        if len(num_samples) == len(paths) - 1:
            print("Automatically merging last two data paths then sample")
            paths[-2].extend(paths[-1])
            paths.pop(-1)
        sampled_paths = [list(np.random.choice(paths[i], num_samples[i], replace=False)) for i in range(len(paths))]
        sampled_paths = [path for sublist in sampled_paths for path in sublist]  # Flatten
    elif len(num_samples) == 1:
        paths = [path for sublist in paths for path in sublist]  # Flatten
        sampled_paths = list(np.random.choice(paths, num_samples[0], replace=False))
    elif ratio is not None:
        if len(ratio) > 1:
            assert len(ratio) == len(paths), "Mismatched list lengths"
            sampled_paths = [
                list(np.random.choice(paths[i], int(len(paths[i]) * ratio[i]), replace=False))
                for i in range(len(paths))
            ]
            sampled_paths = [path for sublist in sampled_paths for path in sublist]  # Flatten
        else:
            paths = [path for sublist in paths for path in sublist]  # Flatten
            sampled_paths = list(np.random.choice(paths, int(len(paths) * ratio[0]), replace=False))
    else:
        sampled_paths = paths  # No change if neither num_samples nor ratio is provided

    return sampled_paths


def sample_paths(traj_paths, num_trajs, ratio=None, sample_strat="uniform", num_base=80, num_delta=20):
    if sample_strat == "uniform":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sorted(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]

            # Define variation indices based on the variation type
            if "lighting" in sorted_paths[0]:
                idx_per_variation = [
                    0,
                    16,
                    30,
                    44,
                    58,
                    72,
                    86,
                    100,
                    116,
                    130,
                    144,
                    158,
                    172,
                    186,
                ]
            elif any(k in sorted_paths[0] for k in ["camera_pose", "distractor", "background"]):
                if "camera_pose_new_new" in sorted_paths[0]:
                    idx_per_variation = [0, 20, 40, 60, 80]
                else:
                    idx_per_variation = [0, 25, 50, 75, 100, 125, 150, 175]
            elif "table_texture" in sorted_paths[0]:
                idx_per_variation = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
            else:
                raise ValueError(f"Unknown variation type in {sorted_paths[0]}")

            paths = []
            for j in range(num_samples):
                variation_idx = j % len(idx_per_variation)
                sample_idx = j // len(idx_per_variation)

                index = idx_per_variation[variation_idx] + sample_idx
                if index < len(sorted_paths):  # Ensure the index is valid
                    paths.append(sorted_paths[index])
                else:
                    print(f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping.")

            output_paths += paths

    elif sample_strat == "rand":
        output_paths = sample_paths_rand(traj_paths, num_trajs, ratio)

    elif sample_strat == "strat2":
        num_samples = num_trajs[1]  # 40
        sorted_paths = sorted(traj_paths[1])
        indices = np.linspace(0, len(sorted_paths) - 1, num_samples, dtype=int)
        sampled_paths = [sorted_paths[i] for i in indices]
        output_paths = [path for path in traj_paths[0]] + sampled_paths

    elif sample_strat == "val":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sorted(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            # Define variation indices based on the variation type
            if "lighting" in sorted_paths[0]:
                idx_per_variation = [
                    0,
                    16,
                    30,
                    44,
                    58,
                    72,
                    86,
                    100,
                    116,
                    130,
                    144,
                    158,
                    172,
                    186,
                ]
            elif any(k in sorted_paths[0] for k in ["camera_pose", "distractor", "background"]):
                idx_per_variation = [0, 25, 50, 75, 100, 125, 150, 175]
            elif "table_texture" in sorted_paths[0]:
                idx_per_variation = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
            else:
                raise ValueError(f"Unknown variation type in {sorted_paths[0]}")

            paths = []
            for j in range(num_samples):
                variation_idx = j % len(idx_per_variation)
                sample_idx = j // len(idx_per_variation)

                delta = idx_per_variation[1] - idx_per_variation[0] - 1

                index = idx_per_variation[variation_idx] + delta - sample_idx
                if index < len(sorted_paths):  # Ensure the index is valid
                    paths.append(sorted_paths[index])
                else:
                    print(f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping.")

            output_paths += paths
    elif sample_strat == "cluster_pick":
        # Cluster pick
        sim_traj_paths = [
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/51_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/12_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/38_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/39_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/61_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/29_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/52_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/13_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/60_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/56_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/11_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/48_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/13_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/28_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/85_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/17_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/36_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/9_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/82_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/67_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/18_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/47_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/24_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/48_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/28_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/22_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/14_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/18_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/24_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/48_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/25_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/10_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/67_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/15_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/62_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/45_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/38_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/77_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/10_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/28_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/39_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/16_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/32_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/90_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/43_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/22_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/48_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/67_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/34_sim.h5",
            "/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/17_sim.h5",
        ]
        return sim_traj_paths[: num_trajs[0]]
        # Cluster pick and place
        # sim_traj_paths = ['/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/42_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/21_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/62_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/61_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/32_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/18_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/90_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/48_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/23_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/53_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/17_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/12_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/37_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/28_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/23_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/14_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/28_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/10_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/19_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/47_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/67_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/46_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/76_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/39_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/63_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/10_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/67_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/34_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/37_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/18_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/54_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/50_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/36_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/47_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/57_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/9_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_trials-date/66_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/44_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/13_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/56_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/25_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/46_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/15_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_dec1/13_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/9_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/51_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/35_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/38_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan24/29_sim.h5', '/n/fs/robot-data/guided-data-collection/data/sim_new_intr/tomato_plate_jan12/30_sim.h5']
    elif sample_strat == "sim_base":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sorted(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            idx_per_variation = {0: 0, 30: 0, 60: 0, 90: 0, 120: 0}
            variation_keys = list(idx_per_variation.keys())
            paths = []
            for j in range(num_samples):
                variation_idx = j % len(idx_per_variation)
                sample_idx = idx_per_variation[variation_keys[variation_idx]]
                index = variation_keys[variation_idx] + sample_idx
                if index < len(sorted_paths):  # Ensure the index is valid
                    if "failed" not in sorted_paths[index]:
                        paths.append(sorted_paths[index])
                        idx_per_variation[variation_keys[variation_idx]] += 1
                    else:
                        while "failed" in sorted_paths[index]:
                            index += 1
                            if index >= variation_keys[variation_idx] + 30:
                                raise ValueError(f"Too many failed paths for variation {variation_idx}. {sorted_paths}")
                        paths.append(sorted_paths[index])
                        idx_per_variation[variation_keys[variation_idx]] = index - variation_keys[variation_idx] + 1
                else:
                    raise ValueError(f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping.")
            output_paths += paths
    elif sample_strat == "sim_factor":
        output_paths = []
        num_factors = len(traj_paths)
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sorted(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            if i < num_factors - 1:
                idx_per_variation = {
                    0: 0,
                    30: 0,
                    60: 0,
                    90: 0,
                    120: 0,
                    150: 0,
                    180: 0,
                    210: 0,
                    240: 0,
                    270: 0,
                }
                variation_keys = list(idx_per_variation.keys())
                paths = []
                for j in range(num_samples):
                    variation_idx = j % len(idx_per_variation)
                    sample_idx = idx_per_variation[variation_keys[variation_idx]]
                    index = variation_keys[variation_idx] + sample_idx
                    if index < len(sorted_paths):  # Ensure the index is valid
                        if "failed" not in sorted_paths[index]:
                            paths.append(sorted_paths[index])
                            idx_per_variation[variation_keys[variation_idx]] += 1
                        else:
                            while "failed" in sorted_paths[index]:
                                index += 1
                                if index >= variation_keys[variation_idx] + 30:
                                    raise ValueError(
                                        f"Too many failed paths for variation {variation_idx}. {sorted_paths}"
                                    )
                            paths.append(sorted_paths[index])
                            idx_per_variation[variation_keys[variation_idx]] = index - variation_keys[variation_idx] + 1
                    else:
                        raise ValueError(
                            f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping."
                        )
            else:
                paths = []
                idx_per_variation_base = {0: 0, 30: 0, 60: 0}
                idx_per_variation_extra = {90: 0, 120: 0, 150: 0}
                for j in range(num_samples):
                    if j < num_base:
                        idx_per_variation = idx_per_variation_base
                    else:
                        idx_per_variation = idx_per_variation_extra
                    variation_keys = list(idx_per_variation.keys())
                    variation_idx = j % len(idx_per_variation)
                    sample_idx = idx_per_variation[variation_keys[variation_idx]]
                    index = variation_keys[variation_idx] + sample_idx
                    if index < len(sorted_paths):  # Ensure the index is valid
                        if "failed" not in sorted_paths[index]:
                            paths.append(sorted_paths[index])
                            idx_per_variation[variation_keys[variation_idx]] += 1
                        else:
                            while "failed" in sorted_paths[index]:
                                index += 1
                                if index >= variation_keys[variation_idx] + 30:
                                    raise ValueError(
                                        f"Too many failed paths for variation {variation_idx}. {sorted_paths}"
                                    )
                            paths.append(sorted_paths[index])
                            idx_per_variation[variation_keys[variation_idx]] = index - variation_keys[variation_idx] + 1
                    else:
                        raise ValueError(
                            f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping."
                        )
            output_paths += paths
    elif sample_strat == "sim_per_factor" or sample_strat == "real_per_factor":
        output_paths = []
        for i, traj_path in enumerate(traj_paths):
            sorted_paths = sorted(traj_path)  # Ensure ordering before sampling
            num_samples = num_trajs[i]
            output_paths += sorted_paths[:num_samples]
    # elif sample_strat == "real_per_factor":
    #     output_paths = []
    #     for i, traj_path in enumerate(traj_paths):
    #         sorted_paths = sorted(traj_path)  # Ensure ordering before sampling
    #         if "lighting" in sorted_paths[0]:
    #             idx_per_variation = {
    #                 0: 0,
    #                 16: 0,
    #                 30: 0,
    #                 44: 0,
    #                 58: 0,
    #                 72: 0,
    #                 86: 0,
    #                 100: 0,
    #                 116: 0,
    #                 130: 0,
    #                 144: 0,
    #                 158: 0,
    #                 172: 0,
    #                 186: 0
    #             }
    #         elif any(
    #             k in sorted_paths[0]
    #             for k in ["camera_pose", "distractor", "background"]
    #         ):
    #             idx_per_variation = {0:0, 25:0, 50:0, 75:0, 100:0, 125:0, 150:0, 175:0}
    #         elif "table_texture" in sorted_paths[0]:
    #             idx_per_variation = {0:0, 20:0, 40:0, 60:0, 80:0, 100:0, 120:0, 140:0, 160:0, 180:0}
    #         else:
    #             raise ValueError(f"Unknow factor {sorted_paths[0]}")
    #         num_samples = num_trajs[i]
    #         variation_keys = list(idx_per_variation.keys())
    #         paths = []
    #         for j in range(num_samples):
    #             variation_idx = j // num_delta
    #             sample_idx = idx_per_variation[variation_keys[variation_idx]]
    #             if sample_idx >= variation_keys[variation_idx]:
    #                 variation_idx += len(idx_per_variation) // 2
    #                 sample_idx = idx_per_variation[variation_keys[variation_idx]]
    #             index = variation_keys[variation_idx] + sample_idx
    #             if index < len(sorted_paths):  # Ensure the index is valid
    #                 paths.append(sorted_paths[index])
    #                 idx_per_variation[variation_keys[variation_idx]] += 1
    #             else:
    #                 raise ValueError(
    #                     f"Warning: Index {index} out of bounds for variation {variation_idx}, skipping."
    #                 )
    #         output_paths += paths
    else:
        raise NotImplementedError(f"Unknown sampling strategy: {sample_strat}")
    return output_paths


def process_pi_dataset(
    input_paths,
    action_keys=["joint_position", "gripper_position"],
    observation_keys=["joint_positions", "gripper_position"],
    horizon_steps=16,
    img_resolution=(192, 192),
    camera_indices=[0, 2],
    num_thread=10,
    skip_image=False,
    keep_bgr=False,
    use_obs_as_action=False,
    obs_gripper_threshold=0.2,
    additional_name="",
    num_real_traj=None,
    num_sim_traj=None,
    sim_data_ratio=None,
    real_data_ratio=None,
    seed=None,
    visualize_image=False,
    sample_real_strat="strat1",
    keep_only_first=False,
    sample_sim_strat="rand",
    num_base=80,
    push_to_hub=False,
):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(42)
    save_image = not skip_image
    bgr2rgb = not keep_bgr

    # Ensure only one of num_sim_traj or sim_data_ratio is specified
    assert not (
        num_sim_traj is not None and sim_data_ratio is not None
    ), "Only one of sim_data_ratio and num_sim_traj should be specified"
    assert not (
        num_real_traj is not None and real_data_ratio is not None
    ), "Only one of real_data_ratio and num_real_traj should be specified"

    real_traj_paths, sim_traj_paths, num_traj_available = get_traj_paths(input_paths)

    sim_traj_paths = sample_paths(
        sim_traj_paths,
        num_sim_traj,
        sim_data_ratio,
        sample_sim_strat,
        num_base=num_base,
    )
    real_traj_paths = sample_paths(real_traj_paths, num_real_traj, real_data_ratio, sample_real_strat)

    num_real_traj = len(real_traj_paths)
    num_sim_traj = len(sim_traj_paths)

    traj_paths = list(real_traj_paths) + list(sim_traj_paths)
    num_traj = len(traj_paths)

    print(f"Processing {num_traj}/{num_traj_available} trajectories with {cpu_count()} cpu threads...")

    # Configure dataset name based on keys
    dataset_name = ""
    # if "cartesian_position" in observation_keys:
    #     dataset_name += "eef"
    # if "joint_positions" in observation_keys:
    #     assert "cartesian_position" not in observation_keys
    #     dataset_name += "js"
    # if "joint_velocities" in observation_keys:
    #     assert "cartesian_position" not in observation_keys
    #     dataset_name += "jv"
    # if "gripper_position" in observation_keys:
    #     dataset_name += "g"
    # dataset_name += "_"
    # if "cartesian_position" in action_keys:
    #     dataset_name += "eef"
    # if "joint_position" in action_keys:
    #     assert "cartesian_position" not in action_keys
    #     dataset_name += "js"
    # if "joint_velocities" in action_keys:
    #     assert "cartesian_position" not in action_keys
    #     dataset_name += "jv"
    # if "gripper_position" in action_keys:
    #     dataset_name += "g"
    # if save_image:
    #     dataset_name += "_"
    #     dataset_name += f"{len(camera_indices)}cam"
    #     dataset_name += f"_{img_resolution[0]}"
    if additional_name:
        dataset_name += f"{additional_name}"
    if seed is not None:
        dataset_name += f"_seed{seed}"
    dataset_name += f"_sim{num_sim_traj}_real{num_real_traj}_pi"

    # Create output directory
    output_path = dataset_name
    # os.makedirs(output_path, exist_ok=True)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`

    dataset = LeRobotDataset.create(
        repo_id=output_path,
        robot_type="panda",
        # root="/n/fs/robot-data/cache/huggingface/lerobot",
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

    for traj_path in tqdm(traj_paths):
        s1 = time.time()
        try:
            if "sim" in traj_path:
                print("Loading sim data")
                traj, camera_indices_raw = load_sim_hdf5_for_training(
                    traj_path,
                    action_keys=action_keys,
                    observation_keys=observation_keys,
                    load_image=save_image,
                )
            else:
                traj, camera_indices_raw = load_hdf5(
                    traj_path,
                    action_keys=action_keys,
                    observation_keys=observation_keys,
                    load_image=save_image,
                )
            print("Time to load h5:", time.time() - s1)
        except Exception:
            print("Failed to load", traj_path)
            continue

        for step in range(len(traj["state"])):
            dataset.add_frame(
                {
                    "image": traj["observation/image/0"][step],
                    "wrist_image": traj["observation/image/1"][step],
                    "state": traj["state"][step],
                    "actions": traj["action"][step],
                }
            )
        dataset.save_episode(task="pick up the tomato and put it into the metal plate")

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # if visualize_image:
    #     if len(camera_indices) == 2:
    #         stack_videos_horizontally(
    #             *[
    #                 output["images"][i][::100].transpose(0, 2, 3, 1)
    #                 for i in camera_indices
    #             ],
    #             os.path.join(output_path, "images.mp4"),
    #             bgr2rgb=False,
    #         )
    #     else:
    #         assert len(camera_indices) == 1
    #         save_array_to_video(
    #             os.path.join(output_path, "images.mp4"),
    #             output["images"][camera_indices[0]][::100].transpose(0, 2, 3, 1),
    #             bgr2rgb=False,
    #         )

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-in",
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
    )
    # parser.add_argument(
    #     "-out",
    #     "--output_parent_dir",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "-n",
    #     "--num_traj",
    #     type=int,
    #     default=None,  # None for processing all available ones
    # )
    parser.add_argument(
        "-t",
        "--num_thread",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-c",
        "--camera_indices",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Raw data uses index from cv2, which might be something like [2, 4, 8]. We will use the index from the raw data, in this case, 0, 1, 2",
    )
    parser.add_argument(
        "-res",
        "--img_resolution",
        type=int,
        nargs=2,
        default=[192, 192],
    )
    parser.add_argument(
        "-a",
        "--action_keys",
        type=str,
        nargs="+",
        default=[
            "joint_position",
            "gripper_position",
        ],  # "cartesian_position"
    )
    parser.add_argument(
        "-o",
        "--observation_keys",
        type=str,
        nargs="+",
        default=[
            "joint_positions",
            "gripper_position",
        ],  # "joint_velocities", "cartesian_positions"
    )
    parser.add_argument(
        "-tp",
        "--horizon_steps",
        type=int,
        default=16,  # we are not saving action chunks, but to get the maximum difference between the state and each step of action, for delta actions
    )
    parser.add_argument(
        "--skip_image",
        action="store_true",
    )
    parser.add_argument(
        "--keep_bgr",
        action="store_true",
    )
    parser.add_argument(
        "--use_obs_as_action",
        action="store_true",
    )
    parser.add_argument(
        "--obs_gripper_threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--additional_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--sim_data_ratio",
        default=None,
    )
    parser.add_argument(
        "--real_data_ratio",
        default=None,
    )
    parser.add_argument(
        "--num_sim_traj",
        type=int,
        nargs="+",  # Accepts one or more integers
        default=None,
    )
    parser.add_argument(
        "--num_real_traj",
        type=int,
        nargs="+",  # Accepts one or more integers
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--visualize_image",
        action="store_true",
    )
    parser.add_argument(
        "--sample_real_strat",
        type=str,
        default="uniform",
        # choices=["strat1", "strat2", "rand", "val", "real_per_factor"],
    )
    parser.add_argument(
        "--keep_only_first",
        action="store_true",
    )
    parser.add_argument(
        "--sample_sim_strat",
        type=str,
        default="rand",
        # choices=["cluster_pick", "rand", "sim_base", "sim_factor"],
    )
    parser.add_argument(
        "--num_base",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    process_pi_dataset(**vars(args))
