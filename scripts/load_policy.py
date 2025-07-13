import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from openpi_client import image_tools

import cv2, os
import numpy as np

import jax.numpy as jnp

DATA_DIR = "/n/fs/robot-data/guided-data-collection/data"

class ImageDataset:
    def __init__(self, images_path):
        self.image_pairs = {}

        # Collect and group image paths by position index
        for image_name in os.listdir(images_path):
            if image_name.endswith(".png"):
                parts = image_name.replace(".png", "").split(
                    "_"
                )  # Remove extension before splitting
                if len(parts) < 3:
                    continue  # Skip malformed filenames

                try:
                    pos_index, cam_index = (
                        int(parts[1]),
                        int(parts[2]),
                    )  # Convert to integers
                except ValueError:
                    print(f"Skipping invalid file: {image_name}")
                    continue  # Skip files that don't match the expected format

                if pos_index not in self.image_pairs:
                    self.image_pairs[pos_index] = {}

                # Load image using OpenCV and convert BGR to RGB
                image = cv2.imread(os.path.join(images_path, image_name))
                if image is None:
                    print(f"Warning: Could not read {image_name}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.transpose(2, 0, 1)  # Convert to CHW format

                self.image_pairs[pos_index][cam_index] = image

        # Ensure all positions have both camera views (0 and 1)
        self.image_pairs = [
            self.image_pairs[k]
            for k in sorted(self.image_pairs.keys())
            if 0 in self.image_pairs[k] and 1 in self.image_pairs[k]
        ]

    def __getitem__(self, idx):
        return {0: self.image_pairs[idx][0][None], 1: self.image_pairs[idx][1][None]}

    def __len__(self):
        return len(self.image_pairs)

class AnomalyNominalDataset:
    def __init__(self, dataset_path):
        # for dataset_path in dataset_paths:
        #     self.load_dataset(dataset_path)
        images = self.load_dataset(dataset_path)

        self.images = images
        self.images_keys = list(images.keys())

    def load_dataset(self, dataset_path):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=True)  # only np arrays
            images = dataset["images"]
            if images.dtype == np.dtype("O"):
                images = images.item()
            else:
                raise NotImplementedError("Only support dict of images for now")
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        return images

    def __getitem__(self, idx):
        return {
            0: self.images[self.images_keys[0]][idx][None],
            1: self.images[self.images_keys[1]][idx][None],
        }

    def __len__(self):
        return len(self.images[self.images_keys[0]])

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}

def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")

def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            print(_config.get_config(args.policy.config))
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)

def create_inputs(side_image, wrist_image, obs_state=None):
    request_data = {
            "observation/image": image_tools.resize_with_pad(
                side_image, 224, 224
            ),
            "observation/wrist_image": image_tools.resize_with_pad(wrist_image, 224, 224),
            "observation/state": obs_state if obs_state is not None else np.zeros(8),
            "prompt": "pick up the tomato and put it into the metal plate",
        }
    return request_data

def cosine_similarity(a, b):
    a_norm = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
    return jnp.sum(a_norm * b_norm, axis=-1)

def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    
    # nominal_dataset = AnomalyNominalDataset(dataset_path=f"{DATA_DIR}/jsg_jsg_2cam_192_base_demo_240_uniform_new_cp_distractor_paths_seed0_sim0_real260/anomaly_dataset.npz")
    # anomaly_dataset = ImageDataset(images_path=f"{DATA_DIR}/anomaly_dataset")
    
    # num_nominal = len(nominal_dataset)
    # num_anomaly = len(anomaly_dataset)

    # nominal_features = []
    # anomaly_features = []

    # for i in range(num_anomaly):
    #     images = nominal_dataset[i]
    #     side_image = images[0][0].transpose(1, 2, 0)
    #     wrist_image = images[1][0].transpose(1, 2, 0)
    #     request_data = create_inputs(side_image, wrist_image)
    #     emb_tokens = policy.get_final_inputs(request_data)
    #     feature = 

    # breakpoint()

    request_data = {
            "observation/image": np.random.randint(256, size=(30, 224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(256, size=(30, 224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros((30, 8)),
            "prompt": "pick up the tomato and put it into the metal plate",
            "batch_size": 30,
        }
    result = policy.infer(request_data)
    breakpoint()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
