# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import cv2

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = (
        "10.249.9.17"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    )
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    env = RobotEnv(
        robot_type="panda",
        action_space="joint_position",
        gripper_action_space="position",
    )
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])
    resized_images = []
    while True:
        # instruction = input("Enter instruction: ")
        instruction = "pick up the tomato and put it into the box"


        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs["observation/image"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if (
                    actions_from_chunk_completed == 0
                    or actions_from_chunk_completed >= args.open_loop_horizon
                ):
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    # request_data = {
                    #     "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    #         curr_obs[f"{args.external_camera}_image"], 224, 224
                    #     ),
                    #     "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                    #     "observation/joint_position": curr_obs["joint_position"],
                    #     "observation/gripper_position": curr_obs["gripper_position"],
                    #     "prompt": instruction,
                    # }
                    request_data = {
                        "observation/image": image_tools.resize_with_pad(
                            curr_obs["observation/image"], 224, 224
                        ),
                        "observation/wrist_image": image_tools.resize_with_pad(
                            curr_obs["observation/wrist_image"], 224, 224
                        ),
                        "observation/state": curr_obs["observation/state"],
                        "prompt": instruction,
                        "batch_size": None,
                        "seed": 0
                    }

                    resized_images.append(image_tools.resize_with_pad(curr_obs["observation/image"], 224, 224))
                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        st = time.time()
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                        print(pred_action_chunk)
                        et = time.time()
                        print(f"Time taken for inference: {et - st}")
                    # assert pred_action_chunk.shape == (10, 8), pred_action_chunk.shape

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                # action = np.clip(action, -1, 1)
                # with prevent_keyboard_interrupt():
                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break
        
        for idx, image in enumerate(resized_images):
            # breakpoint()
            os.makedirs(f"/home/lab/ppi_images", exist_ok=True)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/home/lab/ppi_images/{idx}.png", image)
        # video = np.stack(video)
        # save_filename = "video_" + timestamp
        # ImageSequenceClip(list(video), fps=10).write_videofile(
        #     save_filename + ".mp4", codec="libx264"
        # )

        # success: str | float | None = None
        # while not isinstance(success, float):
        #     success = input(
        #         "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
        #     )
        #     if success == "y":
        #         success = 1.0
        #     elif success == "n":
        #         success = 0.0

        #     success = float(success) / 100
        #     if not (0 <= success <= 1):
        #         print(f"Success must be a number in [0, 100] but got: {success * 100}")

        # df = df.append(
        #     {
        #         "success": success,
        #         "duration": t_step,
        #         "video_filename": save_filename,
        #     },
        #     ignore_index=True,
        # )
        answer = input("Do one more eval? (enter y or n) ")
        if "n" in answer.lower():
            break
        request_data = {
                        "observation/image": image_tools.resize_with_pad(
                            curr_obs["observation/image"], 224, 224
                        ),
                        "observation/wrist_image": image_tools.resize_with_pad(
                            curr_obs["observation/wrist_image"], 224, 224
                        ),
                        "observation/state": curr_obs["observation/state"],
                        "prompt": instruction,
                        "batch_size": None,
                        "seed": 0
                    }
        policy_client.infer(request_data)
        while True:
            env.reset()
            answer = input("Correctly reset (enter y or n)? ")
            if "n" in answer.lower():
                continue
            break

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    # left_image, right_image, wrist_image = None, None, None

    # # Drop the alpha dimension
    # left_image = left_image[..., :3]
    # right_image = right_image[..., :3]
    # wrist_image = wrist_image[..., :3]

    # # Convert to RGB
    # left_image = left_image[..., ::-1]
    # right_image = right_image[..., ::-1]
    # wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    # cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if gripper_position > 0.2:
        gripper_position = 1.0
    else:
        gripper_position = 0.0

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    # if save_to_disk:
    #     combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
    #     combined_image = Image.fromarray(combined_image)
    #     combined_image.save("robot_camera_views.png")

    return {
        "observation/image": image_observations["0"],
        "observation/wrist_image": image_observations["1"],
        "observation/state": np.concatenate([joint_position, [gripper_position]]),
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    main(args)