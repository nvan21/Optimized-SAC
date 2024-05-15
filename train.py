import torch
import gymnasium as gym
import yaml
import argparse
from utils import Logger
import time

import difflib

from sac import SAC


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="MountainCarContinuous-v0",
        help="id of gym environment",
    )
    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help="whether or not to record videos of the training",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="seed used for the random number generators"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get all of the registered gymnasium environments
    all_envs = set(gym.envs.registry)

    # If the environment is not in the registered envs, then suggest the closest match
    if args.env not in all_envs:
        try:
            closest_env = difflib.get_close_matches(args.env, all_envs, n=1)[0]
        except:
            closest_env = "no close match was found"
        raise ValueError(
            f"{args.env} is not a valid gym environment - did you mean '{closest_env}'?"
        )

    env = gym.make(args.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    with open("hyperparams.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        envs = list(hyperparams_dict.keys())
        if args.env in envs:
            hyperparams = hyperparams_dict[args.env]
        else:
            raise ValueError(
                f"No hyperparameters were found for the '{args.env}' environment"
            )

        hyperparams["device"] = device
        hyperparams["act_dim"] = env.action_space.shape[0]
        hyperparams["obs_dim"] = env.observation_space.shape[0]
        hyperparams["model_path"] = f"weights/{args.env}"

    logger = Logger()
    model = SAC(**hyperparams)

    n_timesteps = hyperparams["n_timesteps"]
    train_freq = hyperparams["train_freq"]
    state, _ = env.reset()
    start_time = time.time()
    print("Starting training")
    for step in range(n_timesteps):
        # Get a random action if the learning hasn't started yet
        if model.learning_starts <= step:
            action = env.action_space.sample()
        else:
            action = model.get_action(state)

        next_state, reward, done, truncated, info = env.step(action)
        done = int(done)
        model.buffer.push(state, action, reward, next_state, done)

        if step % train_freq == 0:
            q_loss, actor_loss = model.update_parameters()
            loss_info = {"Critic Loss": q_loss, "Actor Loss": actor_loss}
            logger.add(**loss_info)

        if "episode" in info.keys():
            episode_info = {
                "Episode Return": info["episode"]["r"],
                "Episode Length": info["episode"]["l"],
            }
            logger.add(**episode_info)

        if done or truncated:
            state, _ = env.reset()

        if step % 100 == 0:
            time_info = {
                "Total Timesteps": step,
                "Total Training Time": f"{round(time.time() - start_time, 2)} s",
            }
            logger.add(**time_info)
            logger.print()
