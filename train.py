import replay
import torch
import gymnasium as gym
import yaml
import argparse

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

    with open("hyperparams.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        envs = list(hyperparams_dict.keys())
        if args.env in envs:
            hyperparams = hyperparams_dict[args.env]
        else:
            raise ValueError(
                f"No hyperparameters were found for the '{args.env}' environment"
            )

        for key, value in hyperparams_dict["config"].items():
            hyperparams[key] = value

        hyperparams["device"] = device
        hyperparams["act_dim"] = env.action_space.shape[0]
        hyperparams["obs_dim"] = env.observation_space.shape[0]

    model = SAC(**hyperparams)

    for step in model.n_timesteps:
        print(step)
