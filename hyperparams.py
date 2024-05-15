MOUNTAIN_CAR_CONTINUOUS_V0 = {
    "n_timesteps": 50000,
    "learning_rate": 3e-4,
    "buffer_size": 50000,
    "batch_size": 512,
    "ent_coeff": 0.1,
    "train_freq": 32,
    "gradient_steps": 32,
    "gamma": 0.9999,
    "tau": 0.01,
    "learning_starts": 0,
    "network_arch": [256, 256],
}

HYPERPARAMS = {"MountainCarContinuous-v0": MOUNTAIN_CAR_CONTINUOUS_V0}
