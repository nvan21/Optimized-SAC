# Useful Papers

- [https://arxiv.org/pdf/1801.01290] - Original SAC paper
- [https://arxiv.org/pdf/2007.01932] - Tuning the entropy temperature with metagradient
- [https://arxiv.org/pdf/2112.11115] - SAC with cross-entropy policy optimization
- [https://arxiv.org/pdf/2109.11767] - Prioritized replay buffer paper
- [https://deepganteam.medium.com/basic-policy-gradients-with-the-reparameterization-trick-24312c7dbcd] - High level explanation of the reparameterization trick
- [https://spinningup.openai.com/en/latest/algorithms/sac.html] - Spinningup Pseudocode
- [https://github.com/DLR-RM/rl-baselines3-zoo] - RL Baselines zoo (useful for tuned hyperparameters on gym environments)

# Notes

- The reparameterization trick makes the model train its parameters relative to a distribution so that the policy outputs the mean and standard deviation of a normal distribution for the sampling rather than logits
  - This makes the model more robust
  - It follows the below pseudocode:
    - Create neural network with two linear layers as outputs: one for the mean and one for the standard deviation
      - Make sure all of the weight layers are initialized with a Xavier uniformed distribution (weight of 1) and all bias layers are initialized to 0
      - Use ReLU for hidden layers and then tanh for the output layer
    - Feed the state into the neural network and return the guessed mean and standard deviation
    - Create a normal distribution using this mean and standard deviation (normal = Normal(mean, std))
    - Sample an action from the normal distribution using .rsample()
      - Could also manually do the reparameterization trick in case some environments work better with different noise
    - Apply the squashing function (tanh) to the sampled action
    - Potentially scale and bias the action, although I need to do more research into what this does
