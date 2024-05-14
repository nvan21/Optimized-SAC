import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init(layer: nn.Linear):
    nn.init.xavier_uniform_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 0)

    return layer


class GaussianPolicy(nn.Module):
    def __init__(self, act_dim, obs_dim, hidden_layers, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.input_layers = [weights_init(nn.Linear(obs_dim, hidden_dim))]

        for _ in range(hidden_layers):
            self.input_layers.append(weights_init(nn.Linear(hidden_dim, hidden_dim)))

        self.input_layers = nn.ModuleList(self.input_layers)
        self.mean_layer = weights_init(nn.Linear(hidden_dim, act_dim))
        self.log_std_layer = weights_init(nn.Linear(hidden_dim, act_dim))

    def forward(self, x):
        for layer in self.input_layers:
            x = F.relu(layer(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)

        normal_distribution = Normal(mean, log_std)
        action = normal_distribution.rsample()
        action = F.tanh(action)
        log_prob = normal_distribution.log_prob(action)

        return action, log_prob


class QValueNetwork(nn.Module):
    def __init__(self, act_dim, obs_dim, hidden_layers, hidden_dim):
        super(QValueNetwork, self).__init__()

        self.q1_layers = [weights_init(nn.Linear(act_dim + obs_dim, hidden_dim))]
        for _ in range(hidden_layers):
            self.q1_layers.append(weights_init(nn.Linear(hidden_dim, hidden_dim)))

        self.q1_layers.append(weights_init(nn.Linear(hidden_dim, 1)))
        self.q1_layers = nn.ModuleList(self.q1_layers)
        self.q2_layers = self.q1_layers

    def forward(self, state, action):
        x1 = x2 = torch.cat([state, action], 1)
        for q1_layer, q2_layer in zip(self.q1_layers[:-1], self.q2_layers[:-1]):
            x1 = F.relu(q1_layer(x1))
            x2 = F.relu(q2_layer(x2))

        q1_value = self.q1_layers[-1](x1)
        q2_value = self.q2_layers[-1](x2)

        return q1_value, q2_value
