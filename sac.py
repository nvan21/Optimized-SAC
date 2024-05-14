from replay import UniformReplayMemory
from network import QValueNetwork, GaussianPolicy
import torch


class SAC:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.buffer = UniformReplayMemory(capacity=self.buffer_size, device=self.device)

        self.actor = GaussianPolicy(
            act_dim=1,
            obs_dim=self.obs_dim,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.critic = QValueNetwork(
            act_dim=self.act_dim,
            obs_dim=self.obs_dim,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.target_critic = QValueNetwork(
            act_dim=self.act_dim,
            obs_dim=self.obs_dim,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.hard_network_update()

        self.model_path = f"./weights/{self.env}"

    def get_action(self, state):
        action, log_prob = self.actor.sample(state)

        return action, log_prob

    def update_parameters(self):
        pass

    def evaluate(self):
        pass

    def save_model(self, path=None):
        if path is None:
            path = self.model_path

        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_model(self, path=None):
        if path is None:
            path = self.model_path

        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))
        self.hard_network_update()

    def hard_network_update(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_network_update(self):
        pass
