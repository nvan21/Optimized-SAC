from replay import UniformReplayMemory, PrioritizedReplayMemory
from network import QValueNetwork, GaussianPolicy

import torch
import torch.nn.functional as F
from torch.optim import Adam


class SAC:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.0003)
        self.buffer_size = kwargs.get("buffer_size", 1000000)
        self.batch_size = kwargs.get("batch_size", 512)
        self.ent_coeff = kwargs.get("ent_coeff", 0.1)
        self.train_freq = kwargs.get("train_freq", 32)
        self.gradient_steps = kwargs.get("gradient_steps", 32)
        self.gamma = kwargs.get("gamma", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.learning_starts = kwargs.get("learning_starts", 0)
        self.network_arch = kwargs.get("network_arch", [256, 256])
        self.use_prioritized_buffer = kwargs.get("use_prioritized_buffer", True)
        self.device = kwargs.get("device", torch.device("cpu"))
        self.act_dim = kwargs.get("act_dim", 1)
        self.obs_dim = kwargs.get("obs_dim", 1)
        self.model_path = kwargs.get("model_path", "weights/MountainCarContinuous-v0")

        if self.use_prioritized_buffer:
            self.buffer = PrioritizedReplayMemory(capacity=self.buffer_size)
        else:
            self.buffer = UniformReplayMemory(
                capacity=self.buffer_size,
                device=self.device,
                act_dim=self.act_dim,
                obs_dim=self.obs_dim,
            )

        self.actor = GaussianPolicy(
            act_dim=1, obs_dim=self.obs_dim, hidden_layers=self.network_arch
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = QValueNetwork(
            act_dim=self.act_dim, obs_dim=self.obs_dim, hidden_layers=self.network_arch
        ).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.target_critic = QValueNetwork(
            act_dim=self.act_dim, obs_dim=self.obs_dim, hidden_layers=self.network_arch
        ).to(self.device)

        self.hard_network_update()

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        action, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()

    def update_parameters(self):
        for update in range(self.gradient_steps):
            b_states, b_actions, b_rewards, b_next_states, b_dones = self.buffer.sample(
                self.batch_size
            )

            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(b_next_states)
                q1_targ, q2_targ = self.target_critic(b_next_states, next_actions)
                q_targs = b_rewards + self.gamma * (1 - b_dones) * (
                    torch.min(q1_targ, q2_targ) - self.ent_coeff * next_log_probs
                )

            q1, q2 = self.critic(b_states, b_actions)
            q1_loss = F.mse_loss(q1, q_targs)
            q2_loss = F.mse_loss(q2, q_targs)
            q_loss = q1_loss + q2_loss

            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

            new_actions, new_log_probs = self.actor.sample(b_states)
            q1, q2 = self.critic(b_states, new_actions)
            actor_loss = (self.ent_coeff * new_log_probs - torch.min(q1, q2)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.soft_network_update()

        return q_loss, actor_loss

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
        q_target_params = self.target_critic.parameters()
        q_params = self.critic.parameters()

        for q_param, q_target_param in zip(q_params, q_target_params):
            q_target_param.data.copy_(q_param.data)

    def soft_network_update(self):
        q_target_params = self.target_critic.parameters()
        q_params = self.critic.parameters()

        for q_param, q_target_param in zip(q_params, q_target_params):
            q_target_param.data.copy_(
                q_target_param.data * self.tau + (1 - self.tau) * q_param.data
            )
