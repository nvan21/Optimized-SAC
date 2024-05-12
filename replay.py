import torch
import numpy as np

class UniformReplayMemory:
    def __init__(self, capacity, device):
        self.states = torch.zeros(capacity, 1).to(device)
        self.actions = torch.zeros(capacity, 1).to(device)
        self.rewards = torch.zeros(capacity, 1).to(device)
        self.next_states = torch.zeros(capacity, 1).to(device)
        self.dones = torch.zeros(capacity, 1).to(device)

        self.capacity = capacity
        self.position = 0
        self.is_full = False

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.is_full = self.is_full or self.position == 0

    def sample(self, batch_size):
        end_idx = self.capacity if self.is_full else self.position
        b_idxs = np.random.randint(0, end_idx, size=batch_size)

        b_states = self.states[b_idxs]
        b_actions = self.actions[b_idxs]
        b_rewards = self.rewards[b_idxs]
        b_next_states = self.next_states[b_idxs]
        b_dones = self.dones[b_idxs]

        return b_states, b_actions, b_rewards, b_next_states, b_dones