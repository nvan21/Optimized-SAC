import replay
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

test = replay.UniformReplayMemory(capacity=10, device=device)
