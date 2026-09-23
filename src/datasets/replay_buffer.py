import collections
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=100000, device="cpu"):
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append(
            (
                np.array(state, dtype=np.uint8),
                action,
                reward,
                np.array(next_state, dtype=np.uint8),
                terminated,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminated = zip(*batch)
        return (
            torch.tensor(
                np.array(states), dtype=torch.uint8, device=self.device
            ).float(),
            torch.tensor(actions, dtype=torch.int64, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(
                np.array(next_states), dtype=torch.uint8, device=self.device
            ).float(),
            torch.tensor(terminated, dtype=torch.float32, device=self.device),
        )

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
