"""Rollout buffer for PPO"""

import numpy as np
import torch
from typing import Tuple


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, size: int, obs_dim: int, device: str = "cpu"):
        self.size = size
        self.obs_dim = obs_dim
        self.device = device
        
        self.observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def store(self, obs: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr = 0
    
    def compute_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        size = self.size if self.full else self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
        
        returns = advantages + self.values[:size]
        return advantages, returns
    
    def get(self) -> dict:
        size = self.size if self.full else self.ptr
        data = {
            'observations': torch.from_numpy(self.observations[:size]).to(self.device),
            'actions': torch.from_numpy(self.actions[:size]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[:size]).to(self.device),
            'values': torch.from_numpy(self.values[:size]).to(self.device),
            'log_probs': torch.from_numpy(self.log_probs[:size]).to(self.device),
            'dones': torch.from_numpy(self.dones[:size]).to(self.device),
        }
        self.ptr = 0
        self.full = False
        return data
