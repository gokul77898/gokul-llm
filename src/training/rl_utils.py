"""RL training utilities"""

import torch
import numpy as np
from typing import Tuple


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, 
                last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation"""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    return advantages, returns


def normalize_rewards(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize rewards"""
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize advantages"""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance"""
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
