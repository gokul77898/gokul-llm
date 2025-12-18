"""Policy and Value Networks for RL"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple


class PolicyNetwork(nn.Module):
    """Policy network for discrete actions"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Value network for state value estimation"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined actor-critic network"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim, num_layers)
        self.value = ValueNetwork(obs_dim, hidden_dim, num_layers)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        value = self.value(obs)
        return logits, value
    
    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob = self.policy.get_action(obs, deterministic)
        value = self.value(obs)
        return action, log_prob, value
    
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropy = self.policy.evaluate_actions(obs, actions)
        values = self.value(obs)
        return log_probs, values, entropy
