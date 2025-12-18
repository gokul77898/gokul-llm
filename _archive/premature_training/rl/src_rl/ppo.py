"""PPO implementation for RL training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PPOPolicy(nn.Module):
    """PPO policy network"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        # Build policy network
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.policy_net = nn.Sequential(*layers)
        
        # Value network
        layers_v = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers_v.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers_v.append(nn.Linear(hidden_dim, 1))
        
        self.value_net = nn.Sequential(*layers_v)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        logits = self.policy_net(obs)
        value = self.value_net(obs)
        return logits, value
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        logits, value = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, value.squeeze(-1), entropy


class PPOBuffer:
    """Rollout buffer for PPO"""
    
    def __init__(self, size: int, obs_dim: int):
        self.size = size
        self.obs_dim = obs_dim
        
        self.observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
    
    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition"""
        assert self.ptr < self.size
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all data"""
        assert self.ptr == self.size
        
        data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones,
        }
        
        self.ptr = 0
        return data
    
    def compute_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE"""
        advantages = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
        
        returns = advantages + self.values[:self.ptr]
        return advantages, returns


class PPOAgent:
    """PPO agent"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """Initialize PPO agent"""
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Create policy
        self.policy = PPOPolicy(obs_dim, action_dim, hidden_dim, n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
    
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict action"""
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def update(
        self,
        buffer: PPOBuffer,
        n_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Update policy"""
        data = buffer.get()
        
        # Compute advantages
        last_value = 0.0  # Terminal value
        advantages, returns = buffer.compute_advantages(last_value, self.gamma, self.gae_lambda)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs = torch.from_numpy(data['observations']).float().to(self.device)
        actions = torch.from_numpy(data['actions']).long().to(self.device)
        old_log_probs = torch.from_numpy(data['log_probs']).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)
        
        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for _ in range(n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(obs[idx], actions[idx])
                
                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratio * advantages_tensor[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_tensor[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns_tensor[idx])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
