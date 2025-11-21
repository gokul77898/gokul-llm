"""RL Agents for Legal Document Tasks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LegalDocumentFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for legal documents.
    
    Processes document representations for RL agents.
    """
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        embedding_dim: int = 256
    ):
        super().__init__(observation_space, features_dim)
        
        # Input dimension from observation space
        input_dim = observation_space.shape[0]
        
        # Feature extraction network
        self.network = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class PPOAgent:
    """
    Proximal Policy Optimization agent for legal tasks.
    
    Uses PPO algorithm to learn optimal policies for:
    - Document summarization
    - Question answering
    - Classification
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            env: Legal task environment
            learning_rate: Learning rate
            n_steps: Steps to collect before update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            device: Device to use
        """
        # Policy kwargs with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=LegalDocumentFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
        
        # Initialize PPO agent
        self.agent = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device
        )
        
        self.device = device
    
    def train(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10
    ):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback
            log_interval: Logging interval
        """
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action, state (if any)
        """
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save agent"""
        self.agent.save(path)
        print(f"PPO agent saved to {path}")
    
    def load(self, path: str):
        """Load agent"""
        self.agent = PPO.load(path, device=self.device)
        print(f"PPO agent loaded from {path}")


class DQNAgent:
    """
    Deep Q-Network agent for legal tasks.
    
    Uses DQN algorithm for discrete action spaces.
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            env: Legal task environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Steps before learning starts
            batch_size: Batch size
            tau: Target network update rate
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            target_update_interval: Target network update interval
            exploration_fraction: Fraction of training for exploration
            exploration_initial_eps: Initial epsilon
            exploration_final_eps: Final epsilon
            device: Device to use
        """
        # Policy kwargs
        policy_kwargs = dict(
            features_extractor_class=LegalDocumentFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[256, 256]
        )
        
        # Initialize DQN agent
        self.agent = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device
        )
        
        self.device = device
    
    def train(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10
    ):
        """Train the agent"""
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action"""
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save agent"""
        self.agent.save(path)
        print(f"DQN agent saved to {path}")
    
    def load(self, path: str):
        """Load agent"""
        self.agent = DQN.load(path, device=self.device)
        print(f"DQN agent loaded from {path}")


class CustomPolicyNetwork(nn.Module):
    """
    Custom policy network for legal document tasks.
    
    Can be used for more specialized RL implementations.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256, 128]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Policy head
        self.policy_head = nn.Linear(prev_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: Action logits
            value: State value
        """
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return logits, value
    
    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from observation.
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value
        """
        logits, value = self.forward(x)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)[range(len(action)), action]
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value


class ActorCritic(nn.Module):
    """
    Actor-Critic architecture for policy gradient methods.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor):
        """Forward pass"""
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value
    
    def act(self, state: torch.Tensor, deterministic: bool = False):
        """Select action"""
        action_logits, state_value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            dist = Categorical(logits=action_logits)
            action = dist.sample()
        
        return action, state_value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate state-action pair"""
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_value, dist_entropy
