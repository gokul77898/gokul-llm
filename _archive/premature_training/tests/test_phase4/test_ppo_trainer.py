"""Tests for PPO trainer"""

import pytest
import torch
import numpy as np

from src.rl.policy import ActorCritic
from src.rl.rollout_buffer import RolloutBuffer
from src.training.rl_utils import compute_gae, normalize_advantages


def test_rollout_buffer():
    buffer = RolloutBuffer(size=10, obs_dim=8)
    
    for i in range(10):
        obs = np.random.randn(8).astype(np.float32)
        buffer.store(obs, action=i % 3, reward=0.1, value=0.5, log_prob=-1.0, done=False)
    
    assert buffer.full
    
    advantages, returns = buffer.compute_advantages(last_value=0.0)
    assert advantages.shape == (10,)
    assert returns.shape == (10,)


def test_ppo_update():
    model = ActorCritic(obs_dim=8, action_dim=4, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    obs = torch.randn(16, 8)
    actions = torch.randint(0, 4, (16,))
    advantages = torch.randn(16)
    returns = torch.randn(16)
    old_log_probs = torch.randn(16)
    
    log_probs, values, entropy = model.evaluate(obs, actions)
    
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns)
    
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, model.parameters()))
    assert params_changed


def test_gae_computation():
    rewards = np.array([1.0, 0.5, 0.3, 0.1])
    values = np.array([0.5, 0.4, 0.3, 0.2])
    dones = np.array([0.0, 0.0, 0.0, 1.0])
    
    advantages, returns = compute_gae(rewards, values, dones, last_value=0.0)
    
    assert advantages.shape == (4,)
    assert returns.shape == (4,)
    assert np.all(np.isfinite(advantages))
    assert np.all(np.isfinite(returns))
