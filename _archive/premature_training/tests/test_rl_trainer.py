"""Tests for RL training pipeline"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.rl.env import LegalRLEnv
from src.rl.ppo import PPOAgent, PPOBuffer, PPOPolicy
from src.data import create_sample_data


@pytest.fixture(scope="module")
def temp_dir():
    """Create temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_rl_environment_creation():
    """Test RL environment creation"""
    env = LegalRLEnv(
        task_type="summarization",
        max_episode_steps=10,
        vocab_size=100,
        max_length=32
    )
    
    assert env.action_space.n == 100
    assert env.observation_space.shape == (32,)


def test_rl_environment_reset():
    """Test environment reset"""
    env = LegalRLEnv(max_episode_steps=10)
    
    obs, info = env.reset(options={'document': 'test doc', 'target': 'test target'})
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_rl_environment_step():
    """Test environment step"""
    env = LegalRLEnv(max_episode_steps=10, vocab_size=100)
    
    obs, info = env.reset(options={'document': 'test', 'target': 'test'})
    action = env.action_space.sample()
    
    next_obs, reward, done, truncated, info = env.step(action)
    
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_ppo_policy_creation():
    """Test PPO policy network creation"""
    policy = PPOPolicy(obs_dim=32, action_dim=10, hidden_dim=64, n_layers=2)
    
    assert policy is not None
    assert hasattr(policy, 'policy_net')
    assert hasattr(policy, 'value_net')


def test_ppo_policy_forward():
    """Test PPO policy forward pass"""
    policy = PPOPolicy(obs_dim=32, action_dim=10)
    obs = torch.randn(4, 32)
    
    logits, value = policy(obs)
    
    assert logits.shape == (4, 10)
    assert value.shape == (4, 1)


def test_ppo_policy_get_action():
    """Test action sampling"""
    policy = PPOPolicy(obs_dim=32, action_dim=10)
    obs = torch.randn(1, 32)
    
    action, log_prob, value = policy.get_action(obs)
    
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)


def test_ppo_buffer():
    """Test PPO buffer"""
    buffer = PPOBuffer(size=100, obs_dim=32)
    
    for i in range(10):
        obs = np.random.randn(32)
        buffer.store(obs, action=i % 5, reward=0.1, value=0.5, log_prob=-1.0, done=False)
    
    assert buffer.ptr == 10
    
    # Test advantages computation
    advantages, returns = buffer.compute_advantages(last_value=0.0)
    
    assert advantages.shape == (10,)
    assert returns.shape == (10,)


def test_ppo_agent_creation():
    """Test PPO agent creation"""
    agent = PPOAgent(
        obs_dim=32,
        action_dim=10,
        hidden_dim=64,
        learning_rate=0.001,
        device="cpu"
    )
    
    assert agent.policy is not None
    assert agent.optimizer is not None


def test_ppo_agent_predict():
    """Test PPO agent prediction"""
    agent = PPOAgent(obs_dim=32, action_dim=10, device="cpu")
    obs = np.random.randn(32)
    
    action, log_prob, value = agent.predict(obs)
    
    assert isinstance(action, (int, np.integer, np.ndarray))
    assert isinstance(log_prob, (float, np.floating, np.ndarray))
    assert isinstance(value, (float, np.floating, np.ndarray))


def test_ppo_agent_update():
    """Test PPO agent update"""
    agent = PPOAgent(obs_dim=32, action_dim=10, device="cpu")
    
    # Create buffer with some data
    buffer = PPOBuffer(size=64, obs_dim=32)
    for i in range(64):
        obs = np.random.randn(32)
        buffer.store(obs, action=i % 10, reward=0.1, value=0.5, log_prob=-1.0, done=(i % 10 == 9))
    
    # Update agent
    losses = agent.update(buffer, n_epochs=2, batch_size=32)
    
    assert 'policy_loss' in losses
    assert 'value_loss' in losses
    assert 'entropy' in losses
    assert isinstance(losses['policy_loss'], float)


def test_rl_training_rollout():
    """Test a short RL training rollout"""
    env = LegalRLEnv(max_episode_steps=5, vocab_size=50, max_length=32)
    agent = PPOAgent(obs_dim=32, action_dim=50, device="cpu")
    
    obs, info = env.reset(options={'document': 'test document', 'target': 'target'})
    total_reward = 0
    
    for step in range(5):
        action, log_prob, value = agent.predict(obs)
        action_int = action.item() if hasattr(action, 'item') else int(action)
        
        next_obs, reward, done, truncated, info = env.step(action_int)
        total_reward += reward
        
        obs = next_obs
        if done or truncated:
            break
    
    assert isinstance(total_reward, float)
    assert step >= 0
