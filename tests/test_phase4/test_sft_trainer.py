"""Tests for SFT trainer"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from src.rl.policy import PolicyNetwork, ValueNetwork, ActorCritic


def test_policy_network():
    policy = PolicyNetwork(obs_dim=32, action_dim=10, hidden_dim=16)
    obs = torch.randn(4, 32)
    logits = policy(obs)
    assert logits.shape == (4, 10)
    
    action, log_prob = policy.get_action(obs)
    assert action.shape == (4,)
    assert log_prob.shape == (4,)


def test_value_network():
    value = ValueNetwork(obs_dim=32, hidden_dim=16)
    obs = torch.randn(4, 32)
    values = value(obs)
    assert values.shape == (4,)


def test_actor_critic():
    model = ActorCritic(obs_dim=32, action_dim=10, hidden_dim=16)
    obs = torch.randn(4, 32)
    
    action, log_prob, value = model.get_action_and_value(obs)
    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = log_prob.sum() + value.sum()
    loss.backward()
    optimizer.step()
    
    params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, model.parameters()))
    assert params_changed
