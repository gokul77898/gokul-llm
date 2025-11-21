"""Tests for reward model"""

import pytest
import torch
import torch.nn as nn

from src.training.train_reward_model import RewardModel


def test_reward_model_forward():
    model = RewardModel(input_dim=64, hidden_dim=32, num_layers=2)
    x = torch.randn(8, 64)
    rewards = model(x)
    assert rewards.shape == (8,)
    assert isinstance(rewards, torch.Tensor)


def test_reward_model_training_step():
    model = RewardModel(input_dim=64, hidden_dim=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    x = torch.randn(8, 64)
    target = torch.randn(8)
    
    pred = model(x)
    loss = nn.functional.mse_loss(pred, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, model.parameters()))
    assert params_changed
    assert isinstance(loss.item(), float)
