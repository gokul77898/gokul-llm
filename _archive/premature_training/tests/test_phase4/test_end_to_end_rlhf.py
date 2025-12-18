"""End-to-end RLHF test"""

import pytest
import subprocess
import sys
from pathlib import Path


def test_sft_checkpoint_creation():
    """Test SFT creates checkpoint"""
    checkpoint_dir = Path("checkpoints/rlhf/sft")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        [sys.executable, "-m", "src.training.rlhf_sft", "--config", "configs/rlhf_sft.yaml"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) > 0


def test_reward_model_checkpoint():
    """Test reward model creates checkpoint"""
    checkpoint_dir = Path("checkpoints/rlhf/reward")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train_reward_model", "--config", "configs/reward_model.yaml"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) > 0


def test_ppo_checkpoint():
    """Test PPO creates checkpoint"""
    checkpoint_dir = Path("checkpoints/rlhf/ppo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train_ppo", "--config", "configs/ppo_train.yaml"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) > 0
