"""Reinforcement Learning for Legal Document Tasks"""

from .environment import LegalTaskEnvironment
from .agent import PPOAgent, DQNAgent
from .rewards import RewardCalculator
from .trainer import RLTrainer

__all__ = [
    "LegalTaskEnvironment",
    "PPOAgent",
    "DQNAgent",
    "RewardCalculator",
    "RLTrainer"
]
