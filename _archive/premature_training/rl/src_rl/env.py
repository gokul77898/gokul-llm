"""RL Environment wrapper for legal tasks"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LegalRLEnv(gym.Env):
    """Gymnasium environment for legal document tasks"""
    
    def __init__(
        self,
        task_type: str = "summarization",
        max_episode_steps: int = 50,
        vocab_size: int = 1000,
        max_length: int = 128,
        tokenizer=None
    ):
        """
        Args:
            task_type: Type of task (summarization, qa, classification)
            max_episode_steps: Maximum steps per episode
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            tokenizer: Tokenizer for text processing
        """
        super().__init__()
        
        self.task_type = task_type
        self.max_episode_steps = max_episode_steps
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(vocab_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=vocab_size,
            shape=(max_length,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.current_document = None
        self.current_target = None
        self.generated_sequence = []
        self.episode_reward = 0.0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.generated_sequence = []
        self.episode_reward = 0.0
        
        # Set document and target from options
        if options:
            self.current_document = options.get('document', '')
            self.current_target = options.get('target', '')
        else:
            self.current_document = ''
            self.current_target = ''
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        self.current_step += 1
        self.generated_sequence.append(action)
        
        # Check if episode is done
        done = self.current_step >= self.max_episode_steps
        truncated = False
        
        # Calculate reward
        reward = self._calculate_reward() if done else 0.0
        self.episode_reward += reward
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Simple encoding: use generated sequence as observation
        obs = np.zeros(self.max_length, dtype=np.float32)
        
        if self.generated_sequence:
            seq_len = min(len(self.generated_sequence), self.max_length)
            obs[:seq_len] = np.array(self.generated_sequence[:seq_len], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on task"""
        if not self.generated_sequence or not self.current_target:
            return 0.0
        
        # Simple reward: negative length penalty + completion bonus
        length_penalty = -0.01 * len(self.generated_sequence)
        completion_bonus = 1.0 if self.generated_sequence else 0.0
        
        return length_penalty + completion_bonus
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        return {
            'step': self.current_step,
            'sequence_length': len(self.generated_sequence),
            'episode_reward': self.episode_reward,
        }
