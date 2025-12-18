"""RL Environment for Legal Document Tasks"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class LegalTaskType(Enum):
    """Types of legal tasks for RL"""
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "qa"
    DOCUMENT_CLASSIFICATION = "classification"
    ENTITY_EXTRACTION = "ner"


class LegalTaskEnvironment(gym.Env):
    """
    Custom Gymnasium environment for legal document tasks.
    
    The agent learns to:
    - Generate better summaries
    - Answer questions more accurately
    - Extract relevant entities
    - Classify documents correctly
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        model,
        tokenizer,
        task_type: LegalTaskType = LegalTaskType.SUMMARIZATION,
        max_steps: int = 100,
        vocab_size: int = 50000,
        max_length: int = 512
    ):
        """
        Args:
            model: Language model for generation/prediction
            tokenizer: Tokenizer for text processing
            task_type: Type of legal task
            max_steps: Maximum steps per episode
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_steps = max_steps
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Define action and observation spaces
        if task_type == LegalTaskType.SUMMARIZATION:
            # Action: next token to generate
            self.action_space = spaces.Discrete(vocab_size)
            
            # Observation: current state (token embeddings)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_length,),
                dtype=np.float32
            )
        
        elif task_type == LegalTaskType.QUESTION_ANSWERING:
            # Action: select start and end positions
            self.action_space = spaces.MultiDiscrete([max_length, max_length])
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_length,),
                dtype=np.float32
            )
        
        elif task_type == LegalTaskType.DOCUMENT_CLASSIFICATION:
            # Action: select class
            num_classes = 10  # Adjust based on your task
            self.action_space = spaces.Discrete(num_classes)
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_length,),
                dtype=np.float32
            )
        
        # Environment state
        self.current_step = 0
        self.current_document = None
        self.current_target = None
        self.generated_sequence = []
        self.done = False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.generated_sequence = []
        self.done = False
        
        # Get new document and target from options
        if options and 'document' in options:
            self.current_document = options['document']
            self.current_target = options.get('target', None)
        else:
            # Default empty state
            self.current_document = ""
            self.current_target = None
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Reward for action
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Additional information
        """
        self.current_step += 1
        
        # Process action based on task type
        if self.task_type == LegalTaskType.SUMMARIZATION:
            self.generated_sequence.append(action)
            
            # Check if generation is complete
            if action == self.tokenizer.sep_token_id or self.current_step >= self.max_steps:
                self.done = True
        
        elif self.task_type == LegalTaskType.QUESTION_ANSWERING:
            start_pos, end_pos = action
            self.generated_sequence = [start_pos, end_pos]
            self.done = True
        
        elif self.task_type == LegalTaskType.DOCUMENT_CLASSIFICATION:
            self.generated_sequence = [action]
            self.done = True
        
        # Get reward
        reward = self._calculate_reward()
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode should end
        terminated = self.done
        truncated = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.task_type == LegalTaskType.SUMMARIZATION:
            # Encode current generated sequence
            if len(self.generated_sequence) > 0:
                # Get embeddings for generated tokens
                tokens = torch.tensor([self.generated_sequence[-1]])
                with torch.no_grad():
                    # Simple observation: one-hot or embedding
                    obs = np.zeros(self.max_length, dtype=np.float32)
                    obs[min(len(self.generated_sequence), self.max_length - 1)] = 1.0
                return obs
            else:
                return np.zeros(self.max_length, dtype=np.float32)
        
        else:
            # For other tasks, return document encoding
            if self.current_document:
                encoded = self.tokenizer.encode(
                    self.current_document,
                    padding=True,
                    truncation=True,
                    return_tensors=True
                )
                # Use token IDs as observation (simplified)
                obs = encoded.input_ids[0].float().numpy()
                # Pad or truncate to max_length
                if len(obs) < self.max_length:
                    obs = np.pad(obs, (0, self.max_length - len(obs)))
                else:
                    obs = obs[:self.max_length]
                return obs
            else:
                return np.zeros(self.max_length, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        if not self.done or self.current_target is None:
            return 0.0
        
        reward = 0.0
        
        if self.task_type == LegalTaskType.SUMMARIZATION:
            # Reward based on summary quality
            generated_text = self.tokenizer.decode(
                self.generated_sequence,
                skip_special_tokens=True
            )
            
            # Simple reward: negative length penalty + quality bonus
            length_penalty = -len(self.generated_sequence) * 0.01
            
            # Check if similar to target (simplified)
            if self.current_target:
                # Simple overlap metric
                gen_words = set(generated_text.lower().split())
                target_words = set(self.current_target.lower().split())
                overlap = len(gen_words & target_words) / max(len(target_words), 1)
                reward = overlap * 10.0 + length_penalty
            else:
                reward = length_penalty
        
        elif self.task_type == LegalTaskType.QUESTION_ANSWERING:
            # Reward based on answer correctness
            if isinstance(self.current_target, (list, tuple)) and len(self.generated_sequence) == 2:
                target_start, target_end = self.current_target
                pred_start, pred_end = self.generated_sequence
                
                # F1-based reward
                if pred_start <= target_end and pred_end >= target_start:
                    # Some overlap
                    overlap_start = max(pred_start, target_start)
                    overlap_end = min(pred_end, target_end)
                    overlap_len = overlap_end - overlap_start + 1
                    pred_len = pred_end - pred_start + 1
                    target_len = target_end - target_start + 1
                    
                    precision = overlap_len / pred_len if pred_len > 0 else 0
                    recall = overlap_len / target_len if target_len > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    reward = f1 * 10.0
                else:
                    reward = 0.0
        
        elif self.task_type == LegalTaskType.DOCUMENT_CLASSIFICATION:
            # Reward based on classification accuracy
            if len(self.generated_sequence) > 0 and self.current_target is not None:
                predicted_class = self.generated_sequence[0]
                correct = (predicted_class == self.current_target)
                reward = 10.0 if correct else -1.0
        
        return float(reward)
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'step': self.current_step,
            'task_type': self.task_type.value,
            'generated_length': len(self.generated_sequence),
            'done': self.done
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Task: {self.task_type.value}")
            print(f"Generated: {self.generated_sequence[:10]}...")
            print(f"Done: {self.done}")
    
    def close(self):
        """Clean up environment"""
        pass


class BatchLegalEnvironment:
    """
    Wrapper for batched environment execution.
    
    Allows parallel execution of multiple environments.
    """
    
    def __init__(
        self,
        num_envs: int,
        model,
        tokenizer,
        task_type: LegalTaskType = LegalTaskType.SUMMARIZATION,
        **env_kwargs
    ):
        """
        Args:
            num_envs: Number of parallel environments
            model: Shared model
            tokenizer: Shared tokenizer
            task_type: Type of task
            **env_kwargs: Additional environment arguments
        """
        self.num_envs = num_envs
        self.envs = [
            LegalTaskEnvironment(
                model=model,
                tokenizer=tokenizer,
                task_type=task_type,
                **env_kwargs
            )
            for _ in range(num_envs)
        ]
    
    def reset(self, options_list: Optional[List[Dict]] = None):
        """Reset all environments"""
        if options_list is None:
            options_list = [None] * self.num_envs
        
        observations = []
        infos = []
        
        for env, options in zip(self.envs, options_list):
            obs, info = env.reset(options=options)
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations), infos
    
    def step(self, actions: List[int]):
        """Step all environments"""
        results = []
        
        for env, action in zip(self.envs, actions):
            result = env.step(action)
            results.append(result)
        
        # Unpack results
        observations = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminated = np.array([r[2] for r in results])
        truncated = np.array([r[3] for r in results])
        infos = [r[4] for r in results]
        
        return observations, rewards, terminated, truncated, infos
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
