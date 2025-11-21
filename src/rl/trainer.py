"""RL Trainer for Legal Document Tasks"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from .environment import LegalTaskEnvironment, BatchLegalEnvironment
from .agent import PPOAgent, DQNAgent
from .rewards import RewardCalculator


class RLTrainer:
    """
    Trainer for Reinforcement Learning agents on legal tasks.
    
    Supports:
    - PPO and DQN training
    - Custom reward functions
    - Evaluation and monitoring
    - Checkpoint management
    """
    
    def __init__(
        self,
        agent,
        env: LegalTaskEnvironment,
        reward_calculator: Optional[RewardCalculator] = None,
        checkpoint_dir: str = "./checkpoints/rl",
        log_dir: str = "./logs/rl",
        eval_freq: int = 10000,
        save_freq: int = 50000
    ):
        """
        Args:
            agent: RL agent (PPO or DQN)
            env: Training environment
            reward_calculator: Custom reward calculator
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
            eval_freq: Evaluation frequency (timesteps)
            save_freq: Save frequency (timesteps)
        """
        self.agent = agent
        self.env = env
        self.reward_calculator = reward_calculator or RewardCalculator()
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        
        # Training metrics
        self.training_stats = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'eval_timesteps': []
        }
    
    def train(
        self,
        total_timesteps: int,
        eval_episodes: int = 10,
        callback: Optional[Callable] = None
    ):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_episodes: Number of episodes for evaluation
            callback: Optional callback function
        """
        print(f"Starting RL training for {total_timesteps} timesteps")
        print(f"Environment: {self.env.task_type.value}")
        print(f"Agent: {type(self.agent).__name__}")
        
        # Training callback
        def training_callback(locals_dict, globals_dict):
            timesteps = locals_dict.get('self').num_timesteps
            
            # Periodic evaluation
            if timesteps % self.eval_freq == 0 and timesteps > 0:
                eval_reward = self.evaluate(num_episodes=eval_episodes)
                self.training_stats['eval_rewards'].append(eval_reward)
                self.training_stats['eval_timesteps'].append(timesteps)
                print(f"\nEvaluation at {timesteps} steps: Reward = {eval_reward:.2f}")
            
            # Periodic saving
            if timesteps % self.save_freq == 0 and timesteps > 0:
                self.save_checkpoint(f"checkpoint_{timesteps}")
            
            # Custom callback
            if callback is not None:
                callback(locals_dict, globals_dict)
            
            return True
        
        # Train agent
        self.agent.train(
            total_timesteps=total_timesteps,
            callback=training_callback
        )
        
        # Final evaluation
        final_reward = self.evaluate(num_episodes=eval_episodes)
        print(f"\nFinal evaluation reward: {final_reward:.2f}")
        
        # Save final model
        self.save_checkpoint("final_model")
        
        # Save training stats
        self._save_training_stats()
        
        # Plot training curves
        self.plot_training_curves()
        
        print("\nTraining completed!")
    
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True
    ) -> float:
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Average episode reward
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards)
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save agent
        self.agent.save(str(checkpoint_path / "agent"))
        
        # Save training stats
        with open(checkpoint_path / "training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, name: str):
        """Load training checkpoint"""
        checkpoint_path = self.checkpoint_dir / name
        
        # Load agent
        self.agent.load(str(checkpoint_path / "agent"))
        
        # Load training stats
        with open(checkpoint_path / "training_stats.json", 'r') as f:
            self.training_stats = json.load(f)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def _save_training_stats(self):
        """Save training statistics"""
        stats_path = self.log_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves.
        
        Args:
            save_path: Path to save plot (if None, uses log_dir)
        """
        if save_path is None:
            save_path = self.log_dir / "training_curves.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot evaluation rewards
        if self.training_stats['eval_rewards']:
            axes[0].plot(
                self.training_stats['eval_timesteps'],
                self.training_stats['eval_rewards'],
                marker='o'
            )
            axes[0].set_xlabel('Timesteps')
            axes[0].set_ylabel('Average Reward')
            axes[0].set_title('Evaluation Reward over Time')
            axes[0].grid(True)
        
        # Plot episode lengths (if available)
        if self.training_stats['episode_lengths']:
            axes[1].plot(
                self.training_stats['timesteps'],
                self.training_stats['episode_lengths']
            )
            axes[1].set_xlabel('Timesteps')
            axes[1].set_ylabel('Episode Length')
            axes[1].set_title('Episode Length over Time')
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Training curves saved to {save_path}")


class BatchRLTrainer:
    """
    Trainer for batch RL training with multiple environments.
    """
    
    def __init__(
        self,
        agent,
        batch_env: BatchLegalEnvironment,
        reward_calculator: Optional[RewardCalculator] = None,
        checkpoint_dir: str = "./checkpoints/rl_batch",
        log_dir: str = "./logs/rl_batch"
    ):
        """
        Args:
            agent: RL agent
            batch_env: Batch environment
            reward_calculator: Reward calculator
            checkpoint_dir: Checkpoint directory
            log_dir: Log directory
        """
        self.agent = agent
        self.batch_env = batch_env
        self.reward_calculator = reward_calculator or RewardCalculator()
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, total_timesteps: int):
        """Train with batch environment"""
        print(f"Starting batch RL training with {self.batch_env.num_envs} environments")
        
        # Use standard training (stable-baselines3 handles vectorized envs)
        self.agent.train(total_timesteps=total_timesteps)
        
        print("Batch training completed!")


class CurriculumRLTrainer(RLTrainer):
    """
    RL Trainer with curriculum learning.
    
    Gradually increases task difficulty during training.
    """
    
    def __init__(
        self,
        agent,
        env: LegalTaskEnvironment,
        difficulty_levels: List[Dict],
        **kwargs
    ):
        """
        Args:
            agent: RL agent
            env: Training environment
            difficulty_levels: List of difficulty configurations
            **kwargs: Additional trainer arguments
        """
        super().__init__(agent, env, **kwargs)
        self.difficulty_levels = difficulty_levels
        self.current_level = 0
    
    def train_curriculum(
        self,
        timesteps_per_level: int,
        eval_episodes: int = 10
    ):
        """
        Train with curriculum learning.
        
        Args:
            timesteps_per_level: Timesteps per difficulty level
            eval_episodes: Episodes for evaluation
        """
        print("Starting curriculum learning...")
        
        for level_idx, level_config in enumerate(self.difficulty_levels):
            print(f"\n=== Difficulty Level {level_idx + 1}/{len(self.difficulty_levels)} ===")
            print(f"Config: {level_config}")
            
            self.current_level = level_idx
            
            # Update environment with new difficulty
            # (This would require environment to support difficulty updates)
            
            # Train at this level
            self.train(
                total_timesteps=timesteps_per_level,
                eval_episodes=eval_episodes
            )
            
            # Evaluate progression
            eval_reward = self.evaluate(num_episodes=eval_episodes)
            print(f"Level {level_idx + 1} final reward: {eval_reward:.2f}")
        
        print("\nCurriculum learning completed!")


def create_rl_trainer_for_task(
    task_type: str,
    model,
    tokenizer,
    agent_type: str = "ppo",
    total_timesteps: int = 100000,
    **kwargs
) -> RLTrainer:
    """
    Factory function to create RL trainer for specific task.
    
    Args:
        task_type: Type of task ("summarization", "qa", "classification")
        model: Base model for environment
        tokenizer: Tokenizer
        agent_type: "ppo" or "dqn"
        total_timesteps: Training timesteps
        **kwargs: Additional arguments
        
    Returns:
        Configured RLTrainer
    """
    from .environment import LegalTaskType
    
    # Map task type
    task_map = {
        "summarization": LegalTaskType.SUMMARIZATION,
        "qa": LegalTaskType.QUESTION_ANSWERING,
        "classification": LegalTaskType.DOCUMENT_CLASSIFICATION,
        "ner": LegalTaskType.ENTITY_EXTRACTION
    }
    
    task = task_map.get(task_type, LegalTaskType.SUMMARIZATION)
    
    # Create environment
    env = LegalTaskEnvironment(
        model=model,
        tokenizer=tokenizer,
        task_type=task
    )
    
    # Create agent
    if agent_type.lower() == "ppo":
        agent = PPOAgent(env, **kwargs)
    elif agent_type.lower() == "dqn":
        agent = DQNAgent(env, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        env=env,
        checkpoint_dir=f"./checkpoints/rl_{task_type}_{agent_type}",
        log_dir=f"./logs/rl_{task_type}_{agent_type}"
    )
    
    print(f"RL Trainer created for {task_type} using {agent_type.upper()}")
    
    return trainer
