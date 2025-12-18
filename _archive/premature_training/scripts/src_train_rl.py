"""Training script for RL model"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

from src.common import load_config, seed_everything, get_device, init_logger
from src.common.checkpoints import save_checkpoint, find_latest_checkpoint
from src.data import load_rl_dataset
from src.rl.env import LegalRLEnv
from src.rl.ppo import PPOAgent, PPOBuffer


def main():
    parser = argparse.ArgumentParser(description="Train RL model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.device:
        config.system.device = args.device
    
    # Setup
    seed_everything(config.system.seed)
    device = get_device(config.system.device)
    
    # Create directories
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    log_file = Path(config.paths.log_dir) / "train.log"
    logger = init_logger("rl_train", str(log_file))
    logger.info(f"Training RL model with config: {args.config}")
    logger.info(f"Device: {device}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset = load_rl_dataset(config)
    
    # Create environment
    logger.info("Creating environment...")
    env = LegalRLEnv(
        task_type=getattr(config, 'environment', config.model).task_type if hasattr(getattr(config, 'environment', config.model), 'task_type') else "summarization",
        max_episode_steps=getattr(config, 'environment', config.model).max_episode_steps if hasattr(getattr(config, 'environment', config.model), 'max_episode_steps') else 50,
        vocab_size=getattr(config, 'environment', config.model).vocab_size if hasattr(getattr(config, 'environment', config.model), 'vocab_size') else 1000,
        max_length=getattr(config, 'environment', config.model).max_length if hasattr(getattr(config, 'environment', config.model), 'max_length') else 128
    )
    
    # Create agent
    logger.info("Creating PPO agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.model.hidden_dim,
        n_layers=config.model.n_layers,
        learning_rate=config.training.learning_rate,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        vf_coef=config.training.vf_coef,
        ent_coef=config.training.ent_coef,
        max_grad_norm=config.training.max_grad_norm,
        device=str(device)
    )
    
    logger.info(f"Agent created with {obs_dim} obs dim and {action_dim} action dim")
    
    # Training loop
    logger.info("Starting training...")
    
    n_steps = config.training.n_steps
    buffer = PPOBuffer(n_steps, obs_dim)
    
    total_timesteps = 0
    episode_rewards = []
    current_episode_reward = 0
    
    # Reset environment
    data_idx = 0
    obs, info = env.reset(
        options={
            'document': train_dataset[data_idx]['document'],
            'target': train_dataset[data_idx].get('target', '')
        }
    )
    
    while total_timesteps < config.training.total_timesteps:
        # Collect rollouts
        for step in range(n_steps):
            # Get action from agent
            action, log_prob, value = agent.predict(obs)
            action = action.item() if hasattr(action, 'item') else int(action)
            log_prob = log_prob.item() if hasattr(log_prob, 'item') else float(log_prob)
            value = value.item() if hasattr(value, 'item') else float(value)
            
            # Take step in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store transition
            buffer.store(obs, action, reward, value, log_prob, done)
            
            current_episode_reward += reward
            total_timesteps += 1
            
            # Update observation
            obs = next_obs
            
            # Reset if done
            if done or truncated:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                
                # Get next training sample
                data_idx = (data_idx + 1) % len(train_dataset)
                obs, info = env.reset(
                    options={
                        'document': train_dataset[data_idx]['document'],
                        'target': train_dataset[data_idx].get('target', '')
                    }
                )
            
            # Break if buffer is full
            if buffer.ptr >= n_steps:
                break
        
        # Update policy
        if buffer.ptr >= n_steps:
            logger.info(f"Updating policy at timestep {total_timesteps}")
            losses = agent.update(
                buffer,
                n_epochs=config.training.n_epochs,
                batch_size=config.training.batch_size
            )
            
            # Log metrics
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            logger.info(
                f"Timestep: {total_timesteps} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Policy Loss: {losses['policy_loss']:.4f} | "
                f"Value Loss: {losses['value_loss']:.4f}"
            )
            
            # Reset buffer
            buffer = PPOBuffer(n_steps, obs_dim)
        
        # Save checkpoint periodically
        if total_timesteps % 1000 == 0:
            checkpoint_path = Path(config.paths.checkpoint_dir) / f"checkpoint_step_{total_timesteps}.pt"
            save_checkpoint(
                {
                    'timestep': total_timesteps,
                    'policy_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode_rewards': episode_rewards,
                    'config': config.__dict__,
                },
                str(checkpoint_path)
            )
            logger.info(f"Checkpoint saved at timestep {total_timesteps}")
    
    # Final save
    final_checkpoint = Path(config.paths.checkpoint_dir) / "final_model.pt"
    save_checkpoint(
        {
            'timestep': total_timesteps,
            'policy_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'config': config.__dict__,
        },
        str(final_checkpoint),
        is_best=True
    )
    
    logger.info("Training completed!")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Average reward (last 100): {np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
