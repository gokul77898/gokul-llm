"""PPO Training"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common import load_config, seed_everything, get_device, init_logger
from src.common.checkpoints import save_checkpoint
from src.data import load_rl_dataset
from src.rl.policy import ActorCritic
from src.rl.rollout_buffer import RolloutBuffer
from src.training.rl_utils import normalize_advantages


class SimpleEnv:
    def __init__(self, obs_dim: int, max_steps: int):
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.obs_dim).astype(np.float32)
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = np.random.randn() * 0.1
        done = self.step_count >= self.max_steps
        return obs, reward, done, {}


def main():
    parser = argparse.ArgumentParser(description="Train PPO")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed_everything(config.system.seed)
    device = get_device(config.system.device)
    
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = init_logger("ppo", str(Path(config.paths.log_dir) / "ppo.log"))
    logger.info("Starting PPO training")
    
    model = ActorCritic(
        obs_dim=config.model.obs_dim,
        action_dim=config.model.action_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    env = SimpleEnv(config.model.obs_dim, config.environment.max_episode_steps)
    buffer = RolloutBuffer(config.training.n_steps, config.model.obs_dim, str(device))
    
    obs = env.reset()
    episode_rewards = []
    current_reward = 0.0
    
    for step in range(config.training.total_timesteps):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            action, log_prob, value = model.get_action_and_value(obs_tensor)
            action = action.item()
            log_prob = log_prob.item()
            value = value.item()
        
        next_obs, reward, done, _ = env.step(action)
        buffer.store(obs, action, reward, value, log_prob, done)
        current_reward += reward
        obs = next_obs
        
        if done:
            episode_rewards.append(current_reward)
            current_reward = 0.0
            obs = env.reset()
        
        if buffer.full or (step + 1) % config.training.n_steps == 0:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                _, _, last_value = model.get_action_and_value(obs_tensor)
                last_value = last_value.item()
            
            advantages, returns = buffer.compute_advantages(last_value, config.training.gamma, config.training.gae_lambda)
            advantages = normalize_advantages(advantages)
            
            data = buffer.get()
            data['advantages'] = torch.from_numpy(advantages).to(device)
            data['returns'] = torch.from_numpy(returns).to(device)
            
            for _ in range(config.training.n_epochs):
                indices = torch.randperm(len(data['observations']))
                for start in range(0, len(indices), config.training.batch_size):
                    end = start + config.training.batch_size
                    idx = indices[start:end]
                    
                    log_probs, values, entropy = model.evaluate(data['observations'][idx], data['actions'][idx])
                    
                    ratio = torch.exp(log_probs - data['log_probs'][idx])
                    surr1 = ratio * data['advantages'][idx]
                    surr2 = torch.clamp(ratio, 1 - config.training.clip_range, 1 + config.training.clip_range) * data['advantages'][idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = F.mse_loss(values, data['returns'][idx])
                    entropy_loss = -entropy.mean()
                    
                    loss = policy_loss + config.training.vf_coef * value_loss + config.training.ent_coef * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    optimizer.step()
            
            if (step + 1) % config.training.logging_steps == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                logger.info(f"Step {step+1}: Avg Reward = {avg_reward:.2f}, Policy Loss = {policy_loss.item():.4f}")
    
    checkpoint_path = Path(config.paths.checkpoint_dir) / "ppo_final.pt"
    save_checkpoint({'model_state_dict': model.state_dict()}, str(checkpoint_path))
    logger.info(f"PPO training completed: {checkpoint_path}")


if __name__ == "__main__":
    main()
