"""Reward Model Training"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common import load_config, seed_everything, get_device, init_logger
from src.common.checkpoints import save_checkpoint
from src.data import load_rl_dataset


class RewardModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed_everything(config.system.seed)
    device = get_device(config.system.device)
    
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = init_logger("reward_model", str(Path(config.paths.log_dir) / "reward.log"))
    logger.info("Starting reward model training")
    
    model = RewardModel(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate,
                                 weight_decay=config.training.weight_decay)
    
    # Synthetic preference data
    train_dataset, _ = load_rl_dataset(config)
    
    for epoch in range(config.training.num_epochs):
        total_loss = 0.0
        for batch_idx in range(20):
            state = torch.randn(config.training.batch_size, config.model.input_dim).to(device)
            target_reward = torch.randn(config.training.batch_size).to(device)
            
            predicted_reward = model(state)
            loss = nn.functional.mse_loss(predicted_reward, target_reward)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % config.training.logging_steps == 0:
                logger.info(f"Epoch {epoch} Step {batch_idx+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / 20
        logger.info(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.4f}")
    
    checkpoint_path = Path(config.paths.checkpoint_dir) / "reward_model_final.pt"
    save_checkpoint({'model_state_dict': model.state_dict()}, str(checkpoint_path))
    logger.info(f"Reward model training completed: {checkpoint_path}")


if __name__ == "__main__":
    main()
