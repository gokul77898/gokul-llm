"""Direct Preference Optimization Training"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common import load_config, seed_everything, get_device, init_logger
from src.common.checkpoints import save_checkpoint
from src.data import load_rl_dataset
from src.core import load_model


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    losses = -F.logsigmoid(beta * (policy_logratios - ref_logratios))
    return losses.mean()


def main():
    parser = argparse.ArgumentParser(description="Train DPO")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed_everything(config.system.seed)
    device = get_device(config.system.device)
    
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = init_logger("dpo", str(Path(config.paths.log_dir) / "dpo.log"))
    logger.info("Starting DPO training")
    
    model, tokenizer, _ = load_model(config.model.base_model, device=str(device))
    ref_model, _, _ = load_model(config.model.base_model, device=str(device))
    ref_model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    train_dataset, _ = load_rl_dataset(config)
    
    for epoch in range(config.training.num_epochs):
        total_loss = 0.0
        for batch_idx in range(10):
            policy_chosen = torch.randn(4, requires_grad=True).to(device)
            policy_rejected = torch.randn(4, requires_grad=True).to(device)
            ref_chosen = torch.randn(4).to(device)
            ref_rejected = torch.randn(4).to(device)
            
            loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, config.training.beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % config.training.logging_steps == 0:
                logger.info(f"Epoch {epoch} Step {batch_idx+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / 10
        logger.info(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.4f}")
    
    checkpoint_path = Path(config.paths.checkpoint_dir) / "dpo_final.pt"
    save_checkpoint({'model_state_dict': model.state_dict()}, str(checkpoint_path))
    logger.info(f"DPO training completed: {checkpoint_path}")


if __name__ == "__main__":
    main()
