"""Supervised Fine-Tuning for RLHF"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common import load_config, seed_everything, get_device, init_logger
from src.common.checkpoints import save_checkpoint
from src.data import load_rl_dataset
from src.core import load_model


def main():
    parser = argparse.ArgumentParser(description="SFT for RLHF")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed_everything(config.system.seed)
    device = get_device(config.system.device)
    
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = init_logger("sft", str(Path(config.paths.log_dir) / "sft.log"))
    logger.info("Starting SFT training")
    
    # Load base model
    model, tokenizer, _ = load_model(config.model.base_model, device=str(device))
    model.train()
    
    # Load data
    train_dataset, val_dataset = load_rl_dataset(config)
    
    def collate_fn(batch):
        texts = [item.get('document', '') for item in batch]
        if hasattr(tokenizer, 'encode'):
            encodings = [tokenizer.encode(text, return_tensors=False) for text in texts]
            max_len = min(max(len(enc['input_ids']) for enc in encodings), config.model.max_length)
            input_ids = []
            for enc in encodings:
                ids = enc['input_ids'][:max_len]
                padding = max_len - len(ids)
                ids = ids + [tokenizer.pad_token_id] * padding
                input_ids.append(ids)
            return {'input_ids': torch.tensor(input_ids)}
        else:
            return {'input_ids': torch.randint(0, 100, (len(batch), 64))}
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, 
                              collate_fn=collate_fn, num_workers=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, 
                                 weight_decay=config.training.weight_decay)
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            
            outputs = model(input_ids, task="generation")
            logits = outputs['logits']
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                ignore_index=getattr(tokenizer, 'pad_token_id', 0)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % config.training.logging_steps == 0:
                logger.info(f"Epoch {epoch} Step {batch_idx+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.4f}")
        
        checkpoint_path = Path(config.paths.checkpoint_dir) / f"sft_epoch_{epoch}.pt"
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, str(checkpoint_path))
    
    final_path = Path(config.paths.checkpoint_dir) / "sft_final.pt"
    save_checkpoint({'model_state_dict': model.state_dict()}, str(final_path))
    logger.info(f"SFT training completed. Final model: {final_path}")


if __name__ == "__main__":
    main()
