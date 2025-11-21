"""Training script for Mamba model"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.common import load_config, seed_everything, get_device, init_logger
from src.common.checkpoints import save_checkpoint, find_latest_checkpoint, resume_from_checkpoint
from src.common.utils import GradScalerWrapper, count_parameters
from src.data import load_mamba_dataset
from src.mamba import DocumentTokenizer, MambaModel, MambaTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Mamba model")
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
    logger = init_logger("mamba_train", str(log_file))
    logger.info(f"Training Mamba model with config: {args.config}")
    logger.info(f"Device: {device}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset = load_mamba_dataset(config)
    
    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = DocumentTokenizer(
        vocab_size=config.model.vocab_size,
        max_length=config.model.max_length,
        chunk_size=config.model.chunk_size,
        chunk_overlap=config.model.chunk_overlap
    )
    
    # Build vocabulary from training data
    train_texts = [item['text'] for item in train_dataset]
    tokenizer.build_vocab(train_texts[:min(1000, len(train_texts))])
    logger.info(f"Vocabulary size: {len(tokenizer.token2id)}")
    
    # Save vocabulary
    vocab_path = Path(config.paths.checkpoint_dir) / "vocab.json"
    tokenizer.save_vocab(str(vocab_path))
    
    # Create dataloaders
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encodings = [tokenizer.encode(text, return_tensors=False) for text in texts]
        
        # Pad sequences
        max_len = max(len(enc['input_ids']) for enc in encodings)
        input_ids = []
        attention_mask = []
        
        for enc in encodings:
            ids = enc['input_ids']
            mask = enc['attention_mask']
            
            padding = max_len - len(ids)
            ids = ids + [tokenizer.pad_token_id] * padding
            mask = mask + [0] * padding
            
            input_ids.append(ids)
            attention_mask.append(mask)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    logger.info("Creating model...")
    model = MambaModel(
        vocab_size=len(tokenizer.token2id),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_length=config.model.max_length,
        dropout=config.model.dropout,
        task="classification"
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.warmup_steps,
        T_mult=2
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(config.paths.checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"Resuming from {latest_checkpoint}")
            resume_info = resume_from_checkpoint(
                latest_checkpoint, model, optimizer, scheduler, device
            )
            start_epoch = resume_info['epoch'] + 1
            global_step = resume_info['global_step']
            best_loss = resume_info['best_metric']
    
    # Training loop
    scaler = GradScalerWrapper(config.training.mixed_precision and device.type == "cuda")
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.training.mixed_precision):
                outputs = model(input_ids, attention_mask)
                # Simple language modeling loss
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
            
            # Backward pass
            loss = loss / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
            
            epoch_loss += loss.item() * config.training.gradient_accumulation_steps
            
            # Logging
            if global_step % config.training.logging_steps == 0:
                logger.info(
                    f"Epoch: {epoch} | Step: {global_step} | "
                    f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )
        
        # Epoch end
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = Path(config.paths.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(
            {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_loss,
                'config': config.__dict__,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
            str(checkpoint_path),
            is_best=(avg_loss < best_loss)
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
