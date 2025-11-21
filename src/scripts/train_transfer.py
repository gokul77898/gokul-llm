"""Training script for Transfer Learning model"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.common import load_config, seed_everything, get_device, init_logger
from src.common.checkpoints import save_checkpoint, find_latest_checkpoint, resume_from_checkpoint
from src.common.utils import GradScalerWrapper, count_parameters
from src.data import load_transfer_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Transfer Learning model")
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
    logger = init_logger("transfer_train", str(log_file))
    logger.info(f"Training Transfer model with config: {args.config}")
    logger.info(f"Device: {device}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset = load_transfer_dataset(config)
    
    # Create tokenizer
    logger.info(f"Loading tokenizer: {config.model.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    
    # Create dataloaders
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.data.max_length,
            return_tensors='pt'
        )
        
        encodings['labels'] = torch.tensor(labels)
        return encodings
    
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
    logger.info(f"Creating model: {config.model.base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.base_model,
        num_labels=config.model.num_labels
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    total_steps = len(train_loader) * config.training.num_epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    best_f1 = 0.0
    
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(config.paths.checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"Resuming from {latest_checkpoint}")
            resume_info = resume_from_checkpoint(
                latest_checkpoint, model, optimizer, scheduler, device
            )
            start_epoch = resume_info['epoch'] + 1
            global_step = resume_info['global_step']
            best_f1 = resume_info.get('best_metric', 0.0)
    
    # Training loop
    scaler = GradScalerWrapper(config.training.mixed_precision and device.type == "cuda")
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.training.mixed_precision):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
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
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        avg_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch} completed | Loss: {avg_loss:.4f} | "
            f"Acc: {accuracy:.4f} | F1: {f1:.4f}"
        )
        
        # Save checkpoint
        checkpoint_path = Path(config.paths.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(
            {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': max(best_f1, f1),
                'metrics': {'accuracy': accuracy, 'f1': f1},
                'config': config.__dict__,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
            str(checkpoint_path),
            is_best=(f1 > best_f1)
        )
        
        if f1 > best_f1:
            best_f1 = f1
    
    logger.info("Training completed!")
    logger.info(f"Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
