"""Training utilities for Mamba model"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, List, Callable
from tqdm import tqdm
import os
import json
from pathlib import Path


class MambaTrainer:
    """Trainer for Mamba model with support for classification and generation tasks"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
        accumulation_steps: int = 1,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        task: str = "classification"
    ):
        """
        Args:
            model: Mamba model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer (if None, creates AdamW)
            scheduler: Learning rate scheduler
            device: Device to train on
            max_grad_norm: Maximum gradient norm for clipping
            accumulation_steps: Gradient accumulation steps
            checkpoint_dir: Directory to save checkpoints
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            save_interval: Steps between saving checkpoints
            task: "classification" or "generation"
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.task = task
        
        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=5e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Initialize scheduler if not provided
        if scheduler is None:
            # Warmup for 10% of training, then cosine annealing
            total_steps = len(train_dataloader)
            warmup_steps = int(0.1 * total_steps)
            
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-7
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            self.scheduler = scheduler
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            dynamic_ncols=True
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            chunk_boundaries = batch.get('chunk_boundaries', None)
            if chunk_boundaries is not None:
                chunk_boundaries = chunk_boundaries.to(self.device)
            
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunk_boundaries=chunk_boundaries,
                labels=labels,
                task=self.task
            )
            
            loss = outputs['loss']
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                self.training_history['train_loss'].append(avg_loss)
                self.training_history['learning_rate'].append(
                    self.scheduler.get_last_lr()[0]
                )
            
            # Evaluation
            if self.val_dataloader is not None and self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                self.training_history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        return total_loss / num_batches
    
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                chunk_boundaries = batch.get('chunk_boundaries', None)
                if chunk_boundaries is not None:
                    chunk_boundaries = chunk_boundaries.to(self.device)
                
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    chunk_boundaries=chunk_boundaries,
                    labels=labels,
                    task=self.task
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"\nValidation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self, num_epochs: int):
        """Train for multiple epochs"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        print(f"Trainable parameters: {self.model.get_num_parameters(only_trainable=True):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Evaluate at end of epoch
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        self._save_training_history()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded: {path}")
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


class DocumentDataset(Dataset):
    """Dataset for document classification/generation tasks"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        task: str = "classification"
    ):
        """
        Args:
            texts: List of text documents
            labels: List of labels (class IDs for classification, next tokens for generation)
            tokenizer: Document tokenizer
            max_length: Maximum sequence length
            task: "classification" or "generation"
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors=True
        )
        
        return {
            'input_ids': encoded.input_ids.squeeze(0),
            'attention_mask': encoded.attention_mask.squeeze(0),
            'chunk_boundaries': encoded.chunk_boundaries.squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
