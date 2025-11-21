"""Trainer for Transfer Learning Models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, List
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json


class TransferTrainer:
    """Trainer for legal transfer learning models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints/transfer",
        log_interval: int = 100,
        eval_interval: int = 500,
        save_interval: int = 1000
    ):
        """
        Args:
            model: Legal transfer model
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_interval: Logging interval
            eval_interval: Evaluation interval
            save_interval: Checkpoint save interval
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader)
        if warmup_steps == 0:
            warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_metric = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            dynamic_ncols=True
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions for metrics
            if 'logits' in outputs:
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                if 'labels' in batch:
                    all_labels.extend(batch['labels'].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
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
                val_metrics = self.evaluate()
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_accuracy'].append(val_metrics['accuracy'])
                self.training_history['val_f1'].append(val_metrics['f1'])
                
                # Save best model
                if val_metrics['f1'] > self.best_val_metric:
                    self.best_val_metric = val_metrics['f1']
                    self.save_checkpoint('best_model')
                
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}')
        
        # Calculate epoch metrics
        metrics = {
            'loss': total_loss / num_batches
        }
        
        if all_labels and all_preds:
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['f1'] = f1_score(all_labels, all_preds, average='weighted')
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
                
                if 'logits' in outputs:
                    preds = torch.argmax(outputs['logits'], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    if 'labels' in batch:
                        all_labels.extend(batch['labels'].cpu().numpy())
        
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
        
        if all_labels and all_preds:
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['f1'] = f1_score(all_labels, all_preds, average='weighted')
            metrics['precision'] = precision_score(all_labels, all_preds, average='weighted')
            metrics['recall'] = recall_score(all_labels, all_preds, average='weighted')
        
        print(f"\nValidation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def train(self, num_epochs: int):
        """Train for multiple epochs"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        print(f"Trainable parameters: {self.model.get_num_parameters(only_trainable=True):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            if 'accuracy' in train_metrics:
                print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"  Train F1: {train_metrics['f1']:.4f}")
            
            # Evaluate at end of epoch
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                
                if val_metrics['f1'] > self.best_val_metric:
                    self.best_val_metric = val_metrics['f1']
                    self.save_checkpoint('best_model')
        
        # Save final model
        self.save_checkpoint('final_model')
        self._save_training_history()
        
        print("\nTraining completed!")
        print(f"Best validation F1: {self.best_val_metric:.4f}")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        save_dir = self.checkpoint_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(save_dir))
        
        # Save training state
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'training_history': self.training_history
        }, save_dir / 'training_state.pt')
        
        print(f"\nCheckpoint saved: {save_dir}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint"""
        load_dir = self.checkpoint_dir / name
        
        # Load model
        self.model = self.model.from_pretrained(str(load_dir))
        self.model = self.model.to(self.device)
        
        # Load training state
        state = torch.load(
            load_dir / 'training_state.pt',
            map_location=self.device
        )
        
        self.epoch = state['epoch']
        self.global_step = state['global_step']
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.best_val_metric = state['best_val_metric']
        self.training_history = state['training_history']
        
        print(f"Checkpoint loaded: {load_dir}")
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


class LegalDataset(Dataset):
    """Dataset for legal document tasks"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        preprocess: bool = True
    ):
        """
        Args:
            texts: List of text documents
            labels: List of labels
            tokenizer: LegalTokenizer instance
            max_length: Maximum sequence length
            preprocess: Whether to preprocess legal entities
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoded = self.tokenizer.encode(
            text,
            preprocess=self.preprocess,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
