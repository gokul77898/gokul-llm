"""Training script for Transfer Learning model"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.transfer.model import LegalTransferModel, LegalTaskType
from src.transfer.tokenizer import LegalTokenizer
from src.transfer.trainer import TransferTrainer, LegalDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_data(num_samples: int = 1000):
    """Create sample legal documents for training"""
    texts = [
        f"Legal document {i}: This contract between parties Section {i % 10}. "
        f"The plaintiff filed a case on 12/25/2023 under 15 U.S.C. § {i}."
        for i in range(num_samples)
    ]
    labels = [i % 3 for i in range(num_samples)]  # 3 classes
    return texts, labels


def main(args):
    """Main training function"""
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Create tokenizer
    print("Creating legal tokenizer...")
    tokenizer = LegalTokenizer(
        base_model=config['model']['base_model'],
        max_length=config['model']['max_length'],
        add_legal_tokens=config['tokenizer'].get('add_legal_tokens', True)
    )
    
    # Load or create training data
    print("Loading training data...")
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        # TODO: Implement real data loading
        train_texts, train_labels = create_sample_data(1000)
        val_texts, val_labels = create_sample_data(200)
    else:
        print("Using sample data...")
        train_texts, train_labels = create_sample_data(1000)
        val_texts, val_labels = create_sample_data(200)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = LegalDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        preprocess=config['tokenizer'].get('preprocess', True)
    )
    
    val_dataset = LegalDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        preprocess=config['tokenizer'].get('preprocess', True)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0)
    )
    
    # Map task type
    task_map = {
        'classification': LegalTaskType.CLASSIFICATION,
        'ner': LegalTaskType.NER,
        'summarization': LegalTaskType.SUMMARIZATION,
        'generation': LegalTaskType.GENERATION,
        'qa': LegalTaskType.QA
    }
    task = task_map[config['model']['task']]
    
    # Create model
    print(f"Creating transfer model for {task.value}...")
    model = LegalTransferModel(
        model_name=config['model']['base_model'],
        task=task,
        num_labels=config['model'].get('num_labels', 3),
        dropout=config['model'].get('dropout', 0.1),
        freeze_base=config['model'].get('freeze_base', False)
    )
    
    # Resize embeddings if legal tokens were added
    model.resize_token_embeddings(tokenizer.vocab_size)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_parameters(only_trainable=True):,}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = TransferTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=config['training'].get('learning_rate', 2e-5),
        weight_decay=config['training'].get('weight_decay', 0.01),
        warmup_steps=config['training'].get('warmup_steps', 0),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        device=device,
        checkpoint_dir=config['training']['checkpoint_dir'],
        log_interval=config['training'].get('log_interval', 100),
        eval_interval=config['training'].get('eval_interval', 500),
        save_interval=config['training'].get('save_interval', 1000)
    )
    
    # Train model
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    # Save tokenizer
    tokenizer_path = Path(config['training']['checkpoint_dir']) / 'best_model' / 'tokenizer'
    tokenizer.save_pretrained(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transfer learning model for legal data")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/transfer_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to training data'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    args = parser.parse_args()
    main(args)
