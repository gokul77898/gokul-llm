"""Training script for Mamba model"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.mamba.model import MambaModel
from src.mamba.tokenizer import DocumentTokenizer
from src.mamba.trainer import MambaTrainer, DocumentDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_data(num_samples: int = 1000):
    """Create sample legal documents for training"""
    texts = [
        f"Legal document {i}: This contract discusses terms and conditions. "
        f"Parties involved must comply with regulations. Section {i % 10} applies."
        for i in range(num_samples)
    ]
    labels = [i % 5 for i in range(num_samples)]  # 5 classes
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
    print("Creating tokenizer...")
    tokenizer = DocumentTokenizer(
        vocab_size=config['model']['vocab_size'],
        max_length=config['model']['max_length'],
        chunk_size=config['model']['chunk_size'],
        chunk_overlap=config['model']['chunk_overlap']
    )
    
    # Load or create training data
    print("Loading training data...")
    if args.data_path:
        # Load real data
        # TODO: Implement data loading
        print(f"Loading data from {args.data_path}")
        train_texts, train_labels = create_sample_data(1000)
        val_texts, val_labels = create_sample_data(200)
    else:
        # Use sample data
        print("Using sample data...")
        train_texts, train_labels = create_sample_data(1000)
        val_texts, val_labels = create_sample_data(200)
    
    # Build vocabulary
    print("Building vocabulary...")
    tokenizer.build_vocab(train_texts, min_freq=config['tokenizer'].get('min_freq', 2))
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = DocumentDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        task=config['training']['task']
    )
    
    val_dataset = DocumentDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        task=config['training']['task']
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
    
    # Create model
    print("Creating Mamba model...")
    model = MambaModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model'].get('d_ff', None),
        max_seq_length=config['model']['max_length'],
        dropout=config['model'].get('dropout', 0.1),
        num_classes=config['model'].get('num_classes', None),
        use_hierarchical=config['model'].get('use_hierarchical', True),
        positional_encoding=config['model'].get('positional_encoding', 'absolute')
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_parameters(only_trainable=True):,}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = MambaTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        accumulation_steps=config['training'].get('accumulation_steps', 1),
        checkpoint_dir=config['training']['checkpoint_dir'],
        log_interval=config['training'].get('log_interval', 100),
        eval_interval=config['training'].get('eval_interval', 1000),
        save_interval=config['training'].get('save_interval', 1000),
        task=config['training']['task']
    )
    
    # Train model
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    # Save tokenizer
    tokenizer_path = Path(config['training']['checkpoint_dir']) / 'tokenizer.json'
    tokenizer.save_vocab(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mamba model for legal documents")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mamba_config.yaml',
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
