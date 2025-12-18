"""Training script for Reinforcement Learning"""

import argparse
import yaml
import torch

from src.mamba.model import MambaModel
from src.mamba.tokenizer import DocumentTokenizer
from src.rl.trainer import create_rl_trainer_for_task


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function"""
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Create base model and tokenizer
    print("Creating base model and tokenizer...")
    
    # Use a pre-trained model or create new one
    if args.base_model:
        print(f"Loading base model from {args.base_model}")
        model = MambaModel.from_pretrained(args.base_model)
        tokenizer = DocumentTokenizer()
        tokenizer.load_vocab(f"{args.base_model}/tokenizer.json")
    else:
        print("Creating new model...")
        model = MambaModel(
            vocab_size=config['model']['vocab_size'],
            d_model=config['model']['d_model'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            num_classes=config['model'].get('num_classes', None)
        )
        tokenizer = DocumentTokenizer(vocab_size=config['model']['vocab_size'])
    
    model = model.to(device)
    model.eval()  # Use in eval mode for RL environment
    
    # Create RL trainer
    print(f"Creating RL trainer for task: {config['task']['type']}")
    trainer = create_rl_trainer_for_task(
        task_type=config['task']['type'],
        model=model,
        tokenizer=tokenizer,
        agent_type=config['agent']['type'],
        total_timesteps=config['training']['total_timesteps'],
        learning_rate=config['agent'].get('learning_rate', 3e-4),
        batch_size=config['agent'].get('batch_size', 64),
        device=device
    )
    
    # Train agent
    print(f"\nStarting RL training for {config['training']['total_timesteps']} timesteps...")
    trainer.train(
        total_timesteps=config['training']['total_timesteps'],
        eval_episodes=config['training'].get('eval_episodes', 10)
    )
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_reward = trainer.evaluate(num_episodes=20, deterministic=True)
    print(f"Final average reward: {final_reward:.2f}")
    
    print("\nRL training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for legal document tasks")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rl_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default=None,
        help='Path to pre-trained base model'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    args = parser.parse_args()
    main(args)
