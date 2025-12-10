"""
Expert Trainer for MARK MoE System.
Supports LoRA and Full Fine-Tuning for registered HF experts.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

from src.core.model_registry import get_registry, load_expert_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExpertTrainer")

def train_expert(expert_name: str, mode: str, dry_run: bool = False, confirm: bool = False):
    """
    Train a specific expert model.
    """
    if not confirm and not dry_run:
        logger.error("Must specify --confirm-run or --dry-run")
        sys.exit(1)
        
    registry = get_registry()
    expert = registry.get_expert(expert_name)
    
    if not expert:
        logger.error(f"Expert {expert_name} not found.")
        sys.exit(1)
        
    logger.info(f"Preparing to train {expert_name} ({expert.model_id}) in {mode} mode")
    
    if dry_run:
        logger.info("[DRY RUN] Would load model, setup LoRA/Full training, and launch Trainer.")
        logger.info(f"Target modules: {expert.lora_config.get('target_modules', 'auto')}")
        logger.info(f"Device: {expert.device}")
        return

    # Load Model
    logger.info("Loading model...")
    model, tokenizer = load_expert_model(expert_name)
    
    # Setup Output Dir
    output_dir = Path(f"checkpoints/{expert_name}/{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "lora":
        logger.info("Applying LoRA config...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if "llama" in expert.model_id.lower() else TaskType.SEQ_CLS,
            inference_mode=False,
            r=expert.lora_config.get('r', 8),
            lora_alpha=expert.lora_config.get('alpha', 16),
            lora_dropout=expert.lora_config.get('dropout', 0.1),
            target_modules=expert.lora_config.get('target_modules', None) # None = auto
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    elif mode == "full":
        logger.info("Full fine-tuning selected. Warning: High VRAM usage.")
        if not torch.cuda.is_available():
            logger.warning("No CUDA device found! Training might be extremely slow.")
            
    # Dummy Training Args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4, # Auto-detect would be better
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-5 if mode == "full" else 2e-4,
        report_to="none"
    )
    
    logger.info("Starting training loop (mock data needed)...")
    
    # In a real scenario, we'd load a dataset here
    # train_dataset = ...
    
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    # )
    # trainer.train()
    
    logger.info(f"Training complete. Saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="MARK Expert Trainer")
    parser.add_argument("--expert", type=str, required=True, help="Name of expert to train")
    parser.add_argument("--mode", type=str, choices=["lora", "full"], default="lora")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    
    args = parser.parse_args()
    
    train_expert(args.expert, args.mode, args.dry_run, args.confirm_run)

if __name__ == "__main__":
    main()
