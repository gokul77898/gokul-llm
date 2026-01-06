#!/usr/bin/env python3
"""
TEMPORARY Colab LoRA Training Script
⚠️ FOR VALIDATION ONLY - DELETE BEFORE PRODUCTION

This script is ISOLATED from production code.
Used only for pre-flight validation on Colab T4 GPU.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model

from lora_config import MODEL_NAME, LORA_CONFIG, TRAINING_CONFIG, HARDWARE_REQUIREMENTS


def log(msg: str) -> None:
    """Timestamped logging."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def validate_cuda() -> None:
    """Validate CUDA availability. FAIL HARD if not available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA REQUIRED FOR COLAB TRAINING. "
            "This script does not support CPU execution."
        )
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    log(f"✓ CUDA validated")
    log(f"  GPU: {gpu_name}")
    log(f"  VRAM: {vram_gb:.1f} GB")
    
    if vram_gb < HARDWARE_REQUIREMENTS["min_vram_gb"]:
        raise RuntimeError(
            f"Insufficient VRAM: {vram_gb:.1f} GB < {HARDWARE_REQUIREMENTS['min_vram_gb']} GB required"
        )


class SimpleDataset:
    """Simple dataset for JSONL text files."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        log(f"Loading dataset from: {data_path}")
        
        if not os.path.exists(data_path):
            raise RuntimeError(f"Dataset not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        if 'text' in item and len(item['text']) > 0:
                            self.data.append(item['text'])
                    except json.JSONDecodeError:
                        log(f"WARNING: Invalid JSON at line {line_num}")
        
        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty or contains no valid samples")
        
        log(f"✓ Dataset loaded: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized


class ColabTrainer:
    """TEMPORARY Colab LoRA trainer for validation."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.project_root = Path(__file__).parent
        
        # Validate hardware
        validate_cuda()
        
        self.device = torch.device("cuda")
        
        log("=" * 60)
        log("TEMPORARY COLAB LoRA TRAINING")
        log("⚠️ FOR VALIDATION ONLY")
        log("=" * 60)
        log(f"Model: {MODEL_NAME}")
        log(f"Torch dtype: {HARDWARE_REQUIREMENTS['dtype']}")
    
    def load_model_and_tokenizer(self):
        """Load base model and apply LoRA."""
        log(f"Loading model: {MODEL_NAME}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            padding_side="right",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        log("✓ Tokenizer loaded")
        
        # Load model with FP16
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        log("✓ Base model loaded")
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, LORA_CONFIG)
        
        # Validate trainable parameters
        self._validate_trainable_params()
        
        log("✓ LoRA applied")
    
    def _validate_trainable_params(self):
        """Validate that only LoRA parameters are trainable."""
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        total_params = trainable_params + frozen_params
        trainable_percent = (trainable_params / total_params) * 100
        
        log(f"  Total parameters: {total_params:,}")
        log(f"  Trainable parameters: {trainable_params:,}")
        log(f"  Trainable %: {trainable_percent:.4f}%")
        
        if trainable_percent > 1.0:
            raise RuntimeError(
                f"Too many trainable parameters: {trainable_percent:.2f}% > 1%. "
                "Base model weights may not be properly frozen."
            )
        
        # Verify base model is frozen
        for name, param in self.base_model.named_parameters():
            if "lora" not in name.lower() and param.requires_grad:
                raise RuntimeError(
                    f"Base model parameter {name} has requires_grad=True. "
                    "ABORTING: Base model must be frozen."
                )
    
    def load_dataset(self):
        """Load training dataset."""
        self.train_dataset = SimpleDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            max_length=TRAINING_CONFIG["seq_len"],
        )
    
    def train(self):
        """Run training."""
        log("=" * 60)
        log("STARTING TRAINING")
        log("=" * 60)
        
        # Load everything
        self.load_model_and_tokenizer()
        self.load_dataset()
        
        # Output directory
        output_dir = self.project_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            max_steps=TRAINING_CONFIG["max_steps"],
            logging_steps=TRAINING_CONFIG["logging_steps"],
            save_steps=TRAINING_CONFIG["save_steps"],
            save_total_limit=2,
            fp16=True,
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        log("Training started...")
        start_time = time.time()
        
        trainer.train()
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60  # minutes
        
        log(f"✓ Training completed in {training_time:.1f} minutes")
        
        # Save final adapter
        final_adapter_path = output_dir / "final_adapter"
        self.model.save_pretrained(str(final_adapter_path))
        self.tokenizer.save_pretrained(str(final_adapter_path))
        
        log(f"✓ Adapter saved to: {final_adapter_path}")
        
        return final_adapter_path
    
    def validate_outputs(self, adapter_path: Path):
        """Post-training sanity check - REQUIRED."""
        log("=" * 60)
        log("POST-TRAINING SANITY CHECK")
        log("=" * 60)
        
        test_prompt = "Explain Section 420 of the Indian Penal Code."
        
        log("Testing LoRA model...")
        inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            lora_outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        lora_response = self.tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        
        # Load fresh base model
        log("Testing base model (fresh load)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        base_response = self.tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # Print side-by-side
        log("\n" + "=" * 60)
        log("OUTPUT COMPARISON")
        log("=" * 60)
        log(f"Prompt: {test_prompt}\n")
        log("-" * 30 + " BASE MODEL " + "-" * 30)
        log(base_response[:300])
        log("\n" + "-" * 30 + " LORA MODEL " + "-" * 30)
        log(lora_response[:300])
        log("=" * 60)
        
        # FAIL if outputs are identical
        if base_response == lora_response:
            raise RuntimeError(
                "SANITY CHECK FAILED: Base and LoRA outputs are identical. "
                "LoRA training may not be working correctly."
            )
        
        log("✓ Outputs differ - LoRA is working")
        
        # Cleanup
        del base_model
        torch.cuda.empty_cache()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TEMPORARY Colab LoRA Training")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/train.jsonl",
        help="Path to training data (JSONL)",
    )
    
    args = parser.parse_args()
    
    try:
        trainer = ColabTrainer(args.data_path)
        adapter_path = trainer.train()
        trainer.validate_outputs(adapter_path)
        
        log("=" * 60)
        log("COLAB VALIDATION COMPLETE")
        log("=" * 60)
        log("✓ CUDA GPU used")
        log("✓ LoRA adapters saved")
        log("✓ Base model frozen")
        log("✓ Outputs differ after training")
        log("⚠️ REMEMBER: Delete this directory before production training")
        
    except Exception as e:
        log(f"✗ TRAINING FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
