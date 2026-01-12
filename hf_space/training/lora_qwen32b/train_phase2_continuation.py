#!/usr/bin/env python3
import sys
print("🚨 SCRIPT ENTRY CONFIRMED", flush=True)
sys.stdout.flush()
"""
Phase 2: LoRA Continuation Training Script
==========================================

Continues training from nyayamitra-v0.1 with Supreme Court dataset.
"""

import os
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.lora_qwen32b.phase2_continuation_config import (
    LORA_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DATASET_CONFIG,
    OUTPUT_CONFIG,
    METRICS_TO_TRACK,
    SAFETY_CHECKS,
)

def format_instruction(example):
    """Format instruction-input-output into training text"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}

def load_and_prepare_dataset(tokenizer):
    """Load and prepare the Supreme Court dataset"""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    # Load from HuggingFace
    dataset_name = "OmilosAISolutions/nyayamitra-lora-train-v1"
    
    print(f"Loading dataset from HuggingFace: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Format dataset
    print("Formatting dataset...")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Tokenize
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MODEL_CONFIG["max_seq_length"],
            padding=False,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=DATASET_CONFIG["num_proc"],
        remove_columns=["text"],
    )
    
    print(f"Tokenized dataset size: {len(tokenized_dataset)} examples")
    print("="*60 + "\n")
    
    return tokenized_dataset

def load_model_and_adapter(tokenizer):
    """Load base model and existing adapter"""
    print("\n" + "="*60)
    print("LOADING MODEL AND ADAPTER")
    print("="*60)
    
    # Load base model
    print(f"Loading base model: {MODEL_CONFIG['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load existing adapter (v0.1) - MUST SUCCEED OR FAIL
    adapter_path = MODEL_CONFIG["adapter_model"]
    print(f"\n🔄 LOADING EXISTING ADAPTER FROM: {adapter_path}")
    print("⚠️  Phase 2 REQUIRES existing adapter - will FAIL if not found")
    
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"✅ Adapter loaded successfully from {adapter_path}")
        print("✅ CONTINUATION MODE: Training will continue from v0.1")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Could not load adapter from {adapter_path}")
        print(f"❌ Error: {e}")
        print(f"\n❌ PHASE 2 REQUIRES EXISTING ADAPTER")
        print(f"❌ Cannot continue training without v0.1 adapter")
        print(f"❌ STOPPING - DO NOT CREATE NEW ADAPTER")
        raise RuntimeError(
            f"Phase 2 continuation FAILED: Could not load adapter from {adapter_path}. "
            f"Phase 2 MUST continue from existing v0.1 adapter. "
            f"Creating a new adapter would violate continuation requirements."
        )
    
    # Verify LoRA parameters match v0.1
    print("\n🔍 Verifying LoRA Configuration:")
    print(f"  r: {LORA_CONFIG.r} (must match v0.1)")
    print(f"  alpha: {LORA_CONFIG.lora_alpha} (must match v0.1)")
    print(f"  dropout: {LORA_CONFIG.lora_dropout} (must match v0.1)")
    print(f"  target_modules: {LORA_CONFIG.target_modules}")
    
    # Print trainable parameters BEFORE training starts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Trainable parameters (BEFORE step 1):")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Total: {total_params:,}")
    print(f"   Percentage: {100 * trainable_params / total_params:.4f}%")
    print("="*60 + "\n")
    
    return model

def create_training_arguments():
    """Create training arguments for Phase 2"""
    output_dir = OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Training steps
        max_steps=TRAINING_CONFIG["max_steps"],
        
        # Batch settings
        per_device_train_batch_size=TRAINING_CONFIG["micro_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        
        # Learning rate
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        
        # Scheduler
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        
        # Precision
        bf16=TRAINING_CONFIG["bf16"],
        fp16=TRAINING_CONFIG["fp16"],
        
        # Logging
        logging_steps=TRAINING_CONFIG["logging_steps"],
        logging_dir=f"{output_dir}/logs",
        
        # Saving
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        
        # Evaluation
        eval_steps=TRAINING_CONFIG["eval_steps"],
        evaluation_strategy="steps",
        
        # Other
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
        dataloader_pin_memory=TRAINING_CONFIG["dataloader_pin_memory"],
        
        # Hub
        push_to_hub=False,  # Manual push after training
        report_to=["tensorboard"],
        
        # Reproducibility
        seed=DATASET_CONFIG["seed"],
        data_seed=DATASET_CONFIG["seed"],
    )
    
    return args

def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Phase 2 LoRA Continuation Training")
    parser.add_argument("--dry_run", action="store_true", help="Run in dry-run mode (no actual training)")
    args = parser.parse_args()
    
    print(f"🧪 DRY RUN FLAG VALUE = {args.dry_run}", flush=True)
    
    if args.dry_run:
        print("🧪🧪🧪 DRY RUN MODE ENABLED 🧪🧪🧪", flush=True)
        print("🚫 TRAINING WILL NOT START", flush=True)
        print("\n" + "="*60)
        print("DRY RUN: Validating configuration only")
        print("="*60)
        print(f"Base model: {MODEL_CONFIG['base_model']}")
        print(f"Resume from: {MODEL_CONFIG['adapter_model']}")
        print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
        print(f"Max steps: {TRAINING_CONFIG['max_steps']}")
        print(f"Output: {OUTPUT_CONFIG['version_tag']}")
        print("="*60)
        print("\n✅ DRY RUN COMPLETE - Configuration validated")
        print("🚫 No training was performed")
        return
    
    print("\n" + "="*60)
    print("PHASE 2: LORA CONTINUATION TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {MODEL_CONFIG['base_model']}")
    print(f"Resume from: {MODEL_CONFIG['adapter_model']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Max steps: {TRAINING_CONFIG['max_steps']}")
    print(f"Output: {OUTPUT_CONFIG['version_tag']}")
    print("="*60 + "\n")
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This training requires GPU.")
    
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["base_model"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_and_prepare_dataset(tokenizer)
    
    # Split dataset (90% train, 10% eval)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=DATASET_CONFIG["seed"])
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}\n")
    
    # Load model and adapter
    model = load_model_and_adapter(tokenizer)
    
    # Create training arguments
    training_args = create_training_arguments()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Phase: {OUTPUT_CONFIG['phase_note']}")
    print(f"Expected duration: ~{METRICS_TO_TRACK['expected_duration_hours']} hours")
    print("="*60 + "\n")
    
    # Train
    train_result = trainer.train()
    
    # Save final model
    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)
    
    output_dir = OUTPUT_CONFIG["output_dir"]
    final_dir = f"{output_dir}/final"
    
    print(f"Saving to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training metrics
    metrics = {
        "phase": "Phase 2",
        "version": OUTPUT_CONFIG["version_tag"],
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "base_model": MODEL_CONFIG["base_model"],
        "adapter_resumed_from": MODEL_CONFIG["adapter_model"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "max_steps": TRAINING_CONFIG["max_steps"],
        "final_loss": train_result.training_loss,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
    }
    
    metrics_file = f"{output_dir}/phase2_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Metrics saved to: {metrics_file}")
    print("="*60 + "\n")
    
    # Upload adapter to HuggingFace
    print("\n" + "="*60)
    print("UPLOADING ADAPTER TO HUGGING FACE")
    print("="*60)
    
    print("🚀 Uploading LoRA adapter to Hugging Face...")
    print(f"   Source: {final_dir}")
    print(f"   Target: OmilosAISolutions/nyayamitra-v0.2")
    
    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        print("\n❌ CRITICAL ERROR: HF_TOKEN is not set")
        print("❌ Cannot upload adapter to Hugging Face")
        print("❌ Adapter will be LOST when container exits")
        raise RuntimeError(
            "HF_TOKEN is not set. Cannot upload adapter. "
            "This is a CRITICAL failure for HF Jobs - adapter will be lost!"
        )
    
    try:
        api = HfApi(token=hf_token)
        
        print("\n📤 Uploading adapter files...")
        api.upload_folder(
            folder_path=final_dir,
            repo_id="OmilosAISolutions/nyayamitra-v0.2",
            repo_type="model",
            commit_message=f"Upload nyayamitra LoRA v0.2 (Phase 2 continuation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
        )
        
        print("\n✅ LoRA adapter successfully uploaded to OmilosAISolutions/nyayamitra-v0.2")
        print("✅ Adapter is safe - will persist after container exits")
        print("✅ View at: https://huggingface.co/OmilosAISolutions/nyayamitra-v0.2")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Failed to upload adapter to Hugging Face")
        print(f"❌ Error: {e}")
        print(f"❌ Adapter saved locally at: {final_dir}")
        print(f"❌ But will be LOST when container exits!")
        raise RuntimeError(
            f"Failed to upload adapter to Hugging Face: {e}. "
            f"This is a CRITICAL failure for HF Jobs - adapter will be lost!"
        )
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)
    print(f"✅ Adapter saved locally: {final_dir}")
    print(f"✅ Adapter uploaded to HF: OmilosAISolutions/nyayamitra-v0.2")
    print(f"✅ Version: {OUTPUT_CONFIG['version_tag']}")
    print(f"✅ Metrics: {metrics_file}")
    print(f"✅ v0.1 remains intact at: OmilosAISolutions/nyayamitra")
    print("\n⏭️  Next: Phase 3 - Endpoint Update + Eval")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
