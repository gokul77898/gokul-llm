#!/usr/bin/env python3
"""
TEMPORARY Colab LoRA Configuration
⚠️ FOR VALIDATION ONLY - DELETE BEFORE PRODUCTION
"""

from peft import LoraConfig, TaskType

# Model Configuration (TEMPORARY - Colab T4 compatible)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# LoRA Configuration (matches production shape)
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)

# Training Configuration (Colab-safe limits)
TRAINING_CONFIG = {
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_steps": 300,
    "seq_len": 512,
    "learning_rate": 2e-4,
    "logging_steps": 10,
    "save_steps": 100,
    "warmup_steps": 50,
}

# Hardware Requirements (Colab T4)
HARDWARE_REQUIREMENTS = {
    "cuda_required": True,
    "min_vram_gb": 15,
    "dtype": "float16",
}
