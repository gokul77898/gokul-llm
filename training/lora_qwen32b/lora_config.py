#!/usr/bin/env python3
"""
Production LoRA Configuration for Qwen2.5-32B-Instruct
CUDA-only, multi-GPU with DeepSpeed ZeRO-2
"""

from peft import LoraConfig, TaskType

# LoRA Configuration (Production)
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)

# Model Configuration
MODEL_CONFIG = {
    "base_model": "Qwen/Qwen2.5-32B-Instruct",
    "max_seq_length": 2048,
    "torch_dtype": "bfloat16",
    "trust_remote_code": True,
}

# Training Configuration
TRAINING_CONFIG = {
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "lr_scheduler_type": "cosine",
    "save_steps": 1000,
    "logging_steps": 50,
    "eval_steps": 2000,
    "save_total_limit": 5,
    "max_grad_norm": 1.0,
}

# Hardware Requirements
HARDWARE_REQUIREMENTS = {
    "min_gpus": 2,
    "min_vram_per_gpu_gb": 24,
    "supported_gpus": ["A100", "H100", "RTX 4090", "RTX 3090"],
    "cuda_required": True,
    "bf16_required": True,
}

# Data Configuration
DATA_CONFIG = {
    "max_seq_length": 2048,
    "text_field": "text",
    "streaming": True,
    "num_proc": 8,
}
