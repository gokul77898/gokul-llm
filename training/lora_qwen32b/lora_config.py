#!/usr/bin/env python3
"""
Production LoRA Configuration for Qwen2.5-32B-Instruct
CUDA-only, multi-GPU with DeepSpeed ZeRO-2

Hardware Support:
- 4× RTX 4090 (24GB each) - PRIMARY TARGET
- 4× RTX 3090 (24GB each) - Supported
- 4× A100 (40/80GB each) - Supported
- 4× H100 (80GB each) - Supported

WARNING: 2× RTX 4090 is UNSTABLE for 32B model.
         Use only for experimental/debugging runs.
         OOM and gradient overflow likely with 2 GPUs.
"""

from peft import LoraConfig, TaskType

# =============================================================================
# LoRA Configuration (Memory Safe for 24GB GPUs)
# =============================================================================
# WHY r=8: Higher ranks (16, 32) increase VRAM usage significantly.
#          r=8 with alpha=16 provides good adaptation with ~0.07% trainable params.
# WHY only q_proj, v_proj: Adding more modules increases memory and instability.
#          These two capture most of the adaptation benefit.
LORA_CONFIG = LoraConfig(
    r=8,                              # Low rank for memory efficiency
    lora_alpha=16,                    # alpha/r = 2 is standard scaling
    lora_dropout=0.05,                # Light dropout for regularization
    bias="none",                      # No bias training - saves memory
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],  # ONLY these - do NOT add more
)

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_CONFIG = {
    "base_model": "Qwen/Qwen2.5-32B-Instruct",
    "max_seq_length": 2048,           # DO NOT increase - OOM risk
    "torch_dtype": "bfloat16",        # BF16 required for stability
    "trust_remote_code": True,
}

# =============================================================================
# Training Configuration
# =============================================================================
# WHY batch_size=1: 32B model with 24GB VRAM cannot fit larger batches.
# WHY grad_accum=32: Compensates for small batch, effective batch = 4*1*32 = 128
# WHY this matters: Larger effective batch = more stable gradients
TRAINING_CONFIG = {
    "micro_batch_size": 1,            # DO NOT increase - OOM guaranteed
    "gradient_accumulation_steps": 32, # Effective batch = num_gpus * 1 * 32
    "learning_rate": 1e-4,            # Standard for LoRA
    "weight_decay": 0.01,
    "warmup_steps": 1000,             # ~8% of typical training
    "lr_scheduler_type": "cosine",
    "save_steps": 1000,
    "logging_steps": 50,
    "eval_steps": 2000,
    "save_total_limit": 5,
    "max_grad_norm": 1.0,             # Gradient clipping for stability
}

# =============================================================================
# Hardware Requirements
# =============================================================================
# PRIMARY TARGET: 4× RTX 4090 (24GB each)
# WARNING: 2× RTX 4090 is experimental only - expect OOM or instability
HARDWARE_REQUIREMENTS = {
    "min_gpus": 4,                    # 4 GPUs required for stable 32B training
    "min_vram_per_gpu_gb": 24,        # RTX 4090/3090 minimum
    "supported_gpus": [
        "RTX_4090",                   # Primary target - 24GB
        "RTX_3090",                   # Supported - 24GB
        "A100",                       # Supported - 40/80GB
        "H100",                       # Supported - 80GB
    ],
    "cuda_required": True,
    "bf16_required": True,            # BF16 required for 32B stability
    # EXPERIMENTAL: 2 GPUs - uncomment only for debugging
    # "experimental_min_gpus": 2,     # UNSTABLE - OOM likely
}

# =============================================================================
# Training Safety Caps
# =============================================================================
# WHY caps: Prevent accidental full-corpus training (300GB = days of compute)
# RECOMMENDED: Start with capped run, then scale up
TRAINING_CAPS = {
    "default_max_steps": 10000,       # ~12-16 hours on 4×4090
    "default_max_tokens": 8_000_000_000,  # 8B tokens max
    "full_corpus_warning_gb": 50,     # Warn if data > 50GB
    "checkpoint_every_hours": 1,      # Safety checkpoints
}

# =============================================================================
# Data Configuration
# =============================================================================
DATA_CONFIG = {
    "max_seq_length": 2048,
    "text_field": "text",
    "streaming": False,               # Streaming disabled for deterministic training
    "num_proc": 8,
    "seed": 42,                       # Fixed seed for reproducibility
}
