#!/usr/bin/env python3
"""
Phase 2: Production LoRA Continuation Training Configuration
============================================================

PHASE 2 OBJECTIVE:
Supreme Court reasoning alignment (1950-2025)

CONTINUATION TRAINING:
- Resume from: OmilosAISolutions/nyayamitra (v0.1, 200 steps)
- New dataset: Supreme Court metadata distilled (31,720 examples)
- Goal: Improve legal reasoning, not memorize facts
- Conservative approach: Lower LR, stable training

PRODUCTION SETTINGS:
- Learning rate: 5e-6 (reduced from 1e-4 for continuation)
- Max steps: 750 (continuation from step 200)
- LoRA params: SAME as v0.1 (r=8, alpha=16)
- Output: nyayamitra-v0.2
"""

from peft import LoraConfig, TaskType

# =============================================================================
# Phase 2: LoRA Configuration (CONTINUATION - KEEP SAME AS v0.1)
# =============================================================================
LORA_CONFIG = LoraConfig(
    r=8,                              # SAME as v0.1 - DO NOT CHANGE
    lora_alpha=16,                    # SAME as v0.1 - DO NOT CHANGE
    lora_dropout=0.05,                # SAME as v0.1 - DO NOT CHANGE
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# =============================================================================
# Model Configuration (CONTINUATION)
# =============================================================================
MODEL_CONFIG = {
    "base_model": "Qwen/Qwen2.5-32B-Instruct",
    "adapter_model": "OmilosAISolutions/nyayamitra",  # Resume from v0.1
    "max_seq_length": 2048,
    "torch_dtype": "bfloat16",
    "trust_remote_code": True,
}

# =============================================================================
# Phase 2: Training Configuration (CONTINUATION)
# =============================================================================
TRAINING_CONFIG = {
    # Batch settings (SAME as v0.1)
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 32,
    
    # Learning rate (REDUCED for continuation)
    "learning_rate": 5e-6,            # REDUCED from 1e-4 (continuation training)
    
    # Optimizer settings (SAME)
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Scheduler settings
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,             # 5% warmup
    
    # Training steps
    "max_steps": 750,                 # Phase 2 target (continuation from 200)
    "resume_from_checkpoint": True,   # Resume from v0.1
    
    # Logging and saving
    "logging_steps": 10,
    "save_steps": 150,                # Save every 150 steps
    "eval_steps": 150,
    "save_total_limit": 5,
    
    # Training stability
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True,
    
    # Data settings
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "shuffle": True,                  # Shuffle enabled for better generalization
}

# =============================================================================
# Phase 2: Dataset Configuration
# =============================================================================
DATASET_CONFIG = {
    "dataset_name": "OmilosAISolutions/nyayamitra-lora-train-v1",
    "dataset_file": "sc_lora_training.jsonl",
    "split": "train",
    "text_field": None,               # Will use instruction-input-output format
    "max_seq_length": 2048,
    "streaming": False,
    "num_proc": 8,
    "seed": 42,
}

# =============================================================================
# Phase 2: Output Configuration
# =============================================================================
OUTPUT_CONFIG = {
    "output_dir": "./training/checkpoints/phase2_continuation",
    "hub_model_id": "OmilosAISolutions/nyayamitra",  # SAME repo
    "hub_strategy": "every_save",
    "push_to_hub": True,
    "hub_private_repo": False,
    "version_tag": "v0.2",            # New version tag
    "phase_note": "Phase 2 – Supreme Court reasoning alignment (1950–2025)",
}

# =============================================================================
# Phase 2: Hardware Requirements (SAME as v0.1)
# =============================================================================
HARDWARE_REQUIREMENTS = {
    "min_gpus": 4,
    "min_vram_per_gpu_gb": 24,
    "supported_gpus": ["RTX_4090", "RTX_3090", "A100", "H100"],
    "cuda_required": True,
    "bf16_required": True,
}

# =============================================================================
# Phase 2: Training Metrics to Track
# =============================================================================
METRICS_TO_TRACK = {
    "starting_step": 200,             # Continuation from v0.1
    "target_step": 750,               # Phase 2 target
    "total_new_steps": 550,           # Additional steps in Phase 2
    "expected_duration_hours": 2,     # Estimated on 4×RTX 4090
    "track_metrics": [
        "train_loss",
        "learning_rate",
        "grad_norm",
        "eval_loss",
    ],
}

# =============================================================================
# Phase 2: Safety Checks
# =============================================================================
SAFETY_CHECKS = {
    "verify_adapter_loaded": True,    # Ensure v0.1 adapter is loaded
    "verify_lr_reduced": True,        # Ensure LR is 5e-6
    "verify_same_lora_params": True,  # Ensure r=8, alpha=16
    "max_loss_spike": 0.5,            # Alert if loss spikes > 0.5
    "min_eval_frequency": 150,        # Evaluate every 150 steps
}

# =============================================================================
# Phase 2: Post-Training Actions
# =============================================================================
POST_TRAINING = {
    "save_final_adapter": True,
    "tag_as_v0.2": True,
    "keep_v0.1_intact": True,         # DO NOT overwrite v0.1
    "upload_to_hub": True,
    "stop_after_training": True,      # DO NOT redeploy endpoint
    "next_phase": "Phase 3: Endpoint Update + Eval",
}

# =============================================================================
# Phase 2: Training Notes
# =============================================================================
TRAINING_NOTES = """
Phase 2: Supreme Court Reasoning Alignment (1950-2025)

OBJECTIVE:
- Improve legal reasoning capabilities
- Align with Supreme Court jurisprudence
- NOT memorization - reasoning refinement

DATASET:
- Source: Supreme Court of India metadata
- Years: 1950-2025 (76 years)
- Examples: 31,720
- Size: 23 MB
- Format: Instruction-tuning (legal reasoning)

APPROACH:
- Conservative continuation training
- Reduced learning rate (5e-6)
- Stable, production-quality training
- No architectural changes

EXPECTED OUTCOME:
- Better legal reasoning
- Improved section citation accuracy
- Enhanced case law understanding
- Stable, production-ready adapter (v0.2)

DO NOT:
- Redeploy endpoint yet
- Auto-merge adapters
- Change base model
- Modify inference code
"""
