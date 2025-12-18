# Archived Premature Training Code

**Phase 3.7: Safety Reset**

This directory contains RLHF, RL, and LoRA training code that was prematurely added to the repository. These files have been archived (not deleted) to preserve the work for future phases.

## Why Archived?

1. **Training is NOT enabled** in Phase 3.5
2. **RLHF/PPO/DPO** requires significant infrastructure not yet in place
3. **LoRA fine-tuning** should only be enabled after data pipelines are ready
4. **Inference code must remain isolated** from training dependencies

## Directory Structure

```
_archive/premature_training/
├── rlhf/                    # RLHF pipeline and trainers
│   ├── continuous_rlhf_training.py
│   ├── simple_rlhf_improvement.py
│   ├── rlhf_pipeline.py
│   ├── rlhf_trainer.py
│   ├── rlhf_sft.py
│   ├── ppo_trainer.py
│   ├── train_ppo.py
│   ├── train_dpo.py
│   ├── reward_model.py
│   └── train_reward_model.py
├── rl/                      # Reinforcement learning modules
│   ├── rl_trainer.py
│   ├── rl_utils.py
│   └── src_rl/              # Full src/rl directory
├── lora/                    # LoRA fine-tuning
│   ├── lora_trainer.py
│   ├── training_manager.py
│   ├── expert_trainer.py
│   ├── pdf_fine_tuner.py
│   ├── pdf_trainer_complete.py
│   ├── sft_trainer.py
│   └── sft_train.py
├── configs/                 # Training configs
│   ├── rlhf.yaml
│   ├── rlhf_sft.yaml
│   ├── ppo_train.yaml
│   ├── dpo_train.yaml
│   ├── reward_model.yaml
│   ├── lora_sft.yaml
│   ├── lora_mamba.yaml
│   ├── lora_transformer.yaml
│   ├── rl_train.yaml
│   └── rl_config.yaml
├── tests/                   # Related tests
├── docs/                    # Related documentation
├── scripts/                 # Training scripts
└── README.md
```

## When to Restore

These files should be restored when:

1. Phase 4+ training infrastructure is approved
2. Data pipelines are validated
3. GPU resources are allocated
4. Human preference data is collected (for RLHF)

## Restoration

To restore a file:
```bash
mv _archive/premature_training/rlhf/rlhf_pipeline.py src/training/
```

## DO NOT

- Import these files from inference code
- Run any training without explicit approval
- Delete this archive without backup
