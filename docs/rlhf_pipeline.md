# RLHF Pipeline Documentation

## Overview

The RLHF (Reinforcement Learning from Human Feedback) pipeline fine-tunes models using human preferences through three stages:

1. **Supervised Fine-Tuning (SFT)**: Fine-tune base model on demonstrations
2. **Reward Model Training**: Train a model to predict human preferences
3. **PPO/DPO Training**: Optimize policy using RL

## Data Format

### SFT Data (data/rl_train.jsonl)
```json
{"document": "text to fine-tune on", "target": "expected output"}
```

### Reward Model Data
Pairwise preferences or scalar rewards.

## Commands

### 1. Supervised Fine-Tuning
```bash
python -m src.training.rlhf_sft --config configs/rlhf_sft.yaml
```

**Output**: `checkpoints/rlhf/sft/sft_final.pt`

### 2. Reward Model Training
```bash
python -m src.training.train_reward_model --config configs/reward_model.yaml
```

**Output**: `checkpoints/rlhf/reward/reward_model_final.pt`

### 3. PPO Training
```bash
python -m src.training.train_ppo --config configs/ppo_train.yaml
```

**Output**: `checkpoints/rlhf/ppo/ppo_final.pt`

### 4. DPO Training (Alternative)
```bash
python -m src.training.train_dpo --config configs/dpo_train.yaml
```

**Output**: `checkpoints/rlhf/dpo/dpo_final.pt`

## Configuration

### SFT Config (configs/rlhf_sft.yaml)
- `base_model`: Model to fine-tune (mamba/transformer)
- `batch_size`: Training batch size
- `num_epochs`: Training epochs
- `learning_rate`: Learning rate

### Reward Model Config (configs/reward_model.yaml)
- `input_dim`: Input feature dimension
- `hidden_dim`: Hidden layer size
- `num_layers`: Number of layers

### PPO Config (configs/ppo_train.yaml)
- `obs_dim`: Observation dimension
- `action_dim`: Action space size
- `clip_range`: PPO clip parameter
- `gamma`: Discount factor
- `gae_lambda`: GAE lambda

## Full Pipeline

Run all stages sequentially:
```bash
# 1. SFT
python -m src.training.rlhf_sft --config configs/rlhf_sft.yaml

# 2. Reward Model
python -m src.training.train_reward_model --config configs/reward_model.yaml

# 3. PPO
python -m src.training.train_ppo --config configs/ppo_train.yaml
```

## Export

Final model checkpoint: `checkpoints/rlhf/final_rl_model.pt`

To use:
```python
from src.core import load_model
model, tokenizer, device = load_model("rl_trained")
```

## Testing

```bash
pytest tests/test_phase4/ -v
```

## Reward Shaping

Rewards are normalized using:
- Mean subtraction
- Standard deviation scaling
- Clipping to prevent outliers

## Hyperparameters

Key hyperparameters for tuning:
- **Learning rate**: 1e-4 to 5e-4 for SFT, 3e-4 for PPO
- **Batch size**: 4-16 for SFT, 16-64 for PPO
- **Clip range**: 0.1-0.3 for PPO
- **GAE lambda**: 0.9-0.99
- **Discount (gamma)**: 0.95-0.99

## Monitoring

Training logs are saved to:
- `logs/rlhf/sft/sft.log`
- `logs/rlhf/reward/reward.log`
- `logs/rlhf/ppo/ppo.log`

Monitor:
- Loss curves
- Reward trends
- Policy KL divergence (for PPO)

## Troubleshooting

**Issue**: Training unstable
**Solution**: Reduce learning rate, increase batch size

**Issue**: Poor rewards
**Solution**: Check reward model quality, adjust reward shaping

**Issue**: Out of memory
**Solution**: Reduce batch size, use gradient accumulation
