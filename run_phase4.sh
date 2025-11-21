#!/bin/bash
# Phase 4 RLHF Pipeline Runner

set -e

echo "======================================"
echo "PHASE 4: RLHF PIPELINE"
echo "======================================"

echo "1. Running SFT..."
python3.10 -m src.training.rlhf_sft --config configs/rlhf_sft.yaml

echo "2. Training Reward Model..."
python3.10 -m src.training.train_reward_model --config configs/reward_model.yaml

echo "3. Running PPO..."
python3.10 -m src.training.train_ppo --config configs/ppo_train.yaml

echo "======================================"
echo "RLHF Pipeline Complete!"
echo "Final checkpoint: checkpoints/rlhf/ppo/ppo_final.pt"
echo "======================================"
