# ⚠️ TEMPORARY Colab LoRA Validation

## WARNING: THIS IS TEMPORARY CODE

**This directory is for pre-flight validation ONLY.**

**DELETE THIS DIRECTORY before production training.**

## Purpose

Validate LoRA training pipeline on Google Colab T4 GPU using a small model (Qwen2.5-1.5B) before running production training on Qwen2.5-32B.

## Usage on Colab

```python
# In Colab notebook
!git clone <your-repo>
%cd omilos-FINAL-llm-/training/lora_colab_test

# Install dependencies
!pip install transformers peft torch accelerate

# Run training
!python train_colab.py --data_path ../../data/train.jsonl
```

## Expected Runtime

- Model loading: ~2 minutes
- Training (300 steps): ~30-40 minutes
- Validation: ~2 minutes
- **Total**: ~35-45 minutes on T4

## Hardware Requirements

- GPU: T4 (15GB VRAM minimum)
- CUDA required (no CPU fallback)
- FP16 support

## Output

Saves LoRA adapters to `outputs/final_adapter/`

## Post-Training Check

Script automatically compares base vs LoRA outputs and fails if identical.

## Deletion Checklist

Before production training:
- [ ] Verify Colab training worked
- [ ] Delete `training/lora_colab_test/`
- [ ] Verify deletion: `ls training/` should only show `lora_qwen32b/`
