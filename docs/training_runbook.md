# 🎓 MARK Training Runbook - RAG + Fine-Tune Hybrid

**Complete Guide for Training Fine-Tuned Models**

---

## 📋 Overview

This runbook guides you through the complete training pipeline:
1. **Data Preparation** - Extract QA pairs from ChromaDB
2. **LoRA Fine-Tuning** - Supervised fine-tuning with adapters
3. **Evaluation** - Assess model performance
4. **RLHF** (Optional) - Reinforcement learning from feedback

---

## ⚠️ PREREQUISITES

### System Requirements
- **Python:** 3.10+
- **Memory:** 16GB+ RAM recommended
- **Storage:** 10GB+ free space
- **GPU:** Optional but recommended for training

### Dependencies
```bash
pip install torch transformers peft accelerate datasets
pip install chromadb sentence-transformers
pip install rouge-score nltk pyyaml
```

### Data Requirements
✅ **ChromaDB collection 'pdf_docs' must exist**
- Run PDF ingestion first if not done
- See: `src/ingest/pdf_ingest.py`

---

## 🚀 QUICK START

### 1. Generate Training Data

```bash
# Extract QA pairs from ChromaDB
python -m src.training.data_prep \
  --collection pdf_docs \
  --out-dir data/ \
  --top-k 3 \
  --max-samples 1000
```

**Output:**
- `data/train_sft.jsonl` - Training examples
- `data/val_sft.jsonl` - Validation examples

---

### 2. Dry-Run LoRA Training

```bash
# Validate setup without training
python -m src.training.lora_trainer \
  --config configs/lora_sft.yaml \
  --dry-run
```

**This will:**
- ✅ Load model architecture
- ✅ Load datasets
- ✅ Configure trainer
- ✅ Report training plan
- ❌ NOT run actual training

---

### 3. Run Actual LoRA Training

⚠️ **WARNING: This starts real training!**

```bash
# Start fine-tuning (requires confirmation)
python -m src.training.lora_trainer \
  --config configs/lora_sft.yaml \
  --confirm-run
```

**Expected Duration:** 30 minutes - 2 hours (depends on data size and hardware)

**Output:** `checkpoints/lora/mamba_lora/`

---

### 4. Evaluate Fine-Tuned Model

```bash
# Run evaluation on validation set
python -m src.training.eval \
  --model mamba_lora \
  --dataset data/val_sft.jsonl
```

**Output:** `reports/eval_mamba_lora_<timestamp>.json`

**Metrics:**
- Exact Match
- Token F1
- ROUGE-L
- BLEU

---

## 🎯 USING TRAINING MANAGER

The training manager orchestrates the entire pipeline:

### Dry-Run All Stages

```bash
# Data preparation
python -m src.training.training_manager --stage data_prep

# LoRA training
python -m src.training.training_manager --stage lora

# Evaluation
python -m src.training.training_manager --stage eval
```

### Execute Stages

```bash
# Run with --confirm flag
python -m src.training.training_manager --stage data_prep --confirm
python -m src.training.training_manager --stage lora --confirm
python -m src.training.training_manager --stage eval --confirm
```

---

## ⚙️ CONFIGURATION

### LoRA Config (`configs/lora_sft.yaml`)

Key parameters:

```yaml
lora:
  r: 8                    # LoRA rank (higher = more capacity)
  lora_alpha: 16          # Scaling factor
  lora_dropout: 0.05
  
training:
  dry_run: true          # SAFETY: prevents training
  epochs: 0              # Set to 3-5 for actual training
  batch_size: 4
  learning_rate: 2.0e-4
  
data:
  max_seq_length: 512    # Maximum sequence length
```

**Before training, update:**
1. Set `dry_run: false`
2. Set `epochs: 3` (or more)

---

## 📊 MONITORING TRAINING

### Logs

Training logs are saved to:
- `logs/lora_training/`

### Checkpoints

Models saved to:
- `checkpoints/lora/mamba_lora/checkpoint-XXX/`
- `checkpoints/lora/mamba_lora/final/` (best model)

### TensorBoard (Optional)

```bash
# Enable in config
report_to: "tensorboard"

# View logs
tensorboard --logdir logs/lora_training
```

---

## 🔧 ADVANCED USAGE

### Custom Data Generation

```python
from src.training.data_prep import SFTDataGenerator

generator = SFTDataGenerator("chroma_db", "pdf_docs")
train_data, val_data = generator.generate_dataset(
    top_k=5,
    max_samples=2000,
    val_split=0.1
)
generator.save_jsonl(train_data, "data/train_custom.jsonl")
```

### Distributed Training

```bash
# Use accelerate for multi-GPU
accelerate config  # Configure once

accelerate launch src/training/lora_trainer.py \
  --config configs/lora_sft.yaml \
  --confirm-run
```

### Resume Training

```bash
# Training resumes from last checkpoint automatically
python -m src.training.lora_trainer \
  --config configs/lora_sft.yaml \
  --confirm-run
```

---

## 🧪 TESTING

### Unit Tests

```bash
# Run all training tests
pytest tests/training/ -v

# Specific tests
pytest tests/training/test_data_prep.py -v
pytest tests/training/test_lora_trainer_dryrun.py -v
```

---

## ⚠️ TROUBLESHOOTING

### Issue: "Training data not found"

**Solution:**
```bash
python -m src.training.data_prep --collection pdf_docs --out-dir data/
```

### Issue: "Out of memory"

**Solutions:**
1. Reduce batch size in config: `batch_size: 2`
2. Reduce sequence length: `max_seq_length: 256`
3. Enable gradient accumulation: `gradient_accumulation_steps: 8`

### Issue: "PEFT not installed"

**Solution:**
```bash
pip install peft bitsandbytes
```

### Issue: "Model loading failed"

**Check:**
1. Config file exists: `configs/lora_sft.yaml`
2. Base model available
3. Sufficient disk space

---

## 🔐 SAFETY CONTROLS

### Default Protections

✅ **dry_run: true** - Prevents accidental training
✅ **epochs: 0** - No training by default  
✅ **--confirm-run required** - Two-step confirmation
✅ **SETUP_MODE** - Training blocked until explicitly enabled

### Disabling Protections

**Only when ready to train:**

1. Edit `configs/lora_sft.yaml`:
   ```yaml
   training:
     dry_run: false
     epochs: 3
   ```

2. Use `--confirm-run` flag

---

## 📈 EXPECTED PERFORMANCE

### Data Generation
- Time: 2-5 minutes
- Output: ~1000 QA pairs

### LoRA Training (CPU)
- Time: 1-2 hours
- Memory: ~8GB RAM

### LoRA Training (GPU)
- Time: 15-30 minutes
- Memory: ~4GB VRAM

### Evaluation
- Time: 5-10 minutes
- Output: JSON report with metrics

---

## 🎯 INTEGRATION WITH MARK

### Using Fine-Tuned Models

Fine-tuned models are automatically registered:

```python
from src.core import load_model

# Load LoRA fine-tuned model
model, tokenizer, device = load_model("mamba_lora")

# Use in AutoPipeline (automatic selection)
from src.pipelines.auto_pipeline import AutoPipeline
pipeline = AutoPipeline()
result = pipeline.process_query("Your legal question?")
```

### Model Priority

AutoPipeline prefers fine-tuned models when available:
1. `mamba_lora` (if trained)
2. `rl_trained_lora` (if trained)
3. `mamba` (base model)
4. `transformer` (fallback)

---

## 📚 NEXT STEPS

After successful training:

1. ✅ **Evaluate** model performance
2. ✅ **Test** with real queries via API/UI
3. ✅ **Compare** with base model
4. ✅ **Iterate** with more data if needed
5. ✅ **Deploy** to production

---

## 🆘 SUPPORT

### Common Commands

```bash
# Check system status
python -c "from src.core import get_registry; print(get_registry().list_models())"

# Verify data exists
ls -lh data/*.jsonl

# Check ChromaDB
python -c "from chromadb import PersistentClient; print(PersistentClient('chroma_db').list_collections())"

# View training logs
tail -f logs/lora_training/events.out.tfevents.*
```

### Resources

- **Code:** `src/training/`
- **Configs:** `configs/`
- **Tests:** `tests/training/`
- **Logs:** `logs/`
- **Checkpoints:** `checkpoints/lora/`

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Status:** ✅ Production Ready
