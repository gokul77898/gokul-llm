# Training Flow Documentation

## Overview

MARK provides a unified training infrastructure that supports training individual components or running the full end-to-end pipeline.

## Training Orchestrator

### Command-Line Interface

```bash
python -m src.training.orchestrator train --target=<target> [options]
```

### Available Targets

1. **mamba** - Train Mamba long-document model
2. **transformer** - Train BERT-based transformer
3. **rag** - Build RAG FAISS index
4. **rl** - Train RL policy
5. **full-pipeline** - Train all components sequentially

### Options

- `--config PATH` - Path to custom config file
- `--device DEVICE` - Device to use (cpu/cuda)
- `--resume` - Resume from checkpoint
- `--evaluate` - Run evaluation (RAG only)

### Examples

```bash
# Train Mamba model
python -m src.training.orchestrator train --target=mamba --device=cuda

# Train with custom config
python -m src.training.orchestrator train --target=transformer --config=my_config.yaml

# Resume training
python -m src.training.orchestrator train --target=rl --resume

# Train full pipeline
python -m src.training.orchestrator train --target=full-pipeline --device=cuda
```

## Individual Training Scripts

### 1. Mamba Training

**Script**: `src/scripts/train_mamba.py`

**Command**:
```bash
python -m src.scripts.train_mamba --config configs/mamba_train.yaml
```

**Configuration** (`configs/mamba_train.yaml`):
```yaml
model:
  vocab_size: 5000
  d_model: 256
  n_heads: 8
  n_layers: 4
  max_length: 512

training:
  batch_size: 8
  num_epochs: 10
  learning_rate: 0.0001
  mixed_precision: true
```

**Output**:
- Checkpoints: `checkpoints/mamba/checkpoint_epoch_*.pt`
- Best model: `checkpoints/mamba/best_model.pt`
- Vocabulary: `checkpoints/mamba/vocab.json`
- Logs: `logs/mamba/train.log`

### 2. Transformer Training

**Script**: `src/scripts/train_transfer.py`

**Command**:
```bash
python -m src.scripts.train_transfer --config configs/transfer_train.yaml
```

**Configuration** (`configs/transfer_train.yaml`):
```yaml
model:
  base_model: "bert-base-uncased"
  num_labels: 3

training:
  batch_size: 16
  num_epochs: 5
  learning_rate: 0.00005
```

**Output**:
- Checkpoints: `checkpoints/transfer/checkpoint_epoch_*.pt`
- Best model: `checkpoints/transfer/best_model.pt`
- Logs: `logs/transfer/train.log`

### 3. RAG Index Building

**Script**: `src/scripts/train_rag_indexer.py`

**Command**:
```bash
python -m src.scripts.train_rag_indexer --config configs/rag_indexer.yaml --evaluate
```

**Configuration** (`configs/rag_indexer.yaml`):
```yaml
model:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dim: 384

data:
  documents_file: "data/documents.jsonl"
```

**Output**:
- FAISS index: `checkpoints/rag/faiss.index`
- Metadata: `checkpoints/rag/metadata.json`
- Evaluation metrics in logs

### 4. RL Training

**Script**: `src/scripts/train_rl.py`

**Command**:
```bash
python -m src.scripts.train_rl --config configs/rl_train.yaml
```

**Configuration** (`configs/rl_train.yaml`):
```yaml
model:
  policy_model: "ppo"
  hidden_dim: 256

training:
  total_timesteps: 10000
  learning_rate: 0.0003
```

**Output**:
- Checkpoints: `checkpoints/rl/checkpoint_step_*.pt`
- Final model: `checkpoints/rl/final_model.pt`
- Logs: `logs/rl/train.log`

## Full Pipeline

### Command

```bash
python -m src.pipelines.full_pipeline --device=cuda
```

### Pipeline Stages

1. **Data Preparation**
   - Creates sample data if needed
   - Validates data files

2. **Train Mamba**
   - Loads Mamba config
   - Trains model
   - Saves checkpoint

3. **Train Transformer**
   - Loads transformer config
   - Fine-tunes BERT
   - Saves checkpoint

4. **Build RAG Index**
   - Loads documents
   - Generates embeddings
   - Builds FAISS index

5. **Train RL**
   - Creates environment
   - Trains PPO policy
   - Saves policy

6. **Export Models**
   - Validates all checkpoints
   - Creates export directory
   - Copies final models

### Options

```bash
python -m src.pipelines.full_pipeline \
  --device cuda \
  --skip-data-prep \
  --stages train_mamba build_rag \
  --output-dir my_pipeline_output
```

## RLHF Pipeline

### Overview

RLHF (Reinforcement Learning from Human Feedback) pipeline for fine-tuning models with human preferences.

### Command

```bash
python -m src.training.rlhf_pipeline \
  --base-model mamba \
  --config configs/rl_train.yaml \
  --output-dir checkpoints/rlhf
```

### Pipeline Steps

1. **Supervised Fine-Tuning (SFT)**
   - Loads base model
   - Fine-tunes on supervised data
   - Saves SFT checkpoint

2. **Reward Model Training**
   - Trains reward scorer
   - Uses preference data
   - Saves reward model

3. **PPO Training**
   - Optimizes policy with RL
   - Uses reward model for scoring
   - Saves trained policy

4. **Evaluation**
   - Computes quality metrics
   - Validates alignment
   - Generates report

5. **Export**
   - Saves final RLHF model
   - Includes all metadata

### Options

```bash
--skip-sft          # Skip SFT if already done
--skip-reward       # Skip reward model training
--num-rl-steps N    # Override RL training steps
```

### Output

- SFT model: `checkpoints/rlhf/sft_model.pt`
- Reward model: `checkpoints/rlhf/reward_model.pt`
- PPO policy: `checkpoints/rlhf/ppo_policy.pt`
- Final model: `checkpoints/rlhf/rlhf_final_model.pt`

## Training Best Practices

### 1. Data Preparation

- Ensure data files exist before training
- Validate data format
- Use appropriate sample sizes for experimentation

### 2. Hyperparameter Tuning

- Start with default configs
- Adjust batch size based on memory
- Use learning rate warmup
- Enable mixed precision for speed

### 3. Monitoring

- Check logs regularly
- Monitor loss curves
- Validate on held-out data
- Track GPU/CPU usage

### 4. Checkpoint Management

- Save checkpoints frequently
- Keep best models separate
- Version control configs
- Document experiments

### 5. Debugging

- Start with small datasets
- Use CPU for debugging
- Enable verbose logging
- Test individual components

## Common Issues

### Out of Memory

**Solution**:
- Reduce batch size
- Enable gradient accumulation
- Use smaller model
- Use mixed precision

### Slow Training

**Solution**:
- Use GPU if available
- Increase batch size
- Reduce logging frequency
- Use DataLoader with multiple workers

### Poor Performance

**Solution**:
- Increase training epochs
- Adjust learning rate
- Check data quality
- Try different architectures

### Checkpoint Loading Fails

**Solution**:
- Verify checkpoint exists
- Check model architecture matches
- Ensure compatible PyTorch version
- Start from scratch if corrupted

## Performance Optimization

### GPU Training

```yaml
system:
  device: "cuda"

training:
  mixed_precision: true
  batch_size: 32  # Increase for GPU
  num_workers: 4
```

### Distributed Training

(To be implemented)

### Gradient Accumulation

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size: 32
```

## Monitoring and Logging

### Console Logs

All training scripts log to console with timestamps and progress.

### File Logs

Logs saved to `logs/<component>/train.log`

### WandB Integration

(Optional) Enable in config:
```yaml
system:
  use_wandb: true
  wandb_project: "mark-training"
```

## Resume Training

All training scripts support resuming from checkpoints:

```bash
python -m src.scripts.train_mamba \
  --config configs/mamba_train.yaml \
  --resume
```

The script automatically finds the latest checkpoint and resumes training.

## Evaluation

### During Training

- Validation loss computed every `eval_steps`
- Best model saved based on validation metric

### After Training

```bash
# Evaluate RAG system
python -m src.scripts.train_rag_indexer \
  --config configs/rag_indexer.yaml \
  --evaluate
```

### Custom Evaluation

Use the evaluation modules:
```python
from src.rag.eval import evaluate_rag_pipeline

metrics = evaluate_rag_pipeline(indexer, eval_file, top_k_values=[1, 3, 5])
print(metrics)
```

## Export and Deployment

After training, models are ready for deployment:

1. **API Server**:
   ```bash
   python -m src.api.main
   ```

2. **LangChain Integration**:
   ```python
   from src.integrations import MARKLangChainGraph
   graph = MARKLangChainGraph(llm_model="mamba")
   ```

3. **Direct Loading**:
   ```python
   from src.core import load_model
   model, tokenizer, device = load_model("mamba")
   ```
