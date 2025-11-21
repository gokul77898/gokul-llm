# MARK Phase 2: Training Pipeline

Complete production-ready training pipeline for the MARK (Modular Architecture for Reinforced Knowledge) project.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Mamba Training (1 epoch)

```bash
python examples/quickstart_mamba.py
```

**Output:** `checkpoints/mamba/quickstart_checkpoint.pt`

### 2. Transfer Learning (1 epoch)

```bash
python examples/quickstart_transfer.py
```

**Output:** `checkpoints/transfer/quickstart_checkpoint.pt`

### 3. RAG Indexer

```bash
python examples/quickstart_rag.py
```

**Output:** `checkpoints/rag/quickstart.index`

## Full Training

### Mamba Model Training

```bash
# Train with default config
python -m src.scripts.train_mamba --config configs/mamba_train.yaml

# Train on CPU
python -m src.scripts.train_mamba --config configs/mamba_train.yaml --device cpu

# Resume from checkpoint
python -m src.scripts.train_mamba --config configs/mamba_train.yaml --resume
```

### Transfer Learning Training

```bash
# Train with default config
python -m src.scripts.train_transfer --config configs/transfer_train.yaml

# Train on specific device
python -m src.scripts.train_transfer --config configs/transfer_train.yaml --device cuda:0
```

### RAG Indexer

```bash
# Build index
python -m src.scripts.train_rag_indexer --config configs/rag_indexer.yaml

# Build and evaluate
python -m src.scripts.train_rag_indexer --config configs/rag_indexer.yaml --evaluate
```

### RL Training

```bash
# Train RL agent
python -m src.scripts.train_rl --config configs/rl_train.yaml

# Train on CPU
python -m src.scripts.train_rl --config configs/rl_train.yaml --device cpu
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_train_mamba.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Skip slow tests
pytest tests/ -m "not slow"
```

## Project Structure

```
MARK/
├── configs/                    # Training configurations
│   ├── mamba_train.yaml
│   ├── transfer_train.yaml
│   ├── rag_indexer.yaml
│   └── rl_train.yaml
├── src/
│   ├── common/                 # Common utilities
│   │   ├── config.py          # Config loading
│   │   ├── utils.py           # Helper functions
│   │   └── checkpoints.py     # Checkpoint management
│   ├── data/                   # Data loading
│   │   ├── datasets.py        # Dataset loaders
│   │   └── collate.py         # Collate functions
│   ├── mamba/                  # Mamba architecture
│   ├── transfer/               # Transfer learning
│   ├── rag/                    # RAG system
│   │   ├── indexer.py         # FAISS indexer
│   │   └── eval.py            # Evaluation metrics
│   ├── rl/                     # RL system
│   │   ├── env.py             # Gymnasium environment
│   │   ├── ppo.py             # PPO implementation
│   │   └── trainer.py         # RL trainer
│   └── scripts/                # Training scripts
│       ├── train_mamba.py
│       ├── train_transfer.py
│       ├── train_rag_indexer.py
│       └── train_rl.py
├── tests/                      # Test suite
├── examples/                   # Quickstart examples
└── checkpoints/                # Saved checkpoints
```

## Configuration

All training configurations are in `configs/` directory. Key parameters:

- **batch_size**: Training batch size
- **num_epochs**: Number of training epochs
- **learning_rate**: Learning rate
- **device**: Training device (cpu/cuda)
- **checkpoint_dir**: Where to save checkpoints
- **mixed_precision**: Use automatic mixed precision

## Checkpoints

Checkpoints are saved in `checkpoints/<component>/` with the following structure:

- `checkpoint_epoch_N.pt`: Checkpoint after epoch N
- `best_model.pt`: Best model based on validation metric
- Checkpoints include: model state, optimizer state, scheduler state, epoch number, RNG states

## Logging

Logs are saved in `logs/<component>/train.log` and printed to console.

## GPU Support

All training scripts support both CPU and GPU training. GPU is automatically used if available:

```bash
# Force CPU
python -m src.scripts.train_mamba --config configs/mamba_train.yaml --device cpu

# Use specific GPU
python -m src.scripts.train_mamba --config configs/mamba_train.yaml --device cuda:0
```

## Performance Optimization

- **Mixed Precision**: Set `mixed_precision: true` in config (requires CUDA)
- **Gradient Accumulation**: Set `gradient_accumulation_steps` > 1
- **Num Workers**: Adjust `num_workers` for data loading
- **Batch Size**: Increase for better GPU utilization

## Troubleshooting

**Out of Memory:**
- Reduce batch_size
- Enable gradient_accumulation_steps
- Use mixed_precision

**Slow Training:**
- Increase num_workers
- Use GPU if available
- Enable mixed_precision

**Tests Failing:**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Run with verbose: `pytest tests/ -v -s`

## License

MIT License - see LICENSE file
