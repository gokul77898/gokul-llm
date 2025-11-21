# Legal AI System - Phase 1

A comprehensive machine learning system for legal document processing, combining multiple state-of-the-art architectures:

## 🏗️ Architecture Overview

### 1. **Mamba Architecture** 
Custom transformer-based model with hierarchical attention for processing long legal documents.

### 2. **Transfer Architecture**
Fine-tuned pre-trained models (BERT/GPT-2) specialized for legal domain tasks.

### 3. **RAG System**
Retrieval-Augmented Generation using LangChain for enhanced context-aware responses.

### 4. **Reinforcement Learning**
PPO/DQN-based optimization for improving model outputs based on reward feedback.

## 📁 Project Structure

```
MARK/
├── src/
│   ├── mamba/              # Mamba architecture implementation
│   ├── transfer/           # Transfer learning models
│   ├── rag/                # RAG system components
│   ├── rl/                 # Reinforcement learning setup
│   ├── utils/              # Shared utilities
│   └── data/               # Data processing modules
├── tests/                  # Comprehensive test suite
├── configs/                # Configuration files
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/                # Training and evaluation scripts
└── data/                   # Data directory (gitignored)
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spacy model
python -m spacy download en_core_web_sm
```

### Training

```bash
# Train Mamba model
python scripts/train_mamba.py --config configs/mamba_config.yaml

# Fine-tune Transfer model
python scripts/train_transfer.py --config configs/transfer_config.yaml

# Train RAG system
python scripts/train_rag.py --config configs/rag_config.yaml

# Train with RL
python scripts/train_rl.py --config configs/rl_config.yaml
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific component tests
pytest tests/test_mamba.py -v
pytest tests/test_transfer.py -v
pytest tests/test_rag.py -v
pytest tests/test_rl.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Features

### Mamba Architecture
- ✅ Custom hierarchical attention mechanism
- ✅ Sliding window processing for long documents
- ✅ Efficient memory management with padding/masking
- ✅ Positional encodings for document structure

### Transfer Architecture
- ✅ Pre-trained model fine-tuning (BERT/GPT-2)
- ✅ Legal-specific tokenization
- ✅ Named Entity Recognition
- ✅ Document classification and summarization

### RAG System
- ✅ FAISS/ChromaDB vector stores
- ✅ LangChain integration
- ✅ Dynamic document retrieval
- ✅ Context-augmented generation

### Reinforcement Learning
- ✅ PPO and DQN implementations
- ✅ Custom reward functions for legal tasks
- ✅ Multi-agent support
- ✅ Reward-based fine-tuning

## 📝 Configuration

Edit YAML files in `configs/` to customize:
- Model hyperparameters
- Training settings
- Data paths
- Evaluation metrics

## 🔍 Monitoring

Training progress is logged to:
- TensorBoard: `tensorboard --logdir runs/`
- Weights & Biases: Check your W&B dashboard
- Console: Real-time progress bars

## 📚 Documentation

See `docs/` for detailed documentation on:
- Architecture designs
- API references
- Training guides
- Best practices

## 🤝 Contributing

This is a research and development project. For questions or issues, please refer to the project documentation.

## 📄 License

MIT License - See LICENSE file for details
