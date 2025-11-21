# Quick Start Guide - Legal AI System

This guide will help you get started with the Legal AI System in minutes.

## Installation

### 1. Set Up Environment

```bash
# Clone or navigate to the project
cd MARK

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Verify Installation

```bash
# Run quick start examples
python examples/quickstart.py

# Run tests
pytest tests/ -v
```

## Component Overview

### 1. Mamba Architecture (Long Document Processing)

**Purpose**: Process long legal documents with hierarchical attention

**Quick Example**:
```python
from src.mamba.model import MambaModel
from src.mamba.tokenizer import DocumentTokenizer

# Create tokenizer and model
tokenizer = DocumentTokenizer(vocab_size=30000, max_length=512)
tokenizer.build_vocab(["Your training texts here..."])

model = MambaModel(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_layers=6,
    num_heads=8,
    num_classes=5
)

# Process document
text = "Long legal document..."
encoded = tokenizer.encode(text, return_tensors=True)
outputs = model(input_ids=encoded.input_ids, task="classification")
```

**Train Model**:
```bash
python scripts/train_mamba.py --config configs/mamba_config.yaml
```

### 2. Transfer Architecture (Legal-Specific Models)

**Purpose**: Fine-tune pre-trained models for legal tasks

**Quick Example**:
```python
from src.transfer.model import LegalTransferModel, LegalTaskType
from src.transfer.tokenizer import LegalTokenizer

# Create tokenizer and model
tokenizer = LegalTokenizer(base_model="bert-base-uncased")
model = LegalTransferModel(
    model_name="bert-base-uncased",
    task=LegalTaskType.CLASSIFICATION,
    num_labels=3
)

# Process legal text
text = "Smith v. Jones, 123 U.S. 456"
encoded = tokenizer.encode(text, return_tensors="pt")
outputs = model(**encoded)
```

**Train Model**:
```bash
python scripts/train_transfer.py --config configs/transfer_config.yaml
```

### 3. RAG System (Retrieval-Augmented Generation)

**Purpose**: Answer legal questions using document retrieval

**Quick Example**:
```python
from src.rag.pipeline import create_legal_rag_pipeline
from src.rag.document_store import Document

# Create documents
documents = [
    Document("Contracts are legally binding agreements."),
    Document("Tort law covers civil wrongs.")
]

# Create RAG pipeline
pipeline = create_legal_rag_pipeline(
    documents=documents,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    generation_model="gpt2"
)

# Query
result = pipeline.query("What is a contract?", top_k=3)
print(result['answer'])
```

**Setup RAG System**:
```bash
python scripts/train_rag.py --config configs/rag_config.yaml
```

### 4. Reinforcement Learning (Policy Optimization)

**Purpose**: Optimize model outputs through reward feedback

**Quick Example**:
```python
from src.rl.trainer import create_rl_trainer_for_task
from src.mamba.model import MambaModel
from src.mamba.tokenizer import DocumentTokenizer

# Create base model
model = MambaModel(vocab_size=30000, d_model=512, num_layers=4, num_heads=8)
tokenizer = DocumentTokenizer(vocab_size=30000)

# Create RL trainer
trainer = create_rl_trainer_for_task(
    task_type="summarization",
    model=model,
    tokenizer=tokenizer,
    agent_type="ppo",
    total_timesteps=100000
)

# Train
trainer.train(total_timesteps=100000, eval_episodes=10)
```

**Train RL Agent**:
```bash
python scripts/train_rl.py --config configs/rl_config.yaml
```

## Common Workflows

### Workflow 1: Document Classification

1. **Prepare Data**:
```python
from src.utils.data_loader import LegalDataLoader

loader = LegalDataLoader(data_dir="./data")
documents = loader.load_json("legal_docs.json")
train, val, test = loader.split_data(documents)
```

2. **Train Mamba Model**:
```bash
python scripts/train_mamba.py --config configs/mamba_config.yaml --data-path ./data
```

3. **Evaluate**:
```python
from src.utils.metrics import compute_metrics

metrics = compute_metrics(predictions, labels)
print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"F1 Score: {metrics.f1:.4f}")
```

### Workflow 2: Legal Q&A System

1. **Index Documents**:
```python
from src.rag.document_store import FAISSStore, Document

store = FAISSStore()
documents = [Document(text) for text in legal_texts]
store.add_documents(documents)
store.save("./data/document_store")
```

2. **Setup RAG Pipeline**:
```bash
python scripts/train_rag.py --config configs/rag_config.yaml --data-path ./data
```

3. **Query System**:
```python
from src.rag.pipeline import RAGPipeline

# Load pipeline
pipeline.load("./checkpoints/rag")

# Query
result = pipeline.query("What are contract requirements?")
print(result['answer'])
```

### Workflow 3: Model Fine-tuning with RL

1. **Train Base Model**:
```bash
python scripts/train_mamba.py --config configs/mamba_config.yaml
```

2. **Fine-tune with RL**:
```bash
python scripts/train_rl.py --config configs/rl_config.yaml --base-model ./checkpoints/mamba/best_model
```

3. **Evaluate Improvements**:
```python
from src.rl.rewards import RewardCalculator

calculator = RewardCalculator()
metrics = calculator.calculate_summarization_reward(generated, reference, source)
print(f"Total Reward: {metrics.total_reward:.4f}")
```

## Configuration

All components use YAML configuration files in `configs/`:

- `mamba_config.yaml` - Mamba model settings
- `transfer_config.yaml` - Transfer learning settings
- `rag_config.yaml` - RAG system settings
- `rl_config.yaml` - RL training settings

Edit these files to customize:
- Model architecture
- Training hyperparameters
- Data paths
- Evaluation metrics

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Component Tests
```bash
pytest tests/test_mamba.py -v
pytest tests/test_transfer.py -v
pytest tests/test_rag.py -v
pytest tests/test_rl.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Weights & Biases
```python
import wandb
wandb.login()  # Login to W&B
# Training will automatically log to W&B
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `d_model` or `num_layers`
- Enable gradient accumulation

### Slow Training
- Use GPU: Remove `--cpu` flag
- Increase `num_workers` in config
- Use smaller embedding model for RAG

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version (requires 3.8+)
python --version
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory
2. **Read Documentation**: See `docs/` for detailed guides
3. **Customize Models**: Modify configs and architectures
4. **Add Your Data**: Replace sample data with real legal documents
5. **Deploy**: Use trained models in production

## Support

- **Issues**: Check GitHub issues or create new one
- **Documentation**: See `README.md` and `docs/`
- **Examples**: Run `python examples/quickstart.py`

## Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

Happy coding! 🚀
