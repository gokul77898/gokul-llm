"""Tests for Transfer Learning training pipeline"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from src.common import load_config, seed_everything
from src.data import create_sample_data, load_transfer_dataset


@pytest.fixture(scope="module")
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def sample_data(temp_dir):
    """Create sample data"""
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(exist_ok=True)
    create_sample_data(str(data_dir))
    return data_dir


def test_transfer_data_loading(sample_data):
    """Test transfer learning data loading"""
    config = load_config("configs/transfer_train.yaml")
    config.data.train_file = str(sample_data / "legal_train.jsonl")
    config.data.val_file = str(sample_data / "legal_val.jsonl")
    config.data.max_samples = 10
    
    train_dataset, val_dataset = load_transfer_dataset(config)
    
    assert len(train_dataset) > 0, "Train dataset should not be empty"
    assert len(val_dataset) > 0, "Val dataset should not be empty"
    
    # Check data format
    sample = train_dataset[0]
    assert 'text' in sample
    assert 'label' in sample


def test_transfer_model_training_step(sample_data):
    """Test transfer model training for one step"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    seed_everything(42)
    device = torch.device("cpu")
    
    # Load data
    config = load_config("configs/transfer_train.yaml")
    config.data.train_file = str(sample_data / "legal_train.jsonl")
    config.data.val_file = str(sample_data / "legal_val.jsonl")
    config.data.max_samples = 5
    
    train_dataset, _ = load_transfer_dataset(config)
    
    # Create model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    ).to(device)
    
    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters() if p.requires_grad]
    
    # Training step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    # Create a batch
    sample = train_dataset[0]
    encoding = tokenizer(
        sample['text'],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    labels = torch.tensor([sample['label']]).to(device)
    
    # Forward and backward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    
    # Check parameters changed
    params_changed = False
    for initial, current in zip(initial_params, [p for p in model.parameters() if p.requires_grad]):
        if not torch.equal(initial, current):
            params_changed = True
            break
    
    assert params_changed, "Model parameters should change after training step"
    assert isinstance(loss.item(), float), "Loss should be a float"


def test_transfer_evaluation_metrics(sample_data):
    """Test evaluation metrics calculation"""
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    
    # Create dummy predictions
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 2])
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1"
    assert isinstance(acc, (float, np.floating)), "Accuracy should be float"
    assert isinstance(f1, (float, np.floating)), "F1 should be float"
