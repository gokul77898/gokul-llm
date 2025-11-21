"""Tests for Mamba training pipeline"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from src.common import load_config, seed_everything, get_device
from src.data import load_mamba_dataset, create_sample_data
from src.mamba import DocumentTokenizer, MambaModel


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


def test_mamba_training_single_step(sample_data, temp_dir):
    """Test Mamba training for one step"""
    # Load config
    config = load_config("configs/mamba_train.yaml")
    config.data.train_file = str(sample_data / "train.txt")
    config.data.val_file = str(sample_data / "val.txt")
    config.data.max_samples = 10
    config.training.batch_size = 2
    config.training.num_epochs = 1
    config.paths.checkpoint_dir = str(Path(temp_dir) / "checkpoints")
    
    seed_everything(42)
    device = torch.device("cpu")
    
    # Load data
    train_dataset, val_dataset = load_mamba_dataset(config)
    assert len(train_dataset) > 0
    
    # Create tokenizer
    tokenizer = DocumentTokenizer(vocab_size=100, max_length=64)
    tokenizer.build_vocab([item['text'] for item in train_dataset][:5])
    
    # Create model
    model = MambaModel(
        vocab_size=len(tokenizer.token2id),
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_length=64,
        dropout=0.1,
        num_classes=3
    ).to(device)
    
    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    
    # Training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create a batch
    text = train_dataset[0]['text']
    encoding = tokenizer.encode(text, return_tensors=False)
    input_ids = torch.tensor([encoding['input_ids'][:64]]).to(device)
    attention_mask = torch.tensor([encoding['attention_mask'][:64]]).to(device)
    
    # Forward and backward
    outputs = model(input_ids, attention_mask, task="generation")
    logits = outputs['logits']
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_ids.view(-1)
    )
    
    loss.backward()
    optimizer.step()
    
    # Check parameters changed
    params_changed = False
    for initial, current in zip(initial_params, model.parameters()):
        if not torch.equal(initial, current):
            params_changed = True
            break
    
    assert params_changed, "Model parameters should change after training step"
    assert loss.item() > 0, "Loss should be positive"


def test_mamba_checkpoint_creation(sample_data, temp_dir):
    """Test checkpoint creation"""
    from src.common.checkpoints import save_checkpoint, load_checkpoint
    
    checkpoint_dir = Path(temp_dir) / "test_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create dummy model
    model = torch.nn.Linear(10, 10)
    checkpoint_path = checkpoint_dir / "test.pt"
    
    # Save checkpoint
    save_checkpoint(
        {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'test_value': 42,
        },
        str(checkpoint_path)
    )
    
    # Check file exists
    assert checkpoint_path.exists(), "Checkpoint file should be created"
    
    # Load checkpoint
    loaded = load_checkpoint(str(checkpoint_path))
    assert loaded is not None
    assert loaded['test_value'] == 42
    assert 'model_state_dict' in loaded


def test_mamba_full_pipeline(sample_data, temp_dir):
    """Test full Mamba training pipeline"""
    import subprocess
    import sys
    
    # Create minimal config
    config_path = Path(temp_dir) / "test_config.yaml"
    config_content = f"""
model:
  vocab_size: 100
  d_model: 64
  n_heads: 4
  n_layers: 2
  d_ff: 128
  max_length: 64
  chunk_size: 32
  chunk_overlap: 8
  dropout: 0.1

training:
  batch_size: 2
  num_epochs: 1
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_steps: 10
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  eval_steps: 100
  save_steps: 100
  logging_steps: 10
  mixed_precision: false
  num_workers: 0

data:
  train_file: "{str(sample_data / 'train.txt')}"
  val_file: "{str(sample_data / 'val.txt')}"
  max_samples: 10
  min_doc_length: 50

paths:
  checkpoint_dir: "{str(Path(temp_dir) / 'pipeline_checkpoints')}"
  log_dir: "{str(Path(temp_dir) / 'logs')}"

system:
  device: "cpu"
  seed: 42
  use_wandb: false
"""
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Run training script
    result = subprocess.run(
        [sys.executable, "-m", "src.scripts.train_mamba", "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Check training completed
    assert result.returncode == 0, f"Training failed: {result.stderr}"
    
    # Check checkpoint was created
    checkpoint_dir = Path(temp_dir) / 'pipeline_checkpoints'
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints) > 0, "At least one checkpoint should be created"
