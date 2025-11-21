"""
Unit tests for LoRA trainer (dry-run mode only).
"""

import pytest
from pathlib import Path

from src.training.lora_trainer import LoRATrainer


def test_trainer_initialization():
    """Test trainer can be initialized."""
    config_path = "configs/lora_sft.yaml"
    
    if not Path(config_path).exists():
        pytest.skip(f"Config not found: {config_path}")
    
    trainer = LoRATrainer(config_path)
    
    assert trainer.config is not None
    assert trainer.device in ['cpu', 'cuda', 'mps']


def test_dry_run_mode():
    """Test dry-run validation."""
    config_path = "configs/lora_sft.yaml"
    
    if not Path(config_path).exists():
        pytest.skip(f"Config not found: {config_path}")
    
    trainer = LoRATrainer(config_path)
    
    # Dry run should not raise errors
    trainer.dry_run()


def test_config_safety_defaults():
    """Test that config has safe defaults."""
    config_path = "configs/lora_sft.yaml"
    
    if not Path(config_path).exists():
        pytest.skip(f"Config not found: {config_path}")
    
    from src.training.utils import load_config
    config = load_config(config_path)
    
    # Safety checks
    assert config['training']['dry_run'] == True, "dry_run should be True by default"
    assert config['training']['epochs'] == 0, "epochs should be 0 by default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
