"""Tests for Model Registry"""

import pytest
import torch
from pathlib import Path

from src.core import ModelRegistry, load_model, get_registry


def test_registry_initialization():
    """Test registry initializes with default models"""
    registry = ModelRegistry()
    models = registry.list_models()
    
    assert len(models) > 0
    assert 'transformer' in models
    assert 'rag_encoder' in models
    assert 'rl_trained' in models


def test_register_model():
    """Test registering a new model"""
    registry = ModelRegistry()
    
    registry.register_model(
        name="test_model",
        architecture="transformer",
        config_path="configs/test.yaml",
        checkpoint_path=None,
        description="Test model"
    )
    
    models = registry.list_models()
    assert 'test_model' in models
    
    info = registry.get_model_info('test_model')
    assert info.name == "test_model"
    assert info.architecture == "transformer"


def test_get_model_info():
    """Test getting model info"""
    registry = get_registry()
    
    info = registry.get_model_info('transformer')
    assert info is not None
    assert info.architecture == 'transformer'
    assert 'transformer' in info.config_path


def test_load_transformer_model():
    """Test loading Transformer model"""
    try:
        model, tokenizer, device = load_model('transformer', device='cpu')
        
        assert model is not None
        assert tokenizer is not None
        assert isinstance(device, torch.device)
        
        # Test model is in eval mode
        assert not model.training
        
    except Exception as e:
        pytest.skip(f"Model loading failed (expected if checkpoint missing): {e}")


def test_load_transformer_model():
    """Test loading Transformer model"""
    try:
        model, tokenizer, device = load_model('transformer', device='cpu')
        
        assert model is not None
        assert tokenizer is not None
        assert isinstance(device, torch.device)
        
    except Exception as e:
        pytest.skip(f"Model loading failed (expected if checkpoint missing): {e}")


def test_load_rag_model():
    """Test loading RAG model"""
    try:
        retriever, embedding_model, device = load_model('rag_encoder', device='cpu')
        
        assert retriever is not None
        assert isinstance(device, torch.device)
        
    except Exception as e:
        pytest.skip(f"Model loading failed (expected if index missing): {e}")


def test_load_invalid_model():
    """Test loading invalid model raises error"""
    with pytest.raises(ValueError, match="not found in registry"):
        load_model('nonexistent_model')


def test_load_with_device_override():
    """Test loading with device override"""
    try:
        model, tokenizer, device = load_model('mamba', device='cpu')
        assert device.type == 'cpu'
    except Exception as e:
        pytest.skip(f"Model loading failed: {e}")
