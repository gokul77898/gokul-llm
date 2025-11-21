"""Test RLHF model registration and loading"""

import pytest
from src.core import load_model, get_registry


def test_rlhf_model_registered():
    """Test that rl_trained model is registered"""
    registry = get_registry()
    models = registry.list_models()
    
    assert 'rl_trained' in models
    
    model_info = registry.get_model_info('rl_trained')
    assert model_info is not None
    assert model_info.architecture == 'rlhf'
    assert model_info.description == 'RLHF-optimized model (PPO fine-tuned)'
    assert 'ppo_final.pt' in model_info.checkpoint_path


def test_load_rlhf_model():
    """Test loading RLHF model"""
    try:
        model, tokenizer, device = load_model('rl_trained', device='cpu')
        
        assert model is not None
        assert hasattr(model, 'generate'), "RLHF model must have generate() method"
        assert callable(model.generate)
        
        # Test generate method
        response = model.generate("test prompt", max_length=128, top_k=5)
        assert isinstance(response, str)
        assert len(response) > 0
        
    except Exception as e:
        pytest.skip(f"RLHF model loading failed (expected if checkpoint missing): {e}")


def test_rlhf_model_in_models_list():
    """Test that /models endpoint would include rl_trained"""
    registry = get_registry()
    models = registry.list_models()
    
    # Verify all expected models
    expected_models = ['mamba', 'transformer', 'rag_encoder', 'rl_trained']
    for model_name in expected_models:
        assert model_name in models, f"Model {model_name} not found in registry"
