"""
Tests for REAL Mamba SSM Loading

Verifies that the REAL Mamba State Space Model can be loaded and used.
"""

import pytest
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRealMambaSSM:
    """Test REAL Mamba SSM (State Space Model) loading and inference"""
    
    def test_mamba_availability_check(self):
        """Test that is_mamba_available returns bool"""
        from src.core.mamba_loader import is_mamba_available
        
        available = is_mamba_available()
        assert isinstance(available, bool)
        
        if available:
            print("✅ REAL Mamba SSM package (mamba-ssm) is installed")
        else:
            print("ℹ️  REAL Mamba SSM not available")
            print("   Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
    
    def test_mamba_info_structure(self):
        """Test get_mamba_info returns correct structure"""
        from src.core.mamba_loader import get_mamba_info
        
        info = get_mamba_info()
        
        assert isinstance(info, dict)
        assert "available" in info
        assert "package" in info
        assert "cuda_available" in info
        assert "mps_available" in info
        assert "recommended_device" in info
        
        print(f"Mamba Info: {info}")
    
    @pytest.mark.skipif(
        not pytest.importorskip("mamba_ssm", reason="mamba-ssm not installed"),
        reason="Requires mamba-ssm package"
    )
    def test_load_real_mamba_model(self):
        """Test loading REAL Mamba SSM model"""
        from src.core.mamba_loader import load_mamba_model
        
        # Try to load REAL Mamba
        config = {
            'model_name': 'state-spaces/mamba-130m',
            'enable_mamba': True
        }
        
        model = load_mamba_model(config)
        
        # Check if model loaded or returned shim
        assert hasattr(model, 'available')
        
        if model.available:
            print("✅ REAL Mamba SSM loaded successfully")
            assert hasattr(model, 'model')
            assert hasattr(model, 'tokenizer')
            assert hasattr(model, 'device')
            assert hasattr(model, 'generate_with_state_space')
            print(f"   Device: {model.device}")
        else:
            print(f"ℹ️  REAL Mamba SSM unavailable: {model.reason}")
            pytest.skip("Mamba SSM not available on this system")
    
    @pytest.mark.skipif(
        not pytest.importorskip("mamba_ssm", reason="mamba-ssm not installed"),
        reason="Requires mamba-ssm package"
    )
    def test_real_mamba_forward_pass(self):
        """Test REAL Mamba SSM forward pass"""
        from src.core.mamba_loader import load_mamba_model
        
        model = load_mamba_model()
        
        if not model.available:
            pytest.skip(f"REAL Mamba SSM not available: {model.reason}")
        
        # Create sample input
        input_text = "This is a test of the REAL Mamba State Space Model."
        inputs = model.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(model.device)
        
        # Run forward pass
        with torch.no_grad():
            try:
                outputs = model.model(input_ids)
                assert outputs is not None
                print("✅ REAL Mamba SSM forward pass successful")
                print(f"   Output shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")
            except Exception as e:
                pytest.fail(f"Forward pass failed: {e}")
    
    @pytest.mark.skipif(
        not pytest.importorskip("mamba_ssm", reason="mamba-ssm not installed"),
        reason="Requires mamba-ssm package"
    )
    def test_real_mamba_generation(self):
        """Test REAL Mamba SSM text generation"""
        from src.core.mamba_loader import load_mamba_model
        
        model = load_mamba_model()
        
        if not model.available:
            pytest.skip(f"REAL Mamba SSM not available: {model.reason}")
        
        # Test generation
        prompt = "The legal definition of"
        context = "In contract law, a consideration is"
        
        try:
            answer = model.generate_with_state_space(
                prompt=prompt,
                context=context,
                max_new_tokens=50,
                temperature=0.7
            )
            
            assert isinstance(answer, str)
            assert len(answer) > 0
            print("✅ REAL Mamba SSM generation successful")
            print(f"   Generated: {answer[:100]}...")
            
        except Exception as e:
            # May fail if no CUDA GPU available
            if "CUDA" in str(e) or "MPS" in str(e):
                pytest.skip(f"Generation requires GPU: {e}")
            else:
                pytest.fail(f"Generation failed: {e}")
    
    def test_shim_when_unavailable(self):
        """Test that shim is returned when Mamba unavailable"""
        from src.core.mamba_loader import load_mamba_model
        import os
        
        # Force disable
        old_val = os.environ.get("ENABLE_MAMBA")
        os.environ["ENABLE_MAMBA"] = "false"
        
        try:
            model = load_mamba_model()
            
            assert hasattr(model, 'available')
            assert model.available == False
            assert hasattr(model, 'reason')
            assert "environment variable" in model.reason.lower() or "disabled" in model.reason.lower()
            
            print(f"✅ Shim returned correctly: {model.reason}")
            
            # Test that generation raises error
            with pytest.raises(RuntimeError) as exc_info:
                model.generate(None)
            
            assert "not available" in str(exc_info.value).lower()
            
        finally:
            # Restore
            if old_val:
                os.environ["ENABLE_MAMBA"] = old_val
            else:
                os.environ.pop("ENABLE_MAMBA", None)
    
    def test_model_registry_integration(self):
        """Test REAL Mamba SSM integration with model registry"""
        from src.core.model_registry import is_model_available
        
        # Check if mamba is available in registry
        mamba_available = is_model_available('mamba')
        assert isinstance(mamba_available, bool)
        
        if mamba_available:
            print("✅ REAL Mamba SSM available via model registry")
        else:
            print("ℹ️  REAL Mamba SSM not available in model registry")
    
    def test_config_loading(self):
        """Test REAL Mamba SSM config file exists and is valid"""
        config_path = Path("configs/mamba_real.yaml")
        
        if not config_path.exists():
            pytest.skip("mamba_real.yaml not found")
        
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert 'model' in config
        assert 'base_model' in config['model']
        
        # Check it's the REAL Mamba SSM, not fake
        base_model = config['model']['base_model']
        assert 'mamba' in base_model.lower()
        assert 'state-spaces' in base_model or 'mamba-' in base_model
        
        print(f"✅ REAL Mamba SSM config valid: {base_model}")
    
    def test_lora_config_has_mamba_modules(self):
        """Test LoRA config has REAL Mamba SSM target modules"""
        config_path = Path("configs/lora_mamba.yaml")
        
        if not config_path.exists():
            pytest.skip("lora_mamba.yaml not found")
        
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert 'lora' in config
        assert 'target_modules' in config['lora']
        
        target_modules = config['lora']['target_modules']
        
        # Check for REAL Mamba SSM modules (not Transformer modules like c_attn)
        mamba_modules = ['in_proj', 'out_proj', 'x_proj', 'dt_proj']
        has_mamba_module = any(mod in target_modules for mod in mamba_modules)
        
        assert has_mamba_module, f"LoRA config should have Mamba SSM modules, got: {target_modules}"
        
        # Should NOT have Transformer-specific modules
        transformer_modules = ['c_attn', 'c_proj', 'q_proj', 'k_proj', 'v_proj']
        has_transformer_module = any(mod in target_modules for mod in transformer_modules)
        
        if has_transformer_module:
            pytest.fail(f"LoRA config has Transformer modules, should be Mamba SSM modules: {target_modules}")
        
        print(f"✅ LoRA config has REAL Mamba SSM modules: {target_modules}")


def test_old_custom_mamba_disabled():
    """Verify old custom fake Mamba is disabled/backed up"""
    old_mamba_path = Path("src/mamba")
    backup_path = Path("src/mamba_OLD_FAKE.bak")
    
    if old_mamba_path.exists():
        # Old path exists, check if it's actually the old fake one
        init_file = old_mamba_path / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            if "DocumentTokenizer" in content and "MambaModel" in content:
                pytest.fail("Old custom fake Mamba still active at src/mamba/, should be backed up")
    
    # Check backup exists
    if backup_path.exists():
        print(f"✅ Old custom fake Mamba backed up to: {backup_path}")
    else:
        print("ℹ️  Old custom Mamba not found (may have been removed)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
