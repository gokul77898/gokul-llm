"""
Tests for Mamba Shim (Fallback Handling)

Tests that the system gracefully handles Mamba unavailability.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMambaShim:
    """Test Mamba shim and fallback behavior"""
    
    def test_mamba_loader_import(self):
        """Test that mamba_loader can be imported"""
        try:
            from src.core import mamba_loader
            assert mamba_loader is not None
        except ImportError as e:
            pytest.fail(f"Failed to import mamba_loader: {e}")
    
    def test_mamba_shim_creation(self):
        """Test MambaShim object creation"""
        from src.core.mamba_loader import MambaShim
        
        shim = MambaShim("Test reason")
        
        assert shim.available == False
        assert shim.reason == "Test reason"
    
    def test_mamba_shim_generate_raises_error(self):
        """Test that MambaShim.generate raises informative error"""
        from src.core.mamba_loader import MambaShim
        
        shim = MambaShim("Mamba not installed")
        
        with pytest.raises(RuntimeError) as exc_info:
            shim.generate_with_state_space("test query", "test context")
        
        assert "not available" in str(exc_info.value).lower()
        assert "Mamba not installed" in str(exc_info.value)
    
    def test_load_mamba_model_returns_shim_when_unavailable(self):
        """Test load_mamba_model returns shim when Mamba unavailable"""
        from src.core.mamba_loader import load_mamba_model
        import os
        
        # Temporarily disable Mamba
        original_env = os.environ.get("ENABLE_MAMBA")
        os.environ["ENABLE_MAMBA"] = "false"
        
        try:
            result = load_mamba_model()
            
            # Should return a shim
            assert hasattr(result, 'available')
            assert result.available == False
            assert hasattr(result, 'reason')
            
            print(f"✅ Shim returned with reason: {result.reason}")
            
        finally:
            # Restore env
            if original_env is not None:
                os.environ["ENABLE_MAMBA"] = original_env
            else:
                os.environ.pop("ENABLE_MAMBA", None)
    
    def test_is_mamba_available_returns_bool(self):
        """Test is_mamba_available returns boolean"""
        from src.core.mamba_loader import is_mamba_available
        
        available = is_mamba_available()
        
        assert isinstance(available, bool)
        
        if available:
            print("✅ Mamba is available on this system")
        else:
            print("ℹ️  Mamba not available (expected on systems without mamba-ssm)")
    
    def test_generator_fallback_handling(self):
        """Test generator.py handles Mamba fallback"""
        from src.core.generator import generate_answer
        
        # Try to generate with mamba (should fallback if unavailable)
        result = generate_answer(
            model_key="mamba",
            prompt="Test query",
            context="Test context",
            fallback_enabled=True
        )
        
        assert 'answer' in result
        assert 'model_used' in result
        assert 'fallback_used' in result
        
        # Model should be either mamba or transformer (with fallback)
        assert result['model_used'] in ['mamba', 'transformer']
        
        # If fallback occurred, model should be transformer
        if result['fallback_used']:
            assert result['model_used'] == 'transformer'
            print("✅ Fallback to transformer worked")
        else:
            # Either mamba worked or transformer was used without fallback
            print(f"✅ {result['model_used']} generation completed")
    
    def test_helpful_error_message(self):
        """Test that shim provides helpful error message"""
        from src.core.mamba_loader import MambaShim
        
        reasons = [
            "Import failed: No module named 'mamba_ssm'",
            "Disabled via environment variable",
            "Disabled via configuration",
        ]
        
        for reason in reasons:
            shim = MambaShim(reason)
            
            try:
                shim.generate_with_state_space("query", "context")
                pytest.fail("Should have raised RuntimeError")
            except RuntimeError as e:
                error_msg = str(e)
                assert reason in error_msg, f"Error message should contain reason: {reason}"
                print(f"✅ Error message: {error_msg}")
    
    def test_model_registry_checks_availability(self):
        """Test model_registry.is_model_available checks Mamba correctly"""
        from src.core.model_registry import is_model_available
        
        # Check mamba
        mamba_available = is_model_available('mamba')
        assert isinstance(mamba_available, bool)
        
        # Check transformer (should always be available if registered)
        transformer_available = is_model_available('transformer')
        assert isinstance(transformer_available, bool)
        
        print(f"Mamba available: {mamba_available}")
        print(f"Transformer available: {transformer_available}")


def test_env_variable_control():
    """Test ENABLE_MAMBA environment variable works"""
    import os
    from src.core.mamba_loader import load_mamba_model
    
    # Test disabling via env var
    os.environ["ENABLE_MAMBA"] = "false"
    result = load_mamba_model()
    
    assert result.available == False
    assert "environment variable" in result.reason.lower()
    
    # Cleanup
    os.environ.pop("ENABLE_MAMBA", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
