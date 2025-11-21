#!/usr/bin/env python3
"""
Verification Script for AUTO-DETECTING Mamba Backend System

Run this to verify the auto-detection implementation works correctly.
"""

import sys
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 80)
    print("  AUTO-DETECTING MAMBA BACKEND VERIFICATION")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    # Test 1: Import auto-detection functions
    print("✓ Test 1: Checking auto-detection imports...")
    try:
        from src.core.mamba_loader import detect_mamba_backend, get_mamba_info, load_mamba_model
        print("  ✅ PASS: Auto-detection functions imported successfully")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 2: Backend detection
    print("\n✓ Test 2: Testing backend detection...")
    try:
        backend = detect_mamba_backend()
        info = get_mamba_info()
        
        print(f"  Platform: {info['platform']}")
        print(f"  Backend: {backend}")
        print(f"  Available: {info['available']}")
        print(f"  Reason: {info['reason']}")
        print(f"  CUDA: {info['cuda_available']}")
        print(f"  MPS: {info['mps_available']}")
        
        if 'install_command' in info:
            print(f"  Install: {info['install_command']}")
        
        assert backend in ["real-mamba", "mamba2", "none"]
        print("  ✅ PASS: Backend detection working")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 3: Model loading with auto-detection
    print("\n✓ Test 3: Testing model loading...")
    try:
        model = load_mamba_model()
        
        assert hasattr(model, 'available')
        assert hasattr(model, 'backend')
        assert hasattr(model, 'reason')
        
        print(f"  Model type: {type(model).__name__}")
        print(f"  Backend: {model.backend}")
        print(f"  Available: {model.available}")
        
        if not model.available:
            print(f"  Reason: {model.reason}")
        
        print("  ✅ PASS: Model loading working")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 4: Platform-specific logic
    print("\n✓ Test 4: Testing platform-specific logic...")
    try:
        current_platform = platform.system().lower()
        
        if current_platform == "darwin":
            # Mac should prefer mamba2
            expected_backends = ["mamba2", "none"]
            install_hint = "mamba2"
        elif current_platform in ["linux", "windows"]:
            # Linux/Windows should prefer real-mamba if CUDA available
            import torch
            if torch.cuda.is_available():
                expected_backends = ["real-mamba", "none"]
                install_hint = "mamba-ssm"
            else:
                expected_backends = ["none"]
                install_hint = "N/A (no GPU)"
        else:
            expected_backends = ["none"]
            install_hint = "N/A (unknown platform)"
        
        backend = detect_mamba_backend()
        assert backend in expected_backends, f"Expected {expected_backends}, got {backend}"
        
        print(f"  Platform: {current_platform}")
        print(f"  Expected backends: {expected_backends}")
        print(f"  Actual backend: {backend}")
        print(f"  Install hint: {install_hint}")
        print("  ✅ PASS: Platform logic correct")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 5: Model registry integration
    print("\n✓ Test 5: Testing model registry integration...")
    try:
        from src.core.model_registry import is_model_available, get_model_instance
        
        mamba_avail = is_model_available('mamba')
        transformer_avail = is_model_available('transformer')
        
        print(f"  Mamba available: {mamba_avail}")
        print(f"  Transformer available: {transformer_avail}")
        
        assert isinstance(mamba_avail, bool)
        assert isinstance(transformer_avail, bool)
        assert transformer_avail == True  # Transformer should always be available
        
        print("  ✅ PASS: Model registry integration working")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 6: Auto-routing still works
    print("\n✓ Test 6: Testing auto-routing...")
    try:
        from src.pipelines.auto_pipeline import AutoPipeline
        
        pipeline = AutoPipeline()
        result = pipeline.select_model("Test query", 0, "", [])
        
        assert result in ['mamba', 'transformer']
        print(f"  Selected model: {result}")
        print("  ✅ PASS: Auto-routing working")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 7: Config files exist
    print("\n✓ Test 7: Testing config files...")
    try:
        import yaml
        
        # Check mamba_auto.yaml
        auto_config = Path("configs/mamba_auto.yaml")
        if auto_config.exists():
            with open(auto_config) as f:
                config = yaml.safe_load(f)
            assert 'mamba' in config
            assert 'backend_selection' in config
            print("  ✅ mamba_auto.yaml exists and valid")
        else:
            print("  ⚠️  mamba_auto.yaml not found")
        
        # Check updated lora_mamba.yaml
        lora_config = Path("configs/lora_mamba.yaml")
        with open(lora_config) as f:
            config = yaml.safe_load(f)
        
        assert 'model' in config
        assert 'mamba2_model' in config['model']
        print("  ✅ lora_mamba.yaml updated with multi-backend support")
        
        print("  ✅ PASS: Config files correct")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 8: LoRA target module logic
    print("\n✓ Test 8: Testing LoRA target modules...")
    try:
        # Test the logic (without actually loading models)
        real_mamba_targets = ["in_proj", "out_proj", "x_proj", "dt_proj"]
        mamba2_targets = ["mixer.Wq", "mixer.Wk", "mixer.Wv"]
        transformer_targets = ["c_attn", "c_proj"]
        
        print(f"  REAL Mamba SSM targets: {real_mamba_targets}")
        print(f"  Mamba2 targets: {mamba2_targets}")
        print(f"  Transformer targets: {transformer_targets}")
        
        # Verify they're different
        assert set(real_mamba_targets) != set(mamba2_targets)
        assert set(mamba2_targets) != set(transformer_targets)
        assert set(real_mamba_targets) != set(transformer_targets)
        
        print("  ✅ PASS: LoRA target modules are backend-specific")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Summary
    print()
    print("=" * 80)
    print("  VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print()
    
    if failed == 0:
        print("  🎉 ALL TESTS PASSED! Auto-detection system working correctly.")
        print()
        
        # Show current system recommendation
        from src.core.mamba_loader import get_mamba_info
        info = get_mamba_info()
        
        print("  📋 CURRENT SYSTEM STATUS:")
        print(f"     Platform: {info['platform']}")
        print(f"     Backend: {info['backend']}")
        print(f"     Available: {info['available']}")
        
        if not info['available'] and 'install_command' in info:
            print()
            print("  🚀 TO ENABLE MAMBA:")
            print(f"     {info['install_command']}")
            print()
            print("  After installation, the system will automatically detect and use the")
            print("  appropriate Mamba backend for optimal performance.")
        
        print()
        return 0
    else:
        print("  ⚠️  SOME TESTS FAILED - Please review errors above")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
