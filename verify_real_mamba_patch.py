#!/usr/bin/env python3
"""
Verification Script for REAL Mamba SSM Patch

Run this to verify the patch was applied correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 80)
    print("  REAL MAMBA SSM PATCH VERIFICATION")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    # Test 1: Check mamba_loader imports correctly
    print("✓ Test 1: Checking mamba_loader imports...")
    try:
        from src.core.mamba_loader import is_mamba_available, get_mamba_info, load_mamba_model
        print("  ✅ PASS: mamba_loader imports successfully")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 2: Check REAL Mamba availability
    print("\n✓ Test 2: Checking REAL Mamba SSM availability...")
    try:
        from src.core.mamba_loader import is_mamba_available, get_mamba_info
        available = is_mamba_available()
        info = get_mamba_info()
        print(f"  REAL Mamba SSM available: {available}")
        print(f"  Package: {info['package']}")
        print(f"  CUDA available: {info['cuda_available']}")
        print(f"  MPS available: {info['mps_available']}")
        print(f"  Recommended device: {info['recommended_device']}")
        if not available:
            print("  ℹ️  NOTE: mamba-ssm package not installed (expected)")
            print("  ℹ️  Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
        print("  ✅ PASS: Availability check works")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 3: Check shim returns when unavailable
    print("\n✓ Test 3: Checking MambaShim fallback...")
    try:
        from src.core.mamba_loader import load_mamba_model
        model = load_mamba_model()
        assert hasattr(model, 'available'), "Model should have 'available' attribute"
        assert hasattr(model, 'reason'), "Model should have 'reason' attribute"
        print(f"  Model type: {type(model).__name__}")
        print(f"  Available: {model.available}")
        print(f"  Reason: {model.reason}")
        print("  ✅ PASS: Shim works correctly")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 4: Check model_registry integration
    print("\n✓ Test 4: Checking model_registry integration...")
    try:
        from src.core.model_registry import is_model_available
        mamba_avail = is_model_available('mamba')
        transformer_avail = is_model_available('transformer')
        print(f"  Mamba available via registry: {mamba_avail}")
        print(f"  Transformer available: {transformer_avail}")
        assert isinstance(mamba_avail, bool), "Should return bool"
        assert isinstance(transformer_avail, bool), "Should return bool"
        print("  ✅ PASS: Model registry integration works")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 5: Check old custom mamba is disabled
    print("\n✓ Test 5: Checking old custom mamba is disabled...")
    try:
        old_path = Path("src/mamba")
        backup_path = Path("src/mamba_OLD_FAKE.bak")
        
        if backup_path.exists():
            print(f"  ✅ Old custom mamba backed up to: {backup_path}")
        else:
            print("  ℹ️  Backup not found (may have been removed)")
        
        if old_path.exists():
            # Check if it's the old fake one
            init_file = old_path / "__init__.py"
            if init_file.exists():
                content = init_file.read_text()
                if "DocumentTokenizer" in content:
                    print("  ⚠️  WARNING: Old custom mamba still at src/mamba/")
                    failed += 1
                else:
                    print("  ✅ PASS: Old mamba disabled")
                    passed += 1
            else:
                print("  ✅ PASS: Old mamba disabled")
                passed += 1
        else:
            print("  ✅ PASS: Old mamba directory removed/backed up")
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 6: Check config files
    print("\n✓ Test 6: Checking config files...")
    try:
        import yaml
        
        # Check mamba_real.yaml
        real_config = Path("configs/mamba_real.yaml")
        if real_config.exists():
            with open(real_config) as f:
                config = yaml.safe_load(f)
            base_model = config['model']['base_model']
            assert 'state-spaces' in base_model or 'mamba' in base_model.lower()
            print(f"  ✅ mamba_real.yaml exists: {base_model}")
        else:
            print("  ⚠️  mamba_real.yaml not found")
        
        # Check lora_mamba.yaml has REAL Mamba modules
        lora_config = Path("configs/lora_mamba.yaml")
        with open(lora_config) as f:
            config = yaml.safe_load(f)
        
        base_model = config['model']['base_model']
        targets = config['lora']['target_modules']
        
        print(f"  Base model: {base_model}")
        print(f"  LoRA targets: {targets}")
        
        # Check for REAL Mamba modules
        mamba_modules = ['in_proj', 'out_proj', 'x_proj', 'dt_proj']
        has_mamba = any(m in targets for m in mamba_modules)
        
        # Check NOT Transformer modules
        transformer_modules = ['c_attn', 'c_proj']
        has_transformer = any(m in targets for m in transformer_modules)
        
        if has_mamba and not has_transformer:
            print("  ✅ PASS: LoRA config has REAL Mamba SSM modules")
            passed += 1
        else:
            print(f"  ❌ FAIL: LoRA config has wrong modules (has_mamba={has_mamba}, has_transformer={has_transformer})")
            failed += 1
            
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
    
    # Test 7: Check routing still works
    print("\n✓ Test 7: Checking auto-routing works...")
    try:
        from src.pipelines.auto_pipeline import AutoPipeline
        pipeline = AutoPipeline()
        
        # Test routing
        result = pipeline.select_model("What is a judgment?", 0, "", [])
        print(f"  Selected model: {result}")
        assert result in ['mamba', 'transformer'], f"Unexpected model: {result}"
        print("  ✅ PASS: Routing works")
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
        print("  🎉 ALL TESTS PASSED! REAL Mamba SSM patch verified successfully.")
        print()
        print("  Next steps:")
        print("  1. Install REAL Mamba SSM: pip install mamba-ssm causal-conv1d>=1.2.0")
        print("  2. Test with CUDA GPU for best performance")
        print("  3. Run: pytest tests/test_real_mamba_loading.py")
        print()
        return 0
    else:
        print("  ⚠️  SOME TESTS FAILED - Please review errors above")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
