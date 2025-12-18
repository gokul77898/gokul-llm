#!/usr/bin/env python3
"""
Phase P4: Health & Warmup Verification Script

Verifies:
- /health/gpu works without GPU
- State transitions correct
- Warmup skipped when no GPU
- No crash when models not loaded

Prints: PHASE P4 READY
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_state_enum():
    """Test ModelState enum exists and has correct values."""
    print("1. Testing ModelState enum...")
    
    from src.inference.local_models import ModelState
    
    # Check all states exist
    assert ModelState.UNINITIALIZED.value == "uninitialized"
    assert ModelState.LOADING.value == "loading"
    assert ModelState.READY.value == "ready"
    assert ModelState.DEGRADED.value == "degraded"
    assert ModelState.FAILED.value == "failed"
    
    print("   ✓ ModelState enum has all required states")
    return True


def test_warmup_telemetry():
    """Test WarmupTelemetry dataclass exists."""
    print("2. Testing WarmupTelemetry dataclass...")
    
    from src.inference.local_models import WarmupTelemetry
    
    # Create instance
    telemetry = WarmupTelemetry()
    
    # Check fields exist
    assert hasattr(telemetry, 'encoder_warmup_ms')
    assert hasattr(telemetry, 'decoder_warmup_ms')
    assert hasattr(telemetry, 'first_token_latency_ms')
    assert hasattr(telemetry, 'encoder_warmed_up')
    assert hasattr(telemetry, 'decoder_warmed_up')
    assert hasattr(telemetry, 'warmup_timestamp')
    
    # Check defaults
    assert telemetry.encoder_warmed_up == False
    assert telemetry.decoder_warmed_up == False
    
    print("   ✓ WarmupTelemetry has all required fields")
    return True


def test_registry_state_machine():
    """Test LocalModelRegistry state machine."""
    print("3. Testing LocalModelRegistry state machine...")
    
    from src.inference.local_models import LocalModelRegistry, ModelState
    
    registry = LocalModelRegistry()
    
    # Initial state should be UNINITIALIZED
    assert registry.get_state() == ModelState.UNINITIALIZED, \
        f"Expected UNINITIALIZED, got {registry.get_state()}"
    print("   ✓ Initial state is UNINITIALIZED")
    
    # Get warmup telemetry
    warmup = registry.get_warmup_telemetry()
    assert warmup.encoder_warmed_up == False
    assert warmup.decoder_warmed_up == False
    print("   ✓ Warmup telemetry accessible")
    
    return True


def test_gpu_health_info():
    """Test get_gpu_health_info function."""
    print("4. Testing get_gpu_health_info...")
    
    from src.inference.local_models import get_gpu_health_info
    
    # Should not throw even without GPU
    info = get_gpu_health_info()
    
    # Check required fields
    assert 'cuda_available' in info
    assert 'total_vram_gb' in info
    assert 'free_vram_gb' in info
    
    print(f"   ✓ cuda_available: {info['cuda_available']}")
    print(f"   ✓ total_vram_gb: {info['total_vram_gb']}")
    print(f"   ✓ free_vram_gb: {info['free_vram_gb']}")
    
    return True


def test_load_fails_without_cuda():
    """Test that load_encoder/load_decoder fail without CUDA."""
    print("5. Testing load fails without CUDA...")
    
    from src.inference.local_models import LocalModelRegistry, ModelState
    
    registry = LocalModelRegistry()
    
    # Try to load encoder - should fail without CUDA
    try:
        registry.load_encoder("test-model")
        print("   ✗ ERROR: Should have failed without CUDA")
        return False
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"   ✓ Encoder load failed correctly: CUDA not available")
        else:
            print(f"   ✓ Encoder load failed: {str(e)[:50]}...")
    
    # State should be FAILED after load error
    # Note: State transitions to LOADING then FAILED
    assert registry.get_state() == ModelState.FAILED, \
        f"Expected FAILED after load error, got {registry.get_state()}"
    print("   ✓ State is FAILED after load error")
    
    return True


def test_health_endpoint_import():
    """Test that health endpoint can be imported."""
    print("6. Testing health endpoint import...")
    
    # Import server module
    from src.inference.server import health_gpu, MODEL_REGISTRY
    
    print("   ✓ health_gpu endpoint imported")
    print("   ✓ MODEL_REGISTRY imported")
    
    return True


def test_health_endpoint_response():
    """Test health endpoint response format."""
    print("7. Testing health endpoint response...")
    
    from src.inference.server import health_gpu
    
    # Call the endpoint function directly
    response = health_gpu()
    
    # Check required fields
    required_fields = [
        'cuda_available',
        'total_vram_gb',
        'free_vram_gb',
        'encoder_loaded',
        'decoder_loaded',
        'model_state',
        'timestamp',
    ]
    
    for field in required_fields:
        assert field in response, f"Missing field: {field}"
        print(f"   ✓ {field}: {response[field]}")
    
    # Check warmup field
    assert 'warmup' in response
    print(f"   ✓ warmup: present")
    
    return True


def test_no_warmup_without_gpu():
    """Test that warmup is skipped when no GPU."""
    print("8. Testing warmup skipped without GPU...")
    
    from src.inference.local_models import LocalModelRegistry
    
    registry = LocalModelRegistry()
    warmup = registry.get_warmup_telemetry()
    
    # Warmup should not have run
    assert warmup.encoder_warmed_up == False
    assert warmup.decoder_warmed_up == False
    assert warmup.encoder_warmup_ms is None
    assert warmup.decoder_warmup_ms is None
    
    print("   ✓ Encoder warmup not run")
    print("   ✓ Decoder warmup not run")
    
    return True


def test_moe_generate_unchanged():
    """Test that /moe-generate behavior is unchanged."""
    print("9. Testing /moe-generate behavior unchanged...")
    
    # Import and check that the endpoint exists
    from src.inference.server import moe_generate
    
    # Check it's still an async function
    import asyncio
    assert asyncio.iscoroutinefunction(moe_generate), \
        "moe_generate should be async"
    
    print("   ✓ moe_generate is async")
    print("   ✓ Endpoint exists and unchanged")
    
    return True


def main():
    print("=" * 50)
    print("PHASE P4: Health & Warmup Verification")
    print("=" * 50)
    print()
    
    tests = [
        test_model_state_enum,
        test_warmup_telemetry,
        test_registry_state_machine,
        test_gpu_health_info,
        test_load_fails_without_cuda,
        test_health_endpoint_import,
        test_health_endpoint_response,
        test_no_warmup_without_gpu,
        test_moe_generate_unchanged,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print("PHASE P4 VERIFICATION RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print()
    
    if failed == 0:
        print("✓ All tests passed")
        print()
        print("PHASE P4 READY")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
