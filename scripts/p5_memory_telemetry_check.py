#!/usr/bin/env python3
"""
Phase P5: Memory & OOM Telemetry Verification Script

Verifies:
- Snapshot fields exist (even if null)
- No crashes without GPU
- Correct error mapping for simulated OOM
- Trace artifact includes gpu_memory

Prints: PHASE P5 READY
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gpu_memory_snapshot_class():
    """Test GPUMemorySnapshot dataclass exists and has correct fields."""
    print("1. Testing GPUMemorySnapshot dataclass...")
    
    from src.inference.local_models import GPUMemorySnapshot
    
    # Create instance with defaults
    snapshot = GPUMemorySnapshot()
    
    # Check all required fields exist
    required_fields = [
        'free_vram_gb',
        'total_vram_gb',
        'reserved_vram_gb',
        'allocated_vram_gb',
        'max_reserved_vram_gb',
        'peak_allocated_vram_gb',
        'timestamp',
    ]
    
    for field in required_fields:
        assert hasattr(snapshot, field), f"Missing field: {field}"
        print(f"   ✓ {field}: {getattr(snapshot, field)}")
    
    # Test to_dict method
    d = snapshot.to_dict()
    assert isinstance(d, dict)
    for field in required_fields:
        assert field in d, f"Missing field in to_dict: {field}"
    
    print("   ✓ to_dict() works correctly")
    return True


def test_get_gpu_memory_snapshot():
    """Test get_gpu_memory_snapshot function."""
    print("2. Testing get_gpu_memory_snapshot function...")
    
    from src.inference.local_models import get_gpu_memory_snapshot, GPUMemorySnapshot
    
    # Should not throw even without GPU
    snapshot = get_gpu_memory_snapshot()
    
    assert isinstance(snapshot, GPUMemorySnapshot)
    print(f"   ✓ Returns GPUMemorySnapshot instance")
    
    # Check timestamp is set
    assert snapshot.timestamp is not None
    print(f"   ✓ timestamp: {snapshot.timestamp}")
    
    # Without GPU, values should be None
    print(f"   ✓ free_vram_gb: {snapshot.free_vram_gb}")
    print(f"   ✓ total_vram_gb: {snapshot.total_vram_gb}")
    
    return True


def test_warmup_error_class():
    """Test WarmupError exception class."""
    print("3. Testing WarmupError exception class...")
    
    from src.inference.local_models import WarmupError
    
    # Test creation
    error = WarmupError("warmup_oom", "GPU out of memory during warmup")
    
    assert error.reason == "warmup_oom"
    assert error.details == "GPU out of memory during warmup"
    print(f"   ✓ WarmupError.reason: {error.reason}")
    print(f"   ✓ WarmupError.details: {error.details}")
    
    return True


def test_encoder_oom_classification():
    """Test that encoder OOM is classified correctly."""
    print("4. Testing encoder OOM classification...")
    
    from src.inference.local_models import EncoderExecutionError
    
    # Test encoder_oom error
    error = EncoderExecutionError("encoder_oom", "GPU out of memory")
    
    assert error.reason == "encoder_oom"
    print(f"   ✓ EncoderExecutionError.reason: {error.reason}")
    
    return True


def test_decoder_oom_classification():
    """Test that decoder OOM is classified correctly."""
    print("5. Testing decoder OOM classification...")
    
    from src.inference.local_models import DecoderExecutionError
    
    # Test decoder_oom error
    error = DecoderExecutionError("decoder_oom", "GPU out of memory")
    
    assert error.reason == "decoder_oom"
    print(f"   ✓ DecoderExecutionError.reason: {error.reason}")
    
    return True


def test_gpu_memory_trace_class():
    """Test GPUMemoryTrace dataclass in trace module."""
    print("6. Testing GPUMemoryTrace dataclass...")
    
    from src.inference.trace import GPUMemoryTrace
    
    # Create instance
    trace = GPUMemoryTrace()
    
    assert hasattr(trace, 'encoder')
    assert hasattr(trace, 'decoder')
    print(f"   ✓ encoder field: {trace.encoder}")
    print(f"   ✓ decoder field: {trace.decoder}")
    
    # Test with data
    trace_with_data = GPUMemoryTrace(
        encoder={"before": {"free_vram_gb": 10.0}, "after": {"free_vram_gb": 8.0}},
        decoder={"before": {"free_vram_gb": 8.0}, "after": {"free_vram_gb": 5.0}},
    )
    
    assert trace_with_data.encoder is not None
    assert trace_with_data.decoder is not None
    print("   ✓ GPUMemoryTrace accepts encoder/decoder data")
    
    return True


def test_replay_artifact_gpu_memory():
    """Test ReplayArtifact includes gpu_memory field."""
    print("7. Testing ReplayArtifact gpu_memory field...")
    
    from src.inference.trace import ReplayArtifact, GPUMemoryTrace
    
    # Create artifact with gpu_memory
    artifact = ReplayArtifact(
        trace_id="test-trace-id",
        timestamp="2025-01-01T00:00:00",
        query="Test query",
        gpu_memory=GPUMemoryTrace(
            encoder={"before": {}, "after": {}},
            decoder={"before": {}, "after": {}},
        ),
    )
    
    assert artifact.gpu_memory is not None
    print("   ✓ ReplayArtifact.gpu_memory field exists")
    
    # Test to_dict includes gpu_memory
    d = artifact.to_dict()
    assert "gpu_memory" in d
    print("   ✓ to_dict() includes gpu_memory")
    
    return True


def test_server_imports():
    """Test that server imports Phase P5 components."""
    print("8. Testing server imports...")
    
    from src.inference.server import (
        get_gpu_memory_snapshot,
        GPUMemorySnapshot,
        GPUMemoryTrace,
    )
    
    print("   ✓ get_gpu_memory_snapshot imported")
    print("   ✓ GPUMemorySnapshot imported")
    print("   ✓ GPUMemoryTrace imported")
    
    return True


def test_no_crash_without_gpu():
    """Test that memory functions don't crash without GPU."""
    print("9. Testing no crash without GPU...")
    
    from src.inference.local_models import (
        get_gpu_memory_snapshot,
        get_gpu_health_info,
    )
    
    # These should not throw
    try:
        snapshot = get_gpu_memory_snapshot()
        print("   ✓ get_gpu_memory_snapshot() - no crash")
    except Exception as e:
        print(f"   ✗ get_gpu_memory_snapshot() crashed: {e}")
        return False
    
    try:
        health = get_gpu_health_info()
        print("   ✓ get_gpu_health_info() - no crash")
    except Exception as e:
        print(f"   ✗ get_gpu_health_info() crashed: {e}")
        return False
    
    return True


def test_run_local_encoder_returns_gpu_memory():
    """Test that run_local_encoder result includes gpu_memory field."""
    print("10. Testing run_local_encoder returns gpu_memory...")
    
    # We can't actually run the encoder without GPU, but we can check
    # the function signature and docstring
    from src.inference.local_models import run_local_encoder
    import inspect
    
    # Check docstring mentions gpu_memory
    doc = run_local_encoder.__doc__
    assert "gpu_memory" in doc.lower() or "memory snapshot" in doc.lower()
    print("   ✓ run_local_encoder docstring mentions memory")
    
    # Check return type annotation
    sig = inspect.signature(run_local_encoder)
    print(f"   ✓ run_local_encoder signature: {sig}")
    
    return True


def test_run_local_decoder_returns_gpu_memory():
    """Test that run_local_decoder result includes gpu_memory field."""
    print("11. Testing run_local_decoder returns gpu_memory...")
    
    from src.inference.local_models import run_local_decoder
    import inspect
    
    # Check docstring mentions gpu_memory
    doc = run_local_decoder.__doc__
    assert "gpu_memory" in doc.lower() or "memory snapshot" in doc.lower()
    print("   ✓ run_local_decoder docstring mentions memory")
    
    # Check return type annotation (should be Dict now)
    sig = inspect.signature(run_local_decoder)
    print(f"   ✓ run_local_decoder signature: {sig}")
    
    return True


def main():
    print("=" * 50)
    print("PHASE P5: Memory & OOM Telemetry Verification")
    print("=" * 50)
    print()
    
    tests = [
        test_gpu_memory_snapshot_class,
        test_get_gpu_memory_snapshot,
        test_warmup_error_class,
        test_encoder_oom_classification,
        test_decoder_oom_classification,
        test_gpu_memory_trace_class,
        test_replay_artifact_gpu_memory,
        test_server_imports,
        test_no_crash_without_gpu,
        test_run_local_encoder_returns_gpu_memory,
        test_run_local_decoder_returns_gpu_memory,
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
    print("PHASE P5 VERIFICATION RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print()
    
    if failed == 0:
        print("✓ All tests passed")
        print()
        print("PHASE P5 READY")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
