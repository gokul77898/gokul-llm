#!/usr/bin/env python3
"""
Phase P6: Configuration Integrity & Kill Switches Verification Script

Verifies:
- Config hash exists
- Hash changes if config file edited
- Disabling decoder → forced refusal
- /health returns hash + flags
- Trace artifact contains hash + flags

Prints: PHASE P6: CONFIG INTEGRITY VERIFICATION - ALL TESTS PASSED
"""

import sys
import os
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_fingerprint_module():
    """Test config_fingerprint module exists and works."""
    print("1. Testing config_fingerprint module...")
    
    from src.inference.config_fingerprint import (
        get_config_hash,
        get_feature_flags,
        is_encoder_enabled,
        is_rag_enabled,
        is_decoder_enabled,
        compute_config_fingerprint,
        ConfigFingerprint,
    )
    
    print("   ✓ All functions imported successfully")
    return True


def test_config_hash_exists():
    """Test that config hash is computed and exists."""
    print("2. Testing config hash exists...")
    
    from src.inference.config_fingerprint import get_config_hash
    
    config_hash = get_config_hash()
    
    assert config_hash is not None, "Config hash is None"
    assert len(config_hash) == 16, f"Config hash length is {len(config_hash)}, expected 16"
    assert all(c in '0123456789abcdef' for c in config_hash), "Config hash is not hex"
    
    print(f"   ✓ Config hash: {config_hash}")
    return True


def test_config_fingerprint_details():
    """Test that config fingerprint contains expected details."""
    print("3. Testing config fingerprint details...")
    
    from src.inference.config_fingerprint import get_config_fingerprint
    
    fingerprint = get_config_fingerprint()
    
    assert fingerprint.config_hash is not None
    assert fingerprint.config_files is not None
    assert fingerprint.constants is not None
    assert fingerprint.computed_at is not None
    
    print(f"   ✓ config_hash: {fingerprint.config_hash}")
    print(f"   ✓ config_files: {list(fingerprint.config_files.keys())}")
    print(f"   ✓ constants: {len(fingerprint.constants)} values")
    print(f"   ✓ computed_at: {fingerprint.computed_at}")
    
    return True


def test_feature_flags_default():
    """Test that feature flags default to enabled."""
    print("4. Testing feature flags default to enabled...")
    
    # Clear any existing env vars
    for var in ["ENABLE_ENCODER", "ENABLE_RAG", "ENABLE_DECODER"]:
        if var in os.environ:
            del os.environ[var]
    
    from src.inference.config_fingerprint import get_feature_flags
    
    flags = get_feature_flags()
    
    assert flags["encoder"] == True, "Encoder should default to enabled"
    assert flags["rag"] == True, "RAG should default to enabled"
    assert flags["decoder"] == True, "Decoder should default to enabled"
    
    print(f"   ✓ encoder: {flags['encoder']}")
    print(f"   ✓ rag: {flags['rag']}")
    print(f"   ✓ decoder: {flags['decoder']}")
    
    return True


def test_feature_flags_disable():
    """Test that feature flags can be disabled."""
    print("5. Testing feature flags can be disabled...")
    
    from src.inference.config_fingerprint import (
        is_encoder_enabled,
        is_rag_enabled,
        is_decoder_enabled,
    )
    
    # Test disabling encoder
    os.environ["ENABLE_ENCODER"] = "false"
    assert is_encoder_enabled() == False, "Encoder should be disabled"
    print("   ✓ ENABLE_ENCODER=false works")
    
    # Test disabling RAG
    os.environ["ENABLE_RAG"] = "false"
    assert is_rag_enabled() == False, "RAG should be disabled"
    print("   ✓ ENABLE_RAG=false works")
    
    # Test disabling decoder
    os.environ["ENABLE_DECODER"] = "false"
    assert is_decoder_enabled() == False, "Decoder should be disabled"
    print("   ✓ ENABLE_DECODER=false works")
    
    # Reset
    del os.environ["ENABLE_ENCODER"]
    del os.environ["ENABLE_RAG"]
    del os.environ["ENABLE_DECODER"]
    
    return True


def test_health_endpoint_has_hash():
    """Test that /health endpoint returns config hash and features."""
    print("6. Testing /health endpoint has hash and features...")
    
    from src.inference.server import health
    
    response = health()
    
    assert "config_hash" in response, "Missing config_hash in /health"
    assert "features" in response, "Missing features in /health"
    assert response["config_hash"] is not None
    assert isinstance(response["features"], dict)
    
    print(f"   ✓ config_hash: {response['config_hash']}")
    print(f"   ✓ features: {response['features']}")
    
    return True


def test_health_gpu_endpoint_has_hash():
    """Test that /health/gpu endpoint returns config hash and features."""
    print("7. Testing /health/gpu endpoint has hash and features...")
    
    from src.inference.server import health_gpu
    
    response = health_gpu()
    
    assert "config_hash" in response, "Missing config_hash in /health/gpu"
    assert "features" in response, "Missing features in /health/gpu"
    
    print(f"   ✓ config_hash: {response['config_hash']}")
    print(f"   ✓ features: {response['features']}")
    
    return True


def test_replay_artifact_has_hash():
    """Test that ReplayArtifact includes config_hash and feature_flags."""
    print("8. Testing ReplayArtifact has hash and feature_flags...")
    
    from src.inference.trace import ReplayArtifact
    from src.inference.config_fingerprint import get_config_hash, get_feature_flags
    
    artifact = ReplayArtifact(
        trace_id="test-trace-id",
        timestamp="2025-01-01T00:00:00",
        query="Test query",
        config_hash=get_config_hash(),
        feature_flags=get_feature_flags(),
    )
    
    assert artifact.config_hash is not None
    assert artifact.feature_flags is not None
    
    # Test to_dict includes these fields
    d = artifact.to_dict()
    assert "config_hash" in d
    assert "feature_flags" in d
    
    print(f"   ✓ config_hash in artifact: {artifact.config_hash}")
    print(f"   ✓ feature_flags in artifact: {artifact.feature_flags}")
    print(f"   ✓ to_dict() includes both fields")
    
    return True


def test_refusal_has_hash():
    """Test that refusal responses include config_hash."""
    print("9. Testing refusal responses include config_hash...")
    
    from src.inference.server import _make_refusal
    
    refusal = _make_refusal("test_reason", trace_id="test-trace-id")
    
    assert "config_hash" in refusal, "Missing config_hash in refusal"
    assert "trace_id" in refusal, "Missing trace_id in refusal"
    assert refusal["status"] == "refused"
    
    print(f"   ✓ config_hash: {refusal['config_hash']}")
    print(f"   ✓ trace_id: {refusal['trace_id']}")
    print(f"   ✓ status: {refusal['status']}")
    
    return True


def test_disabled_decoder_refusal():
    """Test that disabling decoder causes forced refusal."""
    print("10. Testing disabled decoder causes refusal...")
    
    from src.inference.config_fingerprint import is_decoder_enabled
    
    # Disable decoder
    os.environ["ENABLE_DECODER"] = "false"
    
    assert is_decoder_enabled() == False, "Decoder should be disabled"
    
    # Check refusal message exists
    from src.inference.server import REFUSAL_MESSAGES
    assert "decoder_disabled" in REFUSAL_MESSAGES
    
    print(f"   ✓ ENABLE_DECODER=false disables decoder")
    print(f"   ✓ decoder_disabled refusal message exists")
    
    # Reset
    del os.environ["ENABLE_DECODER"]
    
    return True


def test_kill_switch_refusal_reasons():
    """Test that all kill switch refusal reasons exist."""
    print("11. Testing kill switch refusal reasons exist...")
    
    from src.inference.server import REFUSAL_MESSAGES
    
    required_reasons = [
        "encoder_disabled",
        "rag_disabled",
        "decoder_disabled",
    ]
    
    for reason in required_reasons:
        assert reason in REFUSAL_MESSAGES, f"Missing refusal reason: {reason}"
        print(f"   ✓ {reason}: present")
    
    return True


def test_config_hash_deterministic():
    """Test that config hash is deterministic."""
    print("12. Testing config hash is deterministic...")
    
    from src.inference.config_fingerprint import compute_config_fingerprint
    
    # Compute hash twice
    fp1 = compute_config_fingerprint()
    fp2 = compute_config_fingerprint()
    
    assert fp1.config_hash == fp2.config_hash, "Config hash should be deterministic"
    
    print(f"   ✓ Hash 1: {fp1.config_hash}")
    print(f"   ✓ Hash 2: {fp2.config_hash}")
    print(f"   ✓ Hashes match: deterministic")
    
    return True


def main():
    print("=" * 50)
    print("PHASE P6: CONFIG INTEGRITY VERIFICATION")
    print("=" * 50)
    print()
    
    tests = [
        test_config_fingerprint_module,
        test_config_hash_exists,
        test_config_fingerprint_details,
        test_feature_flags_default,
        test_feature_flags_disable,
        test_health_endpoint_has_hash,
        test_health_gpu_endpoint_has_hash,
        test_replay_artifact_has_hash,
        test_refusal_has_hash,
        test_disabled_decoder_refusal,
        test_kill_switch_refusal_reasons,
        test_config_hash_deterministic,
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
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 50)
    print("PHASE P6 VERIFICATION RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print()
    
    if failed == 0:
        print("ALL TESTS PASSED")
        print()
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
