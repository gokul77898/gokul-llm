#!/usr/bin/env python3
"""
Phase P7-P10: Final Pre-GPU Hardening Verification Script

Verifies:
- P7: Offline replay module exists and works
- P8: Degraded modes and canary mode
- P9: Token accounting and abuse refusal reasons
- P10: Operational endpoints (/health/full, /ops/state, /ops/force-state)

Prints: PHASE P7-P10: FINAL HARDENING VERIFICATION - ALL TESTS PASSED
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_replay_module_exists():
    """Test that replay module exists and imports correctly."""
    print("1. Testing replay module exists...")
    
    from src.inference.replay import (
        TraceLoader,
        ReplayResult,
        StageResult,
        replay_trace,
        print_replay_summary,
    )
    
    print("   ✓ replay module imported successfully")
    print("   ✓ TraceLoader class exists")
    print("   ✓ ReplayResult dataclass exists")
    print("   ✓ replay_trace function exists")
    
    return True


def test_replay_trace_loader():
    """Test TraceLoader functionality."""
    print("2. Testing TraceLoader...")
    
    from src.inference.replay import TraceLoader
    
    loader = TraceLoader()
    
    # Should not crash even if no traces exist
    traces = loader.list_traces()
    print(f"   ✓ list_traces() returned {len(traces)} traces")
    
    # Load non-existent trace should return None
    result = loader.load("non-existent-trace-id")
    assert result is None, "Should return None for non-existent trace"
    print("   ✓ load() returns None for non-existent trace")
    
    return True


def test_model_state_degraded_modes():
    """Test ModelState has degraded modes."""
    print("3. Testing ModelState degraded modes...")
    
    from src.inference.local_models import ModelState
    
    # Check all states exist
    assert ModelState.UNINITIALIZED.value == "uninitialized"
    assert ModelState.LOADING.value == "loading"
    assert ModelState.READY.value == "ready"
    assert ModelState.DEGRADED.value == "degraded"
    assert ModelState.DEGRADED_ENCODER_ONLY.value == "degraded_encoder_only"
    assert ModelState.DEGRADED_RAG_ONLY.value == "degraded_rag_only"
    assert ModelState.FAILED.value == "failed"
    
    print("   ✓ UNINITIALIZED state exists")
    print("   ✓ LOADING state exists")
    print("   ✓ READY state exists")
    print("   ✓ DEGRADED state exists")
    print("   ✓ DEGRADED_ENCODER_ONLY state exists (P8)")
    print("   ✓ DEGRADED_RAG_ONLY state exists (P8)")
    print("   ✓ FAILED state exists")
    
    return True


def test_canary_mode():
    """Test canary mode functions."""
    print("4. Testing canary mode...")
    
    from src.inference.config_fingerprint import is_canary_mode, get_canary_limits
    
    # Default should be False
    os.environ.pop("CANARY_MODE", None)
    assert is_canary_mode() == False, "Canary mode should default to False"
    print("   ✓ is_canary_mode() defaults to False")
    
    # Test enabling
    os.environ["CANARY_MODE"] = "true"
    assert is_canary_mode() == True, "Canary mode should be True when enabled"
    print("   ✓ CANARY_MODE=true enables canary mode")
    
    # Test limits
    limits = get_canary_limits()
    assert "max_concurrent_requests" in limits
    assert "max_new_tokens" in limits
    print(f"   ✓ Canary limits: {limits}")
    
    # Reset
    del os.environ["CANARY_MODE"]
    
    return True


def test_ops_override():
    """Test ops override function."""
    print("5. Testing ops override...")
    
    from src.inference.config_fingerprint import is_ops_override_allowed
    
    # Default should be False
    os.environ.pop("ALLOW_OPS_OVERRIDE", None)
    assert is_ops_override_allowed() == False, "Ops override should default to False"
    print("   ✓ is_ops_override_allowed() defaults to False")
    
    # Test enabling
    os.environ["ALLOW_OPS_OVERRIDE"] = "true"
    assert is_ops_override_allowed() == True
    print("   ✓ ALLOW_OPS_OVERRIDE=true enables ops override")
    
    # Reset
    del os.environ["ALLOW_OPS_OVERRIDE"]
    
    return True


def test_refusal_reasons_p8_p9():
    """Test Phase P8 and P9 refusal reasons exist."""
    print("6. Testing P8/P9 refusal reasons...")
    
    from src.inference.server import REFUSAL_MESSAGES
    
    # P8 degraded mode reasons
    p8_reasons = [
        "degraded_encoder_only",
        "degraded_rag_only",
        "system_failed",
    ]
    
    for reason in p8_reasons:
        assert reason in REFUSAL_MESSAGES, f"Missing P8 reason: {reason}"
        print(f"   ✓ {reason}: present (P8)")
    
    # P9 cost/abuse reasons
    p9_reasons = [
        "token_budget_exceeded",
        "daily_budget_exceeded",
        "abuse_detected",
    ]
    
    for reason in p9_reasons:
        assert reason in REFUSAL_MESSAGES, f"Missing P9 reason: {reason}"
        print(f"   ✓ {reason}: present (P9)")
    
    return True


def test_health_full_endpoint():
    """Test /health/full endpoint."""
    print("7. Testing /health/full endpoint...")
    
    from src.inference.server import health_full
    
    response = health_full()
    
    # Check required fields
    required_fields = [
        "status",
        "config_hash",
        "model_state",
        "gpu",
        "gpu_memory",
        "models",
        "warmup",
        "features",
        "canary_mode",
        "concurrency",
        "timestamp",
    ]
    
    for field in required_fields:
        assert field in response, f"Missing field: {field}"
    
    print(f"   ✓ status: {response['status']}")
    print(f"   ✓ config_hash: {response['config_hash']}")
    print(f"   ✓ model_state: {response['model_state']}")
    print(f"   ✓ canary_mode: {response['canary_mode']}")
    print(f"   ✓ All {len(required_fields)} required fields present")
    
    return True


def test_ops_state_endpoint():
    """Test /ops/state endpoint."""
    print("8. Testing /ops/state endpoint...")
    
    from src.inference.server import ops_state
    
    response = ops_state()
    
    required_fields = [
        "model_state",
        "config_hash",
        "features",
        "canary_mode",
        "ops_override_allowed",
        "timestamp",
    ]
    
    for field in required_fields:
        assert field in response, f"Missing field: {field}"
        print(f"   ✓ {field}: present")
    
    return True


def test_ops_force_state_endpoint():
    """Test /ops/force-state endpoint."""
    print("9. Testing /ops/force-state endpoint...")
    
    from src.inference.server import ops_force_state, ForceStateRequest
    
    # Without override enabled, should fail
    os.environ.pop("ALLOW_OPS_OVERRIDE", None)
    
    req = ForceStateRequest(state="degraded", reason="test")
    response = ops_force_state(req)
    
    assert response["status"] == "error", "Should fail without override enabled"
    print("   ✓ Fails without ALLOW_OPS_OVERRIDE=true")
    
    # With override enabled
    os.environ["ALLOW_OPS_OVERRIDE"] = "true"
    
    response = ops_force_state(req)
    assert response["status"] == "ok", "Should succeed with override enabled"
    print("   ✓ Succeeds with ALLOW_OPS_OVERRIDE=true")
    
    # Reset
    del os.environ["ALLOW_OPS_OVERRIDE"]
    
    return True


def test_replay_artifact_has_new_fields():
    """Test ReplayArtifact has P8/P9 fields."""
    print("10. Testing ReplayArtifact has P8/P9 fields...")
    
    from src.inference.trace import ReplayArtifact
    from src.inference.config_fingerprint import get_config_hash, get_feature_flags
    
    artifact = ReplayArtifact(
        trace_id="test-trace-id",
        timestamp="2025-01-01T00:00:00",
        query="Test query",
        config_hash=get_config_hash(),
        feature_flags=get_feature_flags(),
        canary_mode=True,
        token_accounting={"input_tokens": 100, "output_tokens": 50, "total": 150},
    )
    
    assert artifact.canary_mode == True
    assert artifact.token_accounting is not None
    print("   ✓ canary_mode field exists (P8)")
    print("   ✓ token_accounting field exists (P9)")
    
    # Test to_dict includes these
    d = artifact.to_dict()
    assert "canary_mode" in d
    assert "token_accounting" in d
    print("   ✓ to_dict() includes canary_mode")
    print("   ✓ to_dict() includes token_accounting")
    
    return True


def test_replay_cli():
    """Test replay CLI can be invoked."""
    print("11. Testing replay CLI...")
    
    import subprocess
    
    # Run with --help to verify it works
    result = subprocess.run(
        [sys.executable, "-m", "src.inference.replay", "--help"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert "--trace-id" in result.stdout
    assert "--stage" in result.stdout
    print("   ✓ CLI --help works")
    print("   ✓ --trace-id argument documented")
    print("   ✓ --stage argument documented")
    
    return True


def test_registry_failure_reason():
    """Test LocalModelRegistry has failure_reason tracking."""
    print("12. Testing LocalModelRegistry failure tracking...")
    
    from src.inference.local_models import LocalModelRegistry
    
    registry = LocalModelRegistry()
    
    assert hasattr(registry, '_failure_reason')
    print("   ✓ _failure_reason attribute exists")
    
    # Test _set_degraded_encoder_only method exists
    assert hasattr(registry, '_set_degraded_encoder_only')
    print("   ✓ _set_degraded_encoder_only method exists (P8)")
    
    return True


def main():
    print("=" * 60)
    print("PHASE P7-P10: FINAL HARDENING VERIFICATION")
    print("=" * 60)
    print()
    
    tests = [
        test_replay_module_exists,
        test_replay_trace_loader,
        test_model_state_degraded_modes,
        test_canary_mode,
        test_ops_override,
        test_refusal_reasons_p8_p9,
        test_health_full_endpoint,
        test_ops_state_endpoint,
        test_ops_force_state_endpoint,
        test_replay_artifact_has_new_fields,
        test_replay_cli,
        test_registry_failure_reason,
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
    
    print("=" * 60)
    print("PHASE P7-P10 VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print()
    
    if failed == 0:
        print("ALL TESTS PASSED")
        print()
        print("System is GPU-READY:")
        print("  ✓ Offline replay & debugging (P7)")
        print("  ✓ Degraded & canary modes (P8)")
        print("  ✓ Cost & abuse safety (P9)")
        print("  ✓ Operational readiness (P10)")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
