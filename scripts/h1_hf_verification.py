#!/usr/bin/env python3
"""
Phase H1: Hugging Face CPU Deployment Verification

Tests:
1. Server boot
2. No model auto-load
3. Health endpoints
4. RAG-only refusal
5. Trace + config_hash present
6. Replay works

Output: PASS or FAIL only
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_server_boot():
    """Test server can be imported without errors."""
    try:
        from src.inference.server import app, MODEL_REGISTRY
        return True, "Server imports OK"
    except Exception as e:
        return False, f"Server import failed: {e}"


def test_no_model_autoload():
    """Test no models are auto-loaded."""
    from src.inference.server import MODEL_REGISTRY
    from src.inference.local_models import ModelState
    
    state = MODEL_REGISTRY.get_state()
    if state in [ModelState.UNINITIALIZED, ModelState.DEGRADED_RAG_ONLY]:
        return True, f"state={state.value}"
    return False, f"Unexpected state: {state.value}"


def test_feature_flags():
    """Test H1 feature flag defaults."""
    # Clear env vars to test defaults
    for var in ["ENABLE_ENCODER", "ENABLE_DECODER", "ENABLE_RAG", "CANARY_MODE"]:
        os.environ.pop(var, None)
    
    from src.inference.config_fingerprint import (
        is_encoder_enabled,
        is_decoder_enabled,
        is_rag_enabled,
        is_canary_mode,
    )
    
    if is_encoder_enabled():
        return False, "Encoder should be disabled by default"
    if is_decoder_enabled():
        return False, "Decoder should be disabled by default"
    if not is_rag_enabled():
        return False, "RAG should be enabled by default"
    if not is_canary_mode():
        return False, "Canary mode should be enabled by default (H1)"
    
    return True, "encoder=false, decoder=false, rag=true, canary=true"


def test_health_endpoint():
    """Test /health endpoint."""
    from src.inference.server import health
    
    resp = health()
    if "config_hash" not in resp:
        return False, "Missing config_hash"
    if "features" not in resp:
        return False, "Missing features"
    return True, f"config_hash={resp['config_hash']}"


def test_health_gpu_endpoint():
    """Test /health/gpu endpoint."""
    from src.inference.server import health_gpu
    
    resp = health_gpu()
    if "model_state" not in resp:
        return False, "Missing model_state"
    if "config_hash" not in resp:
        return False, "Missing config_hash"
    return True, f"model_state={resp['model_state']}"


def test_health_full_endpoint():
    """Test /health/full endpoint."""
    from src.inference.server import health_full
    
    resp = health_full()
    if "gpu_memory" not in resp:
        return False, "Missing gpu_memory"
    if "features" not in resp:
        return False, "Missing features"
    if "canary_mode" not in resp:
        return False, "Missing canary_mode"
    return True, f"status={resp['status']}, canary={resp['canary_mode']}"


def test_refusal_structure():
    """Test refusal includes required fields."""
    from src.inference.server import _make_refusal
    
    refusal = _make_refusal("encoder_disabled", trace_id="test-trace")
    
    required = ["status", "reason", "message", "config_hash", "trace_id"]
    for field in required:
        if field not in refusal:
            return False, f"Missing field: {field}"
    
    if refusal["status"] != "refused":
        return False, f"Wrong status: {refusal['status']}"
    
    return True, f"reason={refusal['reason']}"


def test_replay_module():
    """Test replay module works."""
    try:
        from src.inference.replay import replay_trace, TraceLoader
        
        loader = TraceLoader()
        # Should not crash
        traces = loader.list_traces()
        
        return True, f"Replay module OK, {len(traces)} traces"
    except Exception as e:
        return False, f"Replay failed: {e}"


def test_replay_cli():
    """Test replay CLI."""
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "-m", "src.inference.replay", "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    
    if result.returncode != 0:
        return False, "CLI failed"
    if "--trace-id" not in result.stdout:
        return False, "Missing --trace-id"
    
    return True, "CLI OK"


def test_log_directories():
    """Test log directories exist or can be created."""
    logs_dir = PROJECT_ROOT / "logs" / "traces"
    replay_dir = PROJECT_ROOT / "data" / "replay"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    
    if not logs_dir.exists():
        return False, "Cannot create logs/traces"
    if not replay_dir.exists():
        return False, "Cannot create data/replay"
    
    return True, "Directories OK"


def test_config_hash_visibility():
    """Test config hash is in all endpoints."""
    from src.inference.config_fingerprint import get_config_hash
    from src.inference.server import health, health_gpu, health_full
    
    config_hash = get_config_hash()
    
    h = health()
    if h.get("config_hash") != config_hash:
        return False, "Hash mismatch in /health"
    
    hg = health_gpu()
    if hg.get("config_hash") != config_hash:
        return False, "Hash mismatch in /health/gpu"
    
    hf = health_full()
    if hf.get("config_hash") != config_hash:
        return False, "Hash mismatch in /health/full"
    
    return True, f"hash={config_hash}"


def main():
    tests = [
        ("server_boot", test_server_boot),
        ("no_model_autoload", test_no_model_autoload),
        ("feature_flags", test_feature_flags),
        ("health_endpoint", test_health_endpoint),
        ("health_gpu_endpoint", test_health_gpu_endpoint),
        ("health_full_endpoint", test_health_full_endpoint),
        ("refusal_structure", test_refusal_structure),
        ("replay_module", test_replay_module),
        ("replay_cli", test_replay_cli),
        ("log_directories", test_log_directories),
        ("config_hash_visibility", test_config_hash_visibility),
    ]
    
    all_passed = True
    
    for name, test_fn in tests:
        try:
            passed, details = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}: {details}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            all_passed = False
    
    print()
    if all_passed:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
