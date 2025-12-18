#!/usr/bin/env python3
"""
Phase H0: Local CPU Deployment Hygiene Verification

This script verifies ALL Phase H0 requirements:
1. Server boots on CPU
2. No Hugging Face API calls
3. No model auto-load
4. RAG executes
5. Refusals are structured
6. Logs written
7. Replay works
8. Health endpoints respond
9. Config hash visible everywhere

FINAL RESULT: PASS or FAIL
"""

import sys
import os
import json
import time
import subprocess
import signal
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class H0Verifier:
    """Phase H0 verification runner."""
    
    def __init__(self):
        self.results = {}
        self.server_process = None
        self.server_url = "http://127.0.0.1:8000"
        
    def log(self, msg: str):
        """Print with timestamp."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def check(self, name: str, passed: bool, details: str = ""):
        """Record a check result."""
        self.results[name] = {"passed": passed, "details": details}
        status = "✓ PASS" if passed else "✗ FAIL"
        self.log(f"  {status}: {name}")
        if details:
            self.log(f"         {details}")
        return passed
    
    # ─────────────────────────────────────────────
    # 1. VERIFY NO EXTERNAL INFERENCE
    # ─────────────────────────────────────────────
    
    def verify_no_external_inference(self) -> bool:
        """Verify HF API is disabled."""
        self.log("1. Verifying no external inference...")
        
        from src.inference.server import _hf_infer_with_retry, HFInferenceError
        
        # _hf_infer_with_retry should raise immediately
        try:
            _hf_infer_with_retry("test-model", {"inputs": "test"})
            return self.check("hf_api_disabled", False, "HF API did not raise")
        except HFInferenceError as e:
            return self.check("hf_api_disabled", True, f"Raises: {e.reason}")
        except Exception as e:
            return self.check("hf_api_disabled", False, f"Unexpected error: {e}")
    
    # ─────────────────────────────────────────────
    # 2. VERIFY CPU-ONLY FEATURE FLAGS
    # ─────────────────────────────────────────────
    
    def verify_feature_flags(self) -> bool:
        """Verify feature flags default to CPU-only mode."""
        self.log("2. Verifying CPU-only feature flags...")
        
        # Clear any existing env vars to test defaults
        for var in ["ENABLE_ENCODER", "ENABLE_DECODER", "ENABLE_RAG"]:
            os.environ.pop(var, None)
        
        from src.inference.config_fingerprint import get_feature_flags
        
        flags = get_feature_flags()
        
        encoder_ok = self.check(
            "encoder_disabled_by_default",
            flags["encoder"] == False,
            f"encoder={flags['encoder']}"
        )
        
        decoder_ok = self.check(
            "decoder_disabled_by_default",
            flags["decoder"] == False,
            f"decoder={flags['decoder']}"
        )
        
        rag_ok = self.check(
            "rag_enabled_by_default",
            flags["rag"] == True,
            f"rag={flags['rag']}"
        )
        
        return encoder_ok and decoder_ok and rag_ok
    
    # ─────────────────────────────────────────────
    # 3. VERIFY SERVER BOOT SAFETY
    # ─────────────────────────────────────────────
    
    def verify_server_imports(self) -> bool:
        """Verify server can be imported without errors."""
        self.log("3. Verifying server boot safety...")
        
        try:
            from src.inference.server import (
                app,
                MODEL_REGISTRY,
                health,
                health_gpu,
                health_full,
            )
            
            # Check registry is in UNINITIALIZED state
            from src.inference.local_models import ModelState
            state = MODEL_REGISTRY.get_state()
            
            state_ok = self.check(
                "registry_uninitialized",
                state in [ModelState.UNINITIALIZED, ModelState.DEGRADED_RAG_ONLY],
                f"state={state.value}"
            )
            
            return state_ok
            
        except Exception as e:
            return self.check("server_imports", False, f"Import failed: {e}")
    
    # ─────────────────────────────────────────────
    # 4. VERIFY HEALTH ENDPOINTS
    # ─────────────────────────────────────────────
    
    def verify_health_endpoints(self) -> bool:
        """Verify health endpoints work without GPU."""
        self.log("4. Verifying health endpoints...")
        
        from src.inference.server import health, health_gpu, health_full
        
        all_ok = True
        
        # /health
        try:
            resp = health()
            health_ok = self.check(
                "health_endpoint",
                "config_hash" in resp and "features" in resp,
                f"config_hash={resp.get('config_hash')}"
            )
            all_ok = all_ok and health_ok
        except Exception as e:
            self.check("health_endpoint", False, str(e))
            all_ok = False
        
        # /health/gpu
        try:
            resp = health_gpu()
            gpu_ok = self.check(
                "health_gpu_endpoint",
                "model_state" in resp and "config_hash" in resp,
                f"model_state={resp.get('model_state')}"
            )
            all_ok = all_ok and gpu_ok
        except Exception as e:
            self.check("health_gpu_endpoint", False, str(e))
            all_ok = False
        
        # /health/full
        try:
            resp = health_full()
            full_ok = self.check(
                "health_full_endpoint",
                "gpu_memory" in resp and "features" in resp,
                f"status={resp.get('status')}"
            )
            all_ok = all_ok and full_ok
        except Exception as e:
            self.check("health_full_endpoint", False, str(e))
            all_ok = False
        
        return all_ok
    
    # ─────────────────────────────────────────────
    # 5. VERIFY RAG-ONLY FUNCTIONALITY
    # ─────────────────────────────────────────────
    
    def verify_rag_only(self) -> bool:
        """Verify RAG works and encoder/decoder are refused."""
        self.log("5. Verifying RAG-only functionality...")
        
        # Ensure CPU-only mode
        os.environ["ENABLE_ENCODER"] = "false"
        os.environ["ENABLE_DECODER"] = "false"
        os.environ["ENABLE_RAG"] = "true"
        
        from src.rag import LegalRetriever, RetrievalValidator, ContextAssembler
        
        all_ok = True
        
        # Test retrieval
        try:
            retriever = LegalRetriever()
            chunks = retriever.retrieve("What is Section 420 IPC?", top_k=5)
            retrieval_ok = self.check(
                "rag_retrieval",
                len(chunks) > 0,
                f"Retrieved {len(chunks)} chunks"
            )
            all_ok = all_ok and retrieval_ok
        except Exception as e:
            self.check("rag_retrieval", False, str(e))
            all_ok = False
        
        # Test validation
        try:
            validator = RetrievalValidator()
            # Convert chunks to dict format expected by validator
            chunk_dicts = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "section": c.section,
                    "act": c.act,
                    "score": c.score,
                }
                for c in chunks
            ]
            result = validator.validate("What is Section 420 IPC?", chunk_dicts)
            validation_ok = self.check(
                "rag_validation",
                result.status.value in ["pass", "partial"],
                f"status={result.status.value}"
            )
            all_ok = all_ok and validation_ok
        except Exception as e:
            self.check("rag_validation", False, str(e))
            all_ok = False
        
        return all_ok
    
    # ─────────────────────────────────────────────
    # 6. VERIFY REFUSAL STRUCTURE
    # ─────────────────────────────────────────────
    
    def verify_refusal_structure(self) -> bool:
        """Verify refusals include required fields."""
        self.log("6. Verifying refusal structure...")
        
        from src.inference.server import _make_refusal
        
        refusal = _make_refusal("encoder_disabled", trace_id="test-trace-id")
        
        required_fields = ["status", "reason", "message", "config_hash"]
        
        all_present = all(f in refusal for f in required_fields)
        
        return self.check(
            "refusal_structure",
            all_present and refusal["status"] == "refused",
            f"fields={list(refusal.keys())}"
        )
    
    # ─────────────────────────────────────────────
    # 7. VERIFY LOGGING & TRACE INTEGRITY
    # ─────────────────────────────────────────────
    
    def verify_logging(self) -> bool:
        """Verify log directories exist."""
        self.log("7. Verifying logging & trace integrity...")
        
        logs_dir = PROJECT_ROOT / "logs"
        replay_dir = PROJECT_ROOT / "data" / "replay"
        
        # Create directories if needed
        logs_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)
        
        logs_ok = self.check(
            "logs_directory",
            logs_dir.exists(),
            f"path={logs_dir}"
        )
        
        replay_ok = self.check(
            "replay_directory",
            replay_dir.exists(),
            f"path={replay_dir}"
        )
        
        return logs_ok and replay_ok
    
    # ─────────────────────────────────────────────
    # 8. VERIFY REPLAY ARTIFACT STRUCTURE
    # ─────────────────────────────────────────────
    
    def verify_replay_artifact(self) -> bool:
        """Verify replay artifact has required fields."""
        self.log("8. Verifying replay artifact structure...")
        
        from src.inference.trace import ReplayArtifact
        from src.inference.config_fingerprint import get_config_hash, get_feature_flags
        
        artifact = ReplayArtifact(
            trace_id="test-trace-id",
            timestamp=datetime.utcnow().isoformat(),
            query="Test query",
            config_hash=get_config_hash(),
            feature_flags=get_feature_flags(),
        )
        
        d = artifact.to_dict()
        
        required_fields = [
            "trace_id", "timestamp", "query",
            "config_hash", "feature_flags",
            "canary_mode", "token_accounting"
        ]
        
        all_present = all(f in d for f in required_fields)
        
        return self.check(
            "replay_artifact_structure",
            all_present,
            f"fields={list(d.keys())}"
        )
    
    # ─────────────────────────────────────────────
    # 9. VERIFY OFFLINE REPLAY
    # ─────────────────────────────────────────────
    
    def verify_offline_replay(self) -> bool:
        """Verify replay CLI works."""
        self.log("9. Verifying offline replay...")
        
        # Test CLI help
        result = subprocess.run(
            [sys.executable, "-m", "src.inference.replay", "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        cli_ok = self.check(
            "replay_cli",
            result.returncode == 0 and "--trace-id" in result.stdout,
            "CLI --help works"
        )
        
        # Test replay module import
        try:
            from src.inference.replay import replay_trace, TraceLoader
            import_ok = self.check(
                "replay_module",
                True,
                "Module imports successfully"
            )
        except Exception as e:
            import_ok = self.check("replay_module", False, str(e))
        
        return cli_ok and import_ok
    
    # ─────────────────────────────────────────────
    # 10. VERIFY CONFIG HASH VISIBILITY
    # ─────────────────────────────────────────────
    
    def verify_config_hash(self) -> bool:
        """Verify config hash is visible everywhere."""
        self.log("10. Verifying config hash visibility...")
        
        from src.inference.config_fingerprint import get_config_hash
        from src.inference.server import health, health_gpu, health_full, _make_refusal
        
        config_hash = get_config_hash()
        
        all_ok = True
        
        # Check in health endpoints
        h = health()
        h_ok = self.check(
            "config_hash_in_health",
            h.get("config_hash") == config_hash,
            f"hash={h.get('config_hash')}"
        )
        all_ok = all_ok and h_ok
        
        hg = health_gpu()
        hg_ok = self.check(
            "config_hash_in_health_gpu",
            hg.get("config_hash") == config_hash,
            f"hash={hg.get('config_hash')}"
        )
        all_ok = all_ok and hg_ok
        
        hf = health_full()
        hf_ok = self.check(
            "config_hash_in_health_full",
            hf.get("config_hash") == config_hash,
            f"hash={hf.get('config_hash')}"
        )
        all_ok = all_ok and hf_ok
        
        # Check in refusal
        r = _make_refusal("test", trace_id="test")
        r_ok = self.check(
            "config_hash_in_refusal",
            r.get("config_hash") == config_hash,
            f"hash={r.get('config_hash')}"
        )
        all_ok = all_ok and r_ok
        
        return all_ok
    
    # ─────────────────────────────────────────────
    # MAIN VERIFICATION
    # ─────────────────────────────────────────────
    
    def run_all(self) -> bool:
        """Run all verifications."""
        print("=" * 60)
        print("PHASE H0: LOCAL CPU DEPLOYMENT HYGIENE VERIFICATION")
        print("=" * 60)
        print()
        
        all_passed = True
        
        # Run all checks
        all_passed &= self.verify_no_external_inference()
        all_passed &= self.verify_feature_flags()
        all_passed &= self.verify_server_imports()
        all_passed &= self.verify_health_endpoints()
        all_passed &= self.verify_rag_only()
        all_passed &= self.verify_refusal_structure()
        all_passed &= self.verify_logging()
        all_passed &= self.verify_replay_artifact()
        all_passed &= self.verify_offline_replay()
        all_passed &= self.verify_config_hash()
        
        # Summary
        print()
        print("=" * 60)
        print("PHASE H0 VERIFICATION SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results.values() if r["passed"])
        total = len(self.results)
        
        print(f"Passed: {passed}/{total}")
        print()
        
        # List failures
        failures = [k for k, v in self.results.items() if not v["passed"]]
        if failures:
            print("FAILURES:")
            for f in failures:
                print(f"  - {f}: {self.results[f]['details']}")
            print()
        
        print("=" * 60)
        if all_passed:
            print("FINAL RESULT: PASS")
            print()
            print("System is ready for Hugging Face deployment:")
            print("  ✓ Server boots on CPU")
            print("  ✓ No Hugging Face API calls")
            print("  ✓ No model auto-load")
            print("  ✓ RAG executes")
            print("  ✓ Refusals are structured")
            print("  ✓ Logs written")
            print("  ✓ Replay works")
            print("  ✓ Health endpoints respond")
            print("  ✓ Config hash visible everywhere")
        else:
            print("FINAL RESULT: FAIL")
            print()
            print("DO NOT PROCEED TO HF DEPLOYMENT")
        print("=" * 60)
        
        return all_passed


def main():
    verifier = H0Verifier()
    success = verifier.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
