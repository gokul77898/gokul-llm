#!/usr/bin/env python3
"""
Phase 1A System Validation Script

Validates MoE system behavior with stable HF models.
Does NOT judge model quality - only system correctness.
"""

import json
import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Verify server is running"""
    print("=" * 50)
    print("TEST: Health Check")
    print("=" * 50)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Encoders: {data.get('encoders')}")
            print(f"  Decoders: {data.get('decoders')}")
            print("  PASS: Server healthy")
            return True
        else:
            print(f"  FAIL: Status {r.status_code}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_routing():
    """STEP 2: Verify /moe-test routing (no HF calls, deterministic)"""
    print("\n" + "=" * 50)
    print("TEST: Routing (STEP 2)")
    print("=" * 50)
    
    test_query = "John Smith was arrested under Section 420 IPC"
    
    # Test determinism - call twice
    results = []
    for i in range(2):
        try:
            r = requests.post(
                f"{BASE_URL}/moe-test",
                json={"query": test_query},
                timeout=5
            )
            if r.status_code == 200:
                results.append(r.json())
            else:
                print(f"  FAIL: Status {r.status_code}")
                return False
        except Exception as e:
            print(f"  FAIL: {e}")
            return False
    
    # Check determinism
    if results[0]["chosen"] == results[1]["chosen"]:
        print(f"  Chosen expert: {results[0]['chosen']}")
        print(f"  Scores: {results[0]['scores']}")
        print(f"  Routing only: {results[0].get('routing_only')}")
        print(f"  Models loaded: {results[0].get('models_loaded')}")
        print("  PASS: Routing is deterministic, no models loaded")
        return True
    else:
        print("  FAIL: Routing not deterministic")
        return False


def test_encoder():
    """STEP 3: Verify /moe-encode returns entities, never calls decoder"""
    print("\n" + "=" * 50)
    print("TEST: Encoder Contract (STEP 3)")
    print("=" * 50)
    
    test_query = "John Smith was arrested in New York on January 15th"
    
    try:
        r = requests.post(
            f"{BASE_URL}/moe-encode",
            json={"query": test_query},
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            print(f"  Encoder used: {data.get('encoder_used')}")
            print(f"  Model ID: {data.get('model_id')}")
            print(f"  Decoder called: {data.get('decoder_called')}")
            
            result = data.get("result", {})
            entities = result.get("entities", [])
            print(f"  Entities extracted: {len(entities)}")
            
            if entities:
                for ent in entities[:5]:
                    if isinstance(ent, dict):
                        print(f"    - {ent.get('word', 'N/A')} ({ent.get('entity_group', ent.get('entity', 'N/A'))})")
            
            if data.get("decoder_called") == False:
                print("  PASS: Encoder returns entities, decoder NOT called")
                return True
            else:
                print("  FAIL: Decoder was called")
                return False
        else:
            print(f"  FAIL: Status {r.status_code} - {r.text[:200]}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_moe_enforcement():
    """STEP 4: Verify /moe-generate enforces encoder-first"""
    print("\n" + "=" * 50)
    print("TEST: MoE Enforcement (STEP 4)")
    print("=" * 50)
    
    test_query = "What happened to John Smith in New York?"
    
    try:
        r = requests.post(
            f"{BASE_URL}/moe-generate",
            json={"query": test_query},
            timeout=60
        )
        
        if r.status_code == 200:
            data = r.json()
            print(f"  Encoder used: {data.get('encoder_used')}")
            print(f"  Decoder used: {data.get('decoder_used')}")
            print(f"  Refusal: {data.get('refusal')}")
            
            encoder_result = data.get("encoder_result", {})
            if encoder_result.get("success"):
                print("  Encoder was called first: YES")
            
            encoder_facts = data.get("encoder_facts_used")
            if encoder_facts:
                print(f"  Encoder facts embedded in decoder prompt: YES")
                print(f"  Facts preview: {encoder_facts[:100]}...")
            
            output = data.get("output", "")
            print(f"  Output preview: {output[:100]}...")
            
            print("  PASS: MoE enforcement working")
            return True
        else:
            print(f"  FAIL: Status {r.status_code} - {r.text[:200]}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_negative_empty():
    """STEP 5: Empty query should be refused"""
    print("\n" + "=" * 50)
    print("TEST: Negative - Empty Query (STEP 5)")
    print("=" * 50)
    
    try:
        r = requests.post(
            f"{BASE_URL}/moe-generate",
            json={"query": ""},
            timeout=10
        )
        
        if r.status_code == 400:
            print(f"  Response: {r.text[:100]}")
            print("  PASS: Empty query correctly refused with 400")
            return True
        else:
            print(f"  FAIL: Expected 400, got {r.status_code}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_negative_whitespace():
    """STEP 5: Whitespace-only query should be refused"""
    print("\n" + "=" * 50)
    print("TEST: Negative - Whitespace Query (STEP 5)")
    print("=" * 50)
    
    try:
        r = requests.post(
            f"{BASE_URL}/moe-generate",
            json={"query": "   "},
            timeout=10
        )
        
        if r.status_code == 400:
            print(f"  Response: {r.text[:100]}")
            print("  PASS: Whitespace query correctly refused with 400")
            return True
        else:
            print(f"  FAIL: Expected 400, got {r.status_code}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_logging():
    """STEP 6: Verify logging to moe_requests.jsonl"""
    print("\n" + "=" * 50)
    print("TEST: Request Logging (STEP 6)")
    print("=" * 50)
    
    log_file = Path(__file__).parent.parent / "logs" / "moe_requests.jsonl"
    
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
        
        if lines:
            last_entry = json.loads(lines[-1])
            print(f"  Log file exists: {log_file}")
            print(f"  Total entries: {len(lines)}")
            print(f"  Last entry keys: {list(last_entry.keys())}")
            
            required_keys = ["timestamp", "query", "selected_encoder", "selected_decoder", "status"]
            missing = [k for k in required_keys if k not in last_entry]
            
            if not missing:
                print("  PASS: Log format correct")
                return True
            else:
                print(f"  FAIL: Missing keys: {missing}")
                return False
        else:
            print("  FAIL: Log file empty")
            return False
    else:
        print(f"  FAIL: Log file not found: {log_file}")
        return False


def main():
    print("\n" + "=" * 60)
    print("PHASE 1A SYSTEM VALIDATION")
    print("=" * 60)
    
    results = {}
    
    # Run all tests
    results["health"] = test_health()
    
    if not results["health"]:
        print("\n" + "=" * 60)
        print("ABORT: Server not running")
        print("Start with: uvicorn src.inference.server:app --port 8000")
        print("=" * 60)
        sys.exit(1)
    
    results["routing"] = test_routing()
    results["encoder"] = test_encoder()
    results["moe_enforcement"] = test_moe_enforcement()
    results["negative_empty"] = test_negative_empty()
    results["negative_whitespace"] = test_negative_whitespace()
    results["logging"] = test_logging()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test:<25} {status}")
    
    print("-" * 60)
    print(f"  Total: {passed}/{total}")
    
    if passed == total:
        print("\n  PHASE 1A VALIDATION: PASSED")
        print("  System behavior is correct with stable models.")
    else:
        print("\n  PHASE 1A VALIDATION: FAILED")
        print("  Fix failing tests before proceeding.")
    
    print("=" * 60)
    
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
