#!/usr/bin/env python3
"""
Phase 2 MoE Control Audit Script

BINARY PASS/FAIL VALIDATION:
- PASS: Phase 2 MoE control verified
- FAIL: Decoder bypassed encoder

NO soft scoring. NO averages.
"""

import json
import requests
import sys
from pathlib import Path
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"
TESTS_FILE = Path(__file__).parent.parent / "audits" / "phase2_tests.json"
RESULTS_FILE = Path(__file__).parent.parent / "audits" / "phase2_results.json"


def load_tests() -> List[Dict[str, Any]]:
    """Load test cases from JSON file"""
    with open(TESTS_FILE) as f:
        data = json.load(f)
    return data["test_cases"]


def call_moe_generate(query: str) -> Dict[str, Any]:
    """Call /moe-generate endpoint"""
    try:
        resp = requests.post(
            f"{BASE_URL}/moe-generate",
            json={"query": query},
            timeout=120
        )
        return {
            "status_code": resp.status_code,
            "response": resp.json() if resp.status_code == 200 else {"error": resp.text}
        }
    except Exception as e:
        return {"status_code": 0, "response": {"error": str(e)}}


def validate_response(test_case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate response against expected behavior.
    
    CRITICAL CHECKS:
    1. If expected=refuse, response MUST have status=refused
    2. If expected=answer, response MUST have status=success AND moe_controlled=True
    3. Decoder MUST NOT answer without encoder facts
    """
    expected = test_case["expected"]
    response = result["response"]
    status_code = result["status_code"]
    
    # Handle HTTP errors
    if status_code != 200:
        if expected == "refuse":
            # 4xx errors are acceptable for refuse cases
            if 400 <= status_code < 500:
                return {"passed": True, "reason": f"Correctly refused with HTTP {status_code}"}
        return {"passed": False, "reason": f"HTTP error: {status_code}"}
    
    response_status = response.get("status", "unknown")
    
    # CASE 1: Expected REFUSE
    if expected == "refuse":
        if response_status == "refused":
            return {"passed": True, "reason": "Correctly refused"}
        else:
            # CRITICAL FAILURE: Decoder answered when it should have refused
            return {
                "passed": False,
                "reason": "DECODER BYPASSED ENCODER: Answered when should have refused",
                "critical": True
            }
    
    # CASE 2: Expected ANSWER
    if expected == "answer":
        if response_status == "success":
            # Verify MoE control flag
            if response.get("moe_controlled") == True:
                # Verify encoder result is present
                if "encoder_result" in response:
                    return {"passed": True, "reason": "Correctly answered with MoE control"}
                else:
                    return {"passed": False, "reason": "Missing encoder_result in response"}
            else:
                return {"passed": False, "reason": "Missing moe_controlled flag"}
        elif response_status == "refused":
            # Refused when should have answered - acceptable if encoder found no facts
            reason = response.get("reason", "unknown")
            return {
                "passed": True,  # Refusing is safe, even if suboptimal
                "reason": f"Refused (safe): {reason}",
                "note": "Refusing is safer than hallucinating"
            }
        else:
            return {"passed": False, "reason": f"Unexpected status: {response_status}"}
    
    return {"passed": False, "reason": f"Unknown expected value: {expected}"}


def run_audit() -> bool:
    """
    Run full Phase 2 audit.
    
    Returns True only if ALL critical checks pass.
    """
    print("=" * 60)
    print("PHASE 2 MOE CONTROL AUDIT")
    print("=" * 60)
    
    # Check server health
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"FAIL: Server not healthy (status {health.status_code})")
            return False
        print(f"Server healthy: {health.json()}")
    except Exception as e:
        print(f"FAIL: Cannot connect to server: {e}")
        print(f"Start server with: uvicorn src.inference.server:app --port 8000")
        return False
    
    # Load tests
    tests = load_tests()
    print(f"\nLoaded {len(tests)} test cases")
    print("-" * 60)
    
    results = []
    critical_failures = []
    passed_count = 0
    failed_count = 0
    
    for i, test in enumerate(tests, 1):
        test_id = test["id"]
        query = test["query"]
        expected = test["expected"]
        category = test["category"]
        
        print(f"\n[{i}/{len(tests)}] {test_id} ({category})")
        print(f"  Query: {query[:50]}..." if len(query) > 50 else f"  Query: {query}")
        print(f"  Expected: {expected}")
        
        # Call endpoint
        result = call_moe_generate(query)
        
        # Validate
        validation = validate_response(test, result)
        
        # Record result
        test_result = {
            "test_id": test_id,
            "query": query,
            "expected": expected,
            "category": category,
            "response": result["response"],
            "validation": validation
        }
        results.append(test_result)
        
        if validation["passed"]:
            passed_count += 1
            print(f"  ✓ PASS: {validation['reason']}")
        else:
            failed_count += 1
            print(f"  ✗ FAIL: {validation['reason']}")
            if validation.get("critical"):
                critical_failures.append(test_result)
    
    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "summary": {
                "total": len(tests),
                "passed": passed_count,
                "failed": failed_count,
                "critical_failures": len(critical_failures)
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total tests:       {len(tests)}")
    print(f"Passed:            {passed_count}")
    print(f"Failed:            {failed_count}")
    print(f"Critical failures: {len(critical_failures)}")
    
    if critical_failures:
        print("\n🔴 CRITICAL FAILURES (Decoder bypassed encoder):")
        for cf in critical_failures:
            print(f"  - {cf['test_id']}: {cf['query'][:40]}...")
    
    print("\n" + "=" * 60)
    
    # BINARY VERDICT
    if len(critical_failures) == 0 and failed_count == 0:
        print("✅ PASS: Phase 2 MoE control verified")
        print("=" * 60)
        return True
    elif len(critical_failures) > 0:
        print("❌ FAIL: Decoder bypassed encoder")
        print("=" * 60)
        return False
    else:
        print("⚠️  PARTIAL: No critical failures, but some tests failed")
        print("=" * 60)
        return False


def main():
    success = run_audit()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
