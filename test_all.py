#!/usr/bin/env python
"""
Comprehensive Test Runner for Legal AI System

This script runs all tests and generates a detailed report.
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and return results"""
    print(f"\n{'='*70}")
    print(f"🧪 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        duration = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
        print(f"\n{status} (Duration: {duration:.2f}s)")
        
        return result.returncode == 0, duration
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, 0


def main():
    """Main test runner"""
    print("\n" + "="*70)
    print("🚀 LEGAL AI SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = []
    total_duration = 0
    
    # Test configurations
    tests = [
        {
            "cmd": ["pytest", "tests/test_mamba.py", "-v"],
            "desc": "Mamba Architecture Tests",
            "category": "mamba"
        },
        {
            "cmd": ["pytest", "tests/test_transfer.py", "-v"],
            "desc": "Transfer Learning Tests",
            "category": "transfer"
        },
        {
            "cmd": ["pytest", "tests/test_rag.py", "-v", "-m", "not gpu"],
            "desc": "RAG System Tests (CPU only)",
            "category": "rag"
        },
        {
            "cmd": ["pytest", "tests/test_rl.py", "-v", "-k", "not agent"],
            "desc": "RL System Tests (Core components)",
            "category": "rl"
        },
    ]
    
    # Run each test suite
    for test_config in tests:
        success, duration = run_command(test_config["cmd"], test_config["desc"])
        results.append({
            "category": test_config["category"],
            "description": test_config["desc"],
            "success": success,
            "duration": duration
        })
        total_duration += duration
    
    # Generate coverage report
    print("\n" + "="*70)
    print("📊 Generating Coverage Report")
    print("="*70)
    
    coverage_cmd = [
        "pytest", "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term",
        "-m", "not gpu"
    ]
    
    coverage_success, coverage_duration = run_command(
        coverage_cmd,
        "Full Test Suite with Coverage"
    )
    total_duration += coverage_duration
    
    # Print summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"\nTest Suites: {passed}/{total} passed")
    print(f"Total Duration: {total_duration:.2f}s")
    print("\nDetails:")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {result['description']}: {result['duration']:.2f}s")
    
    if coverage_success:
        print(f"\n  ✅ Coverage Report: {coverage_duration:.2f}s")
        print(f"     📄 HTML Report: htmlcov/index.html")
    
    # Final status
    print("\n" + "="*70)
    
    all_passed = all(r["success"] for r in results) and coverage_success
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\n✨ Next steps:")
        print("  1. View coverage: open htmlcov/index.html")
        print("  2. Run specific tests: pytest tests/test_mamba.py -v")
        print("  3. Run with GPU: pytest tests/ -v")
        print("\n")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("="*70)
        print("\n🔧 Troubleshooting:")
        print("  1. Check error messages above")
        print("  2. Run failed test individually: pytest tests/test_<name>.py -v")
        print("  3. Run with more details: pytest tests/test_<name>.py -v -s")
        print("\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
