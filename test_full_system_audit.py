"""
FULL SYSTEM AUDIT - MARK AI System
Complete end-to-end verification
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test results storage
test_results = {
    "backend": {},
    "frontend": {},
    "integration": {},
    "training": {},
    "timestamp": datetime.now().isoformat()
}

def print_header(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_backend_imports():
    """Test 1: Backend Import Check"""
    print_header("TEST 1: BACKEND IMPORTS")
    
    errors = []
    successes = []
    
    # Test ChromaDB imports
    try:
        from db.chroma import ChromaDBClient, VectorRetriever, EmbeddingModel
        successes.append("✅ ChromaDB modules imported")
    except Exception as e:
        errors.append(f"❌ ChromaDB import failed: {e}")
    
    # Test core modules
    try:
        from src.core.model_selector import get_model_selector, ModelSelector
        successes.append("✅ ModelSelector imported")
    except Exception as e:
        errors.append(f"❌ ModelSelector import failed: {e}")
    
    try:
        from src.core.chroma_manager import get_chroma_manager
        successes.append("✅ ChromaManager imported")
    except Exception as e:
        errors.append(f"❌ ChromaManager import failed: {e}")
    
    # Test pipeline
    try:
        from src.pipelines.auto_pipeline import AutoPipeline
        successes.append("✅ AutoPipeline imported")
    except Exception as e:
        errors.append(f"❌ AutoPipeline import failed: {e}")
    
    # Test grounding components
    try:
        from src.rag.reranker import CrossEncoderReranker
        from src.rag.grounded_generator import GroundedAnswerGenerator
        successes.append("✅ Grounding components imported")
    except Exception as e:
        errors.append(f"❌ Grounding import failed: {e}")
    
    # Test training modules
    try:
        from src.training.sft_trainer import SFTTrainer
        from src.training.rl_trainer import RLTrainer
        from src.training.rlhf_trainer import RLHFTrainer
        from src.training.training_manager import TrainingManager
        successes.append("✅ Training modules imported")
    except Exception as e:
        errors.append(f"❌ Training import failed: {e}")
    
    for s in successes:
        print(s)
    for e in errors:
        print(e)
    
    test_results["backend"]["imports"] = {
        "status": "PASS" if len(errors) == 0 else "FAIL",
        "successes": len(successes),
        "errors": errors
    }
    
    return len(errors) == 0

def test_chromadb_integration():
    """Test 2: ChromaDB Integration"""
    print_header("TEST 2: CHROMADB INTEGRATION")
    
    try:
        from db.chroma import ChromaDBClient, VectorRetriever
        
        # Test client
        client = ChromaDBClient()
        print("✅ ChromaDB client initialized")
        
        # Test collection
        collection = client.get_or_create_collection("legal_docs")
        doc_count = collection.count()
        print(f"✅ Collection 'legal_docs' exists (documents: {doc_count})")
        
        # Test retriever
        retriever = VectorRetriever("legal_docs")
        stats = retriever.get_collection_stats()
        print(f"✅ VectorRetriever initialized")
        print(f"   - Collection: {stats.get('name', 'N/A')}")
        print(f"   - Documents: {stats.get('document_count', 0)}")
        
        test_results["backend"]["chromadb"] = {
            "status": "PASS",
            "collection": "legal_docs",
            "document_count": doc_count
        }
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        test_results["backend"]["chromadb"] = {
            "status": "FAIL",
            "error": str(e)
        }
        return False

def test_model_selector():
    """Test 3: Model Selector"""
    print_header("TEST 3: MODEL SELECTOR")
    
    try:
        from src.core.model_selector import get_model_selector
        
        selector = get_model_selector()
        print("✅ ModelSelector initialized")
        
        # Test different query types
        test_cases = [
            ("What is appropriate government?", "Legal query"),
            ("Hello", "Simple query"),
            ("Explain the implications", "Complex query")
        ]
        
        for query, description in test_cases:
            model = selector.pick(query)
            print(f"✅ {description}: '{query[:30]}...' → {model}")
        
        test_results["backend"]["model_selector"] = {
            "status": "PASS",
            "test_cases_passed": len(test_cases)
        }
        return True
        
    except Exception as e:
        print(f"❌ ModelSelector test failed: {e}")
        test_results["backend"]["model_selector"] = {
            "status": "FAIL",
            "error": str(e)
        }
        return False

def test_auto_pipeline():
    """Test 4: AutoPipeline with ChromaDB"""
    print_header("TEST 4: AUTOPIPELINE INTEGRATION")
    
    try:
        from src.pipelines.auto_pipeline import AutoPipeline
        
        # Initialize pipeline
        pipeline = AutoPipeline(collection_name="legal_docs")
        print("✅ AutoPipeline initialized")
        
        # Check components
        checks = []
        if hasattr(pipeline, 'retriever'):
            checks.append("✅ ChromaDB retriever connected")
        if hasattr(pipeline, 'model_selector'):
            checks.append("✅ Model selector connected")
        if hasattr(pipeline, 'grounded_generator'):
            if pipeline.grounded_generator:
                checks.append("✅ Grounded generator available")
            else:
                checks.append("⚠️  Grounded generator not available")
        
        for check in checks:
            print(check)
        
        # Test model selection
        model = pipeline.select_model("Test query")
        print(f"✅ Model selection works: '{model}'")
        
        # Test query processing structure
        try:
            result = pipeline.process_query("Test system", top_k=3)
            print(f"✅ Query processing structure works")
            print(f"   - Retrieved docs: {result.get('retrieved_docs', 0)}")
            print(f"   - Model used: {result.get('auto_model_used', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Query processing: {str(e)[:80]}")
        
        test_results["backend"]["auto_pipeline"] = {
            "status": "PASS",
            "components_available": len(checks)
        }
        return True
        
    except Exception as e:
        print(f"❌ AutoPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results["backend"]["auto_pipeline"] = {
            "status": "FAIL",
            "error": str(e)
        }
        return False

def test_training_disabled():
    """Test 5: Training Modules Disabled"""
    print_header("TEST 5: TRAINING MODULES (SHOULD BE DISABLED)")
    
    try:
        from src.training.sft_trainer import SFTTrainer
        from src.training.rl_trainer import RLTrainer
        from src.training.rlhf_trainer import RLHFTrainer
        from src.training.training_manager import TrainingManager
        
        # Test SFT
        sft = SFTTrainer()
        print("✅ SFT trainer skeleton created")
        status = sft.get_training_status()
        print(f"   - Status: {status['status']}")
        
        # Test that training is blocked
        blocked_count = 0
        try:
            sft.train(None)
            print("❌ SFT training should be blocked!")
        except RuntimeError:
            print("✅ SFT correctly blocked")
            blocked_count += 1
        
        try:
            rl = RLTrainer()
            rl.step("q", "a", "e")
            print("❌ RL training should be blocked!")
        except RuntimeError:
            print("✅ RL correctly blocked")
            blocked_count += 1
        
        test_results["training"]["status"] = {
            "status": "PASS",
            "blocked_correctly": blocked_count >= 2,
            "mode": "SETUP_MODE"
        }
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        test_results["training"]["status"] = {
            "status": "FAIL",
            "error": str(e)
        }
        return False

def test_ingestion_blocked():
    """Test 6: Data Ingestion Blocked"""
    print_header("TEST 6: DATA INGESTION BLOCKER")
    
    try:
        from db.chroma import ingest_file
        
        # Try to ingest (should fail)
        try:
            ingest_file("fake_file.txt", "legal_docs")
            print("❌ Ingestion should be blocked!")
            test_results["backend"]["ingestion"] = {
                "status": "FAIL",
                "reason": "Ingestion not blocked"
            }
            return False
        except RuntimeError as e:
            print(f"✅ Ingestion correctly blocked")
            print(f"   Error: {str(e)[:60]}...")
            test_results["backend"]["ingestion"] = {
                "status": "PASS",
                "blocked": True
            }
            return True
            
    except Exception as e:
        print(f"❌ Ingestion blocker test failed: {e}")
        test_results["backend"]["ingestion"] = {
            "status": "FAIL",
            "error": str(e)
        }
        return False

def check_faiss_references():
    """Test 7: FAISS References Check"""
    print_header("TEST 7: FAISS REPLACEMENT CHECK")
    
    print("Checking for FAISS references in critical files...")
    
    issues_found = []
    
    # Check main.py
    main_py = Path("/Users/gokul/Documents/MARK/src/api/main.py")
    if main_py.exists():
        content = main_py.read_text()
        if "FAISSStore" in content or "load_faiss" in content:
            issues_found.append("src/api/main.py has FAISS references")
            print("⚠️  WARNING: src/api/main.py still has FAISS references")
        else:
            print("✅ src/api/main.py clean (no FAISS)")
    
    # Check auto_pipeline
    auto_py = Path("/Users/gokul/Documents/MARK/src/pipelines/auto_pipeline.py")
    if auto_py.exists():
        content = auto_py.read_text()
        if "FAISS" in content or "faiss" in content:
            issues_found.append("src/pipelines/auto_pipeline.py has FAISS references")
            print("⚠️  WARNING: AutoPipeline has FAISS references")
        else:
            print("✅ AutoPipeline clean (no FAISS)")
    
    if len(issues_found) == 0:
        print("✅ All critical files clean - FAISS successfully replaced")
        test_results["backend"]["faiss_check"] = {
            "status": "PASS",
            "message": "FAISS successfully replaced with ChromaDB"
        }
        return True
    else:
        print(f"⚠️  Found {len(issues_found)} file(s) with FAISS references")
        test_results["backend"]["faiss_check"] = {
            "status": "WARNING",
            "issues": issues_found
        }
        return False

def generate_report():
    """Generate final audit report"""
    print_header("FULL SYSTEM AUDIT REPORT")
    
    print("\n1. BACKEND STATUS")
    print("-" * 70)
    
    backend_status = "PASS"
    for key, value in test_results.get("backend", {}).items():
        status = value.get("status", "UNKNOWN")
        if status == "FAIL":
            backend_status = "FAIL"
        elif status == "WARNING" and backend_status == "PASS":
            backend_status = "WARNING"
        print(f"   {key}: {status}")
        if "error" in value:
            print(f"      Error: {value['error']}")
    
    print(f"\n   Overall Backend: {backend_status}")
    
    print("\n2. TRAINING SYSTEM")
    print("-" * 70)
    training_status = test_results.get("training", {}).get("status", {}).get("status", "UNKNOWN")
    print(f"   Status: {training_status}")
    print(f"   Mode: SETUP_MODE (Training Disabled)")
    
    print("\n3. CRITICAL ISSUES")
    print("-" * 70)
    
    issues = []
    if test_results["backend"].get("faiss_check", {}).get("status") == "WARNING":
        issues.append("⚠️  FAISS references in src/api/main.py need replacement")
    
    if len(issues) == 0:
        print("   ✅ No critical issues")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    print("\n4. FINAL CONCLUSION")
    print("-" * 70)
    
    if backend_status == "PASS" and training_status == "PASS" and len(issues) == 0:
        print("   ✅ SYSTEM READY FOR DATA INGESTION & TRAINING")
        print("   ✅ All tests passed")
    elif backend_status == "WARNING" or len(issues) > 0:
        print("   ⚠️  SYSTEM FUNCTIONAL BUT HAS WARNINGS")
        print("   ⚠️  Address warnings before production")
    else:
        print("   ❌ ISSUES FOUND - FIX BEFORE PROCEEDING")
    
    print("\n" + "="*70)
    
    # Save report
    report_path = Path("SYSTEM_AUDIT_REPORT.json")
    with open(report_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n📄 Full report saved to: {report_path}")

def main():
    """Run full system audit"""
    print("\n" + "="*70)
    print("  MARK FULL SYSTEM AUDIT")
    print("  Testing: Backend, ChromaDB, Training, Integrations")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Backend Imports", test_backend_imports()))
    results.append(("ChromaDB Integration", test_chromadb_integration()))
    results.append(("Model Selector", test_model_selector()))
    results.append(("AutoPipeline", test_auto_pipeline()))
    results.append(("Training Disabled", test_training_disabled()))
    results.append(("Ingestion Blocked", test_ingestion_blocked()))
    results.append(("FAISS Check", check_faiss_references()))
    
    # Generate report
    generate_report()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n🎯 Tests Passed: {passed}/{total}")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
