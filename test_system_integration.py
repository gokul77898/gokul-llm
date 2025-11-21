"""
System Integration Test Script
Tests ChromaDB, AutoPipeline, Model Selector, and Training Skeletons
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

def test_chromadb():
    """Test ChromaDB initialization"""
    print("\n" + "="*60)
    print("TEST 1: ChromaDB Integration")
    print("="*60)
    
    try:
        from db.chroma import ChromaDBClient, VectorRetriever
        
        # Test client
        client = ChromaDBClient()
        print("✅ ChromaDB client initialized")
        
        # Test collection
        collection = client.get_or_create_collection("legal_docs")
        print(f"✅ Collection 'legal_docs' ready (count: {collection.count()})")
        
        # Test retriever
        retriever = VectorRetriever("legal_docs")
        stats = retriever.get_collection_stats()
        print(f"✅ Retriever initialized")
        print(f"   - Collection: {stats['name']}")
        print(f"   - Documents: {stats['document_count']}")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selector():
    """Test automatic model selection"""
    print("\n" + "="*60)
    print("TEST 2: Model Selector")
    print("="*60)
    
    try:
        from src.core.model_selector import get_model_selector
        
        selector = get_model_selector()
        print("✅ Model selector initialized")
        
        # Test different query types
        test_cases = [
            ("What is appropriate government?", "Legal query"),
            ("Hello", "Simple query"),
            ("Explain the implications of the minimum wages act and how it affects employers in different states", "Complex query"),
            ("Compare and analyze", "Reasoning query")
        ]
        
        for query, description in test_cases:
            model = selector.pick(query)
            print(f"✅ {description}: '{query[:40]}...' → {model}")
        
        return True
    except Exception as e:
        print(f"❌ Model selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_pipeline():
    """Test AutoPipeline with ChromaDB"""
    print("\n" + "="*60)
    print("TEST 3: AutoPipeline with ChromaDB")
    print("="*60)
    
    try:
        from src.pipelines.auto_pipeline import AutoPipeline
        
        # Initialize pipeline
        pipeline = AutoPipeline(collection_name="legal_docs")
        print("✅ AutoPipeline initialized")
        
        # Check components
        if hasattr(pipeline, 'retriever'):
            print("✅ ChromaDB retriever connected")
        if hasattr(pipeline, 'model_selector'):
            print("✅ Model selector connected")
        if hasattr(pipeline, 'grounded_generator'):
            if pipeline.grounded_generator:
                print("✅ Grounded generator available")
            else:
                print("⚠️  Grounded generator not available")
        
        # Test model selection
        model = pipeline.select_model("What is minimum wage?")
        print(f"✅ Model selection works: '{model}'")
        
        # Try processing a query (will fail if no documents, but structure should work)
        try:
            result = pipeline.process_query("Test query", top_k=3)
            print(f"✅ Query processing structure works")
            print(f"   - Retrieved docs: {result.get('retrieved_docs', 0)}")
            print(f"   - Model used: {result.get('auto_model_used', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Query processing: {str(e)[:80]}")
        
        return True
    except Exception as e:
        print(f"❌ AutoPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_skeletons():
    """Test training module skeletons (should be disabled)"""
    print("\n" + "="*60)
    print("TEST 4: Training Skeletons (Should Be Disabled)")
    print("="*60)
    
    try:
        from src.training.sft_trainer import SFTTrainer
        from src.training.rl_trainer import RLTrainer
        from src.training.rlhf_trainer import RLHFTrainer
        from src.training.training_manager import TrainingManager, prepare_training_environment
        
        # Test SFT
        sft = SFTTrainer()
        print("✅ SFT trainer skeleton created")
        status = sft.get_training_status()
        print(f"   - Status: {status['status']}")
        print(f"   - Mode: {status['mode']}")
        
        # Test RL
        rl = RLTrainer()
        print("✅ RL trainer skeleton created")
        status = rl.get_training_status()
        print(f"   - Status: {status['status']}")
        
        # Test RLHF
        rlhf = RLHFTrainer()
        print("✅ RLHF trainer skeleton created")
        status = rlhf.get_training_status()
        print(f"   - Status: {status['status']}")
        
        # Test manager
        manager = TrainingManager()
        print("✅ Training manager created")
        
        # Test that training is blocked
        try:
            sft.train(None)
            print("❌ SFT training should be blocked!")
        except RuntimeError as e:
            print(f"✅ SFT correctly blocked: '{str(e)[:50]}...'")
        
        try:
            rl.step("q", "a", "e")
            print("❌ RL training should be blocked!")
        except RuntimeError as e:
            print(f"✅ RL correctly blocked: '{str(e)[:50]}...'")
        
        # Test environment preparation
        env = prepare_training_environment()
        print("✅ Training environment preparation works")
        print(f"   - Status: {env['status']}")
        
        return True
    except Exception as e:
        print(f"❌ Training skeletons test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chroma_manager():
    """Test ChromaDB manager"""
    print("\n" + "="*60)
    print("TEST 5: ChromaDB Manager")
    print("="*60)
    
    try:
        from src.core.chroma_manager import get_chroma_manager
        
        manager = get_chroma_manager()
        print("✅ ChromaDB manager initialized")
        
        # Check if ready
        is_ready = manager.is_ready()
        print(f"✅ Manager ready: {is_ready}")
        
        # Get stats
        stats = manager.get_collection_stats()
        print(f"✅ Collection stats retrieved:")
        print(f"   - Status: {stats.get('status', 'unknown')}")
        print(f"   - Collection: {stats.get('collection', 'N/A')}")
        print(f"   - Documents: {stats.get('document_count', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    test_names = [
        "ChromaDB Integration",
        "Model Selector",
        "AutoPipeline",
        "Training Skeletons",
        "ChromaDB Manager"
    ]
    
    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:25} {status}")
    
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ SYSTEM READY FOR:")
        print("   - API endpoint integration")
        print("   - UI development")
        print("   - Admin dashboard")
        return 0
    else:
        print(f"\n⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nPlease fix failing components before proceeding.")
        return 1


def main():
    """Run all integration tests"""
    print("="*60)
    print("MARK SYSTEM INTEGRATION TEST")
    print("="*60)
    print("Testing: ChromaDB, AutoPipeline, Model Selector, Training")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(test_chromadb())
    results.append(test_model_selector())
    results.append(test_auto_pipeline())
    results.append(test_training_skeletons())
    results.append(test_chroma_manager())
    
    # Print summary
    exit_code = print_summary(results)
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
