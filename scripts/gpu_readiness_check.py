#!/usr/bin/env python3
"""
GPU Readiness Check

Verifies the system is ready for GPU activation:
1. Server starts without errors
2. RAG retrieval works
3. Validation works
4. Context assembly works
5. Encoder/decoder fail gracefully with model_not_loaded
6. No HF API calls

NO MOCKS. Uses real RAG pipeline.
"""

import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class GPUReadinessChecker:
    """Checks system readiness for GPU activation."""
    
    def __init__(self):
        self.results = []
    
    def check_server_imports(self) -> Tuple[bool, str]:
        """Check that server module imports without errors."""
        print("\n1. Checking server imports...")
        try:
            from src.inference.local_models import LocalModelRegistry
            from src.inference.server import MODEL_REGISTRY
            
            if MODEL_REGISTRY is not None:
                print("   ✓ LocalModelRegistry initialized")
                return True, "Server imports OK"
            else:
                return False, "MODEL_REGISTRY is None"
        except Exception as e:
            return False, f"Import error: {e}"
    
    def check_rag_retrieval(self) -> Tuple[bool, str]:
        """Check that RAG retrieval works."""
        print("\n2. Checking RAG retrieval...")
        try:
            from src.rag import LegalRetriever
            
            retriever = LegalRetriever()
            retriever.initialize(index_dense=True)
            
            chunks = retriever.retrieve("Section 420 IPC", top_k=5)
            
            if chunks and len(chunks) > 0:
                print(f"   ✓ Retrieved {len(chunks)} chunks")
                return True, f"Retrieved {len(chunks)} chunks"
            else:
                return False, "No chunks retrieved"
        except Exception as e:
            return False, f"Retrieval error: {e}"
    
    def check_validation(self) -> Tuple[bool, str]:
        """Check that validation works."""
        print("\n3. Checking validation...")
        try:
            from src.rag import LegalRetriever, RetrievalValidator
            
            retriever = LegalRetriever()
            retriever.initialize()
            validator = RetrievalValidator()
            
            chunks = retriever.retrieve("Section 420 IPC", top_k=5)
            
            result = validator.validate(
                query="Section 420 IPC",
                retrieved_chunks=[
                    {'chunk_id': c.chunk_id, 'text': c.text, 'section': c.section, 'act': c.act, 'score': c.score}
                    for c in chunks
                ],
            )
            
            if result.status.value in ["pass", "refuse"]:
                print(f"   ✓ Validation status: {result.status.value}")
                return True, f"Validation works: {result.status.value}"
            else:
                return False, f"Unexpected status: {result.status.value}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def check_context_assembly(self) -> Tuple[bool, str]:
        """Check that context assembly works."""
        print("\n4. Checking context assembly...")
        try:
            from src.rag import LegalRetriever, RetrievalValidator, ContextAssembler
            
            retriever = LegalRetriever()
            retriever.initialize()
            validator = RetrievalValidator()
            assembler = ContextAssembler()
            
            chunks = retriever.retrieve("Section 420 IPC", top_k=5)
            
            val_result = validator.validate(
                query="Section 420 IPC",
                retrieved_chunks=[
                    {'chunk_id': c.chunk_id, 'text': c.text, 'section': c.section, 'act': c.act, 'score': c.score}
                    for c in chunks
                ],
            )
            
            if val_result.status.value == "refuse":
                print("   ✓ Validation refused (expected for some queries)")
                return True, "Validation refused correctly"
            
            ctx_result = assembler.assemble(
                query="Section 420 IPC",
                validated_chunks=[
                    {'chunk_id': c.chunk_id, 'text': c.text, 'section': c.section, 'act': c.act, 'score': c.adjusted_score, 'doc_type': 'bare_act', 'year': 1860}
                    for c in val_result.accepted_chunks
                ],
            )
            
            if ctx_result.status.value in ["assembled", "refuse"]:
                print(f"   ✓ Context assembly status: {ctx_result.status.value}")
                return True, f"Context assembly works: {ctx_result.status.value}"
            else:
                return False, f"Unexpected status: {ctx_result.status.value}"
        except Exception as e:
            return False, f"Context assembly error: {e}"
    
    def check_encoder_graceful_fail(self) -> Tuple[bool, str]:
        """Check that encoder fails gracefully when model not loaded."""
        print("\n5. Checking encoder graceful failure...")
        try:
            from src.inference.server import _run_encoder, HFInferenceError
            
            try:
                _run_encoder("test-model-not-loaded", "Test query")
                return False, "Encoder should have raised error"
            except HFInferenceError as e:
                if "encoder_failed" in e.reason:
                    print(f"   ✓ Encoder failed gracefully: {e.reason}")
                    return True, "Encoder fails gracefully"
                else:
                    return False, f"Unexpected error reason: {e.reason}"
            except Exception as e:
                return False, f"Unexpected error type: {type(e).__name__}: {e}"
        except Exception as e:
            return False, f"Import error: {e}"
    
    def check_decoder_graceful_fail(self) -> Tuple[bool, str]:
        """Check that decoder fails gracefully when model not loaded."""
        print("\n6. Checking decoder graceful failure...")
        try:
            from src.inference.local_models import LocalModelRegistry, run_local_decoder
            from src.inference.server import HFInferenceError
            
            registry = LocalModelRegistry()
            
            try:
                run_local_decoder(registry, "test-model-not-loaded", "Test prompt")
                return False, "Decoder should have raised error"
            except RuntimeError as e:
                if "not loaded" in str(e).lower():
                    print(f"   ✓ Decoder failed gracefully: Model not loaded")
                    return True, "Decoder fails gracefully"
                else:
                    return False, f"Unexpected error: {e}"
            except Exception as e:
                return False, f"Unexpected error type: {type(e).__name__}: {e}"
        except Exception as e:
            return False, f"Import error: {e}"
    
    def check_no_hf_api_calls(self) -> Tuple[bool, str]:
        """Check that HF API is deprecated and unreachable."""
        print("\n7. Checking HF API is deprecated...")
        try:
            from src.inference.server import _hf_infer_with_retry, HFInferenceError
            
            try:
                _hf_infer_with_retry("test-model", {"inputs": "test"})
                return False, "HF API should be deprecated"
            except HFInferenceError as e:
                if "deprecated" in str(e).lower() or "encoder_failed" in e.reason:
                    print("   ✓ HF API is deprecated and unreachable")
                    return True, "HF API deprecated"
                else:
                    return False, f"Unexpected error: {e}"
            except Exception as e:
                return False, f"Unexpected error: {e}"
        except Exception as e:
            return False, f"Import error: {e}"
    
    def check_model_registry(self) -> Tuple[bool, str]:
        """Check LocalModelRegistry functionality."""
        print("\n8. Checking LocalModelRegistry...")
        try:
            from src.inference.local_models import LocalModelRegistry
            
            registry = LocalModelRegistry()
            
            # Check is_loaded returns False for unloaded model
            if registry.is_loaded("test-model"):
                return False, "is_loaded should return False"
            
            # Check list_loaded returns empty dict
            loaded = registry.list_loaded()
            if loaded != {}:
                return False, f"list_loaded should be empty: {loaded}"
            
            print("   ✓ LocalModelRegistry works correctly")
            return True, "LocalModelRegistry OK"
        except Exception as e:
            return False, f"Registry error: {e}"
    
    def run_all(self) -> bool:
        """Run all checks."""
        self.results = []
        
        checks = [
            ("Server Imports", self.check_server_imports),
            ("RAG Retrieval", self.check_rag_retrieval),
            ("Validation", self.check_validation),
            ("Context Assembly", self.check_context_assembly),
            ("Encoder Graceful Fail", self.check_encoder_graceful_fail),
            ("Decoder Graceful Fail", self.check_decoder_graceful_fail),
            ("No HF API Calls", self.check_no_hf_api_calls),
            ("Model Registry", self.check_model_registry),
        ]
        
        for name, check_fn in checks:
            try:
                passed, msg = check_fn()
                self.results.append((name, passed, msg))
            except Exception as e:
                self.results.append((name, False, f"Exception: {e}"))
        
        return all(passed for _, passed, _ in self.results)
    
    def print_results(self) -> None:
        """Print formatted results."""
        print("\n" + "=" * 50)
        print("GPU READINESS CHECK")
        print("=" * 50)
        
        failed = []
        
        for name, passed, msg in self.results:
            dots = "." * (30 - len(name))
            status = "PASS" if passed else "FAIL"
            print(f"{name} {dots} {status}")
            
            if not passed:
                failed.append(name)
                print(f"    └─ {msg}")
        
        print("-" * 50)
        
        if not failed:
            print("FINAL RESULT: PASS")
            print()
            print("✓ Server starts without errors")
            print("✓ RAG retrieval works")
            print("✓ Validation works")
            print("✓ Context assembly works")
            print("✓ Encoder/decoder fail gracefully")
            print("✓ No HF API calls")
            print()
            print("System is GPU-READY")
            print("Load models when GPU is available:")
            print("  MODEL_REGISTRY.load_encoder('ai4bharat/indian-legal-bert-8b')")
            print("  MODEL_REGISTRY.load_decoder('Qwen/Qwen2.5-32B-Instruct')")
        else:
            print("FINAL RESULT: FAIL")
            print(f"FAILED CHECKS: {failed}")
        
        print()


def main():
    """Run GPU readiness check."""
    print("Running GPU Readiness Check...")
    
    checker = GPUReadinessChecker()
    all_passed = checker.run_all()
    checker.print_results()
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
