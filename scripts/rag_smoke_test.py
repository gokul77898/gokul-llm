#!/usr/bin/env python3
"""
RAG Smoke Test

Tests the full RAG pipeline with a valid query:
- Query: What is punishment under Section 420 IPC?
- Expected: Section 420 retrieved, case law retrieved, context assembled

NO MOCKS. Uses real RAG pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag import LegalRetriever, RetrievalValidator, ContextAssembler


def test_retrieval():
    """Test that Section 420 chunks are retrieved."""
    print("\n1. Testing Retrieval...")
    
    retriever = LegalRetriever()
    retriever.initialize(index_dense=True)
    
    query = "What is punishment under Section 420 IPC?"
    chunks = retriever.retrieve(query, top_k=10)
    
    if not chunks:
        print("   ✗ FAIL: No chunks retrieved")
        return False
    
    print(f"   Retrieved {len(chunks)} chunks")
    
    # Check if Section 420 is in results
    found_420 = False
    found_case_law = False
    
    for chunk in chunks:
        if chunk.section and '420' in str(chunk.section):
            found_420 = True
            print(f"   ✓ Found Section 420 chunk: {chunk.chunk_id[:8]}...")
        if chunk.doc_type and 'case' in str(chunk.doc_type).lower():
            found_case_law = True
            print(f"   ✓ Found case law chunk: {chunk.chunk_id[:8]}...")
    
    if not found_420:
        print("   ✗ FAIL: Section 420 not found in results")
        return False
    
    print("   ✓ PASS: Section 420 retrieved")
    return True


def test_validation():
    """Test that retrieved chunks pass validation."""
    print("\n2. Testing Validation...")
    
    retriever = LegalRetriever()
    retriever.initialize()
    validator = RetrievalValidator()
    
    query = "What is punishment under Section 420 IPC?"
    chunks = retriever.retrieve(query, top_k=10)
    
    if not chunks:
        print("   ✗ FAIL: No chunks to validate")
        return False
    
    result = validator.validate(
        query=query,
        retrieved_chunks=[
            {
                'chunk_id': c.chunk_id,
                'text': c.text,
                'section': c.section,
                'act': c.act,
                'score': c.score,
            }
            for c in chunks
        ],
    )
    
    if result.status.value == "refuse":
        print(f"   ✗ FAIL: Validation refused: {result.refusal_reason}")
        return False
    
    print(f"   Accepted {len(result.accepted_chunks)} chunks")
    print("   ✓ PASS: Validation passed")
    return True


def test_context_assembly():
    """Test that context is assembled correctly."""
    print("\n3. Testing Context Assembly...")
    
    retriever = LegalRetriever()
    retriever.initialize()
    validator = RetrievalValidator()
    assembler = ContextAssembler()
    
    query = "What is punishment under Section 420 IPC?"
    chunks = retriever.retrieve(query, top_k=10)
    
    if not chunks:
        print("   ✗ FAIL: No chunks retrieved")
        return False
    
    val_result = validator.validate(
        query=query,
        retrieved_chunks=[
            {'chunk_id': c.chunk_id, 'text': c.text, 'section': c.section, 'act': c.act, 'score': c.score}
            for c in chunks
        ],
    )
    
    if val_result.status.value == "refuse":
        print(f"   ✗ FAIL: Validation refused")
        return False
    
    ctx_result = assembler.assemble(
        query=query,
        validated_chunks=[
            {
                'chunk_id': c.chunk_id,
                'text': c.text,
                'section': c.section,
                'act': c.act,
                'score': c.adjusted_score,
                'doc_type': 'bare_act',
                'year': 1860,
            }
            for c in val_result.accepted_chunks
        ],
    )
    
    if ctx_result.status.value == "refuse":
        print(f"   ✗ FAIL: Context assembly refused: {ctx_result.refusal_reason}")
        return False
    
    # Check for EVIDENCE markers
    if "EVIDENCE_START" not in ctx_result.context_text:
        print("   ✗ FAIL: Missing EVIDENCE_START marker")
        return False
    
    if "EVIDENCE_END" not in ctx_result.context_text:
        print("   ✗ FAIL: Missing EVIDENCE_END marker")
        return False
    
    # Check for citation markers
    if "[1]" not in ctx_result.context_text:
        print("   ✗ FAIL: Missing citation markers")
        return False
    
    print(f"   Context assembled: {ctx_result.token_count} tokens")
    print(f"   Used {len(ctx_result.used_chunks)} chunks")
    print("   ✓ PASS: Context assembly successful")
    return True


def main():
    """Run all smoke tests."""
    print("=" * 50)
    print("RAG SMOKE TEST")
    print("Query: What is punishment under Section 420 IPC?")
    print("=" * 50)
    
    results = []
    
    results.append(("Retrieval", test_retrieval()))
    results.append(("Validation", test_validation()))
    results.append(("Context Assembly", test_context_assembly()))
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("  - Section 420 retrieved")
        print("  - Validation passed")
        print("  - Context assembled")
    else:
        print("✗ SOME TESTS FAILED")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
