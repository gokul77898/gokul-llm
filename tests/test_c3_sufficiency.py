#!/usr/bin/env python3
"""
Tests for Phase-3 Evidence Sufficiency Checking

Tests the is_evidence_sufficient() function with various query types
and evidence scenarios.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.c3_synthesize import (
    RetrievedChunk,
    is_evidence_sufficient,
    classify_query_type,
)


# Test fixtures
def create_definition_chunk():
    """Create a chunk with definitional language."""
    return RetrievedChunk(
        chunk_id="abc123",
        semantic_id="MinimumWagesAct_2_1",
        text="Section 2. Definitions. In this Act, employer means any person who employs, whether directly or through another person, one or more employees in any scheduled employment.",
        score=0.95,
        act="Minimum Wages Act",
        section="2"
    )


def create_punishment_chunk():
    """Create a chunk with punishment language."""
    return RetrievedChunk(
        chunk_id="def456",
        semantic_id="IPC_420_0",
        text="Section 420 IPC – Cheating and dishonestly inducing delivery of property. Whoever cheats shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
        score=0.95,
        act="IPC",
        section="420"
    )


def create_general_chunk():
    """Create a general informational chunk."""
    return RetrievedChunk(
        chunk_id="ghi789",
        semantic_id="MinimumWagesAct_1_0",
        text="Section 1. Short title, extent and commencement. This Act may be called the Minimum Wages Act, 1948. It extends to the whole of India.",
        score=0.85,
        act="Minimum Wages Act",
        section="1"
    )


def create_irrelevant_chunk():
    """Create an irrelevant chunk."""
    return RetrievedChunk(
        chunk_id="jkl012",
        semantic_id="SupremeCourt_1_2020_0",
        text="The appellant filed a petition challenging the order. The court observed that procedural requirements must be met.",
        score=0.50,
        act=None,
        section="1"
    )


# ============================================================================
# TEST: classify_query_type
# ============================================================================

def test_classify_query_type_definition():
    """Test classification of definition queries."""
    queries = [
        "What is the definition of employer?",
        "Define employer",
        "What does employer mean?",
        "What is an employer?"
    ]
    
    for query in queries:
        query_type = classify_query_type(query)
        assert query_type == 'definition', f"Failed for: {query}"
    
    print("✓ test_classify_query_type_definition PASSED")


def test_classify_query_type_punishment():
    """Test classification of punishment queries."""
    queries = [
        "What is the punishment for cheating?",
        "What is the penalty under Section 420?",
        "What is the sentence for fraud?",
        "What is the imprisonment for cheating?"
    ]
    
    for query in queries:
        query_type = classify_query_type(query)
        assert query_type == 'punishment', f"Failed for: {query}"
    
    print("✓ test_classify_query_type_punishment PASSED")


def test_classify_query_type_procedure():
    """Test classification of procedure queries."""
    queries = [
        "How to file a complaint?",
        "What is the procedure for appeal?",
        "What are the steps to register?"
    ]
    
    for query in queries:
        query_type = classify_query_type(query)
        assert query_type == 'procedure', f"Failed for: {query}"
    
    print("✓ test_classify_query_type_procedure PASSED")


def test_classify_query_type_scope():
    """Test classification of scope queries."""
    queries = [
        "What is the extent of the Act?",
        "What is the applicability of this law?",
        "Does this apply to all states?"
    ]
    
    for query in queries:
        query_type = classify_query_type(query)
        assert query_type == 'scope', f"Failed for: {query}"
    
    print("✓ test_classify_query_type_scope PASSED")


# ============================================================================
# TEST: is_evidence_sufficient - POSITIVE CASES
# ============================================================================

def test_evidence_sufficient_definition_query():
    """Test sufficiency for definition query with definitional chunk."""
    query = "What is the definition of employer?"
    chunks = [create_definition_chunk()]
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is True
    assert sufficiency.query_type == 'definition'
    assert "definitional language" in sufficiency.reason.lower()
    assert len(sufficiency.relevant_chunks) > 0
    
    print("✓ test_evidence_sufficient_definition_query PASSED")


def test_evidence_sufficient_punishment_query():
    """Test sufficiency for punishment query with punishment chunk."""
    query = "What is the punishment for cheating under Section 420?"
    chunks = [create_punishment_chunk()]
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is True
    assert sufficiency.query_type == 'punishment'
    assert "punishment" in sufficiency.reason.lower() or "penalty" in sufficiency.reason.lower()
    assert len(sufficiency.relevant_chunks) > 0
    
    print("✓ test_evidence_sufficient_punishment_query PASSED")


def test_evidence_sufficient_general_query():
    """Test sufficiency for general query with relevant chunk."""
    query = "What is the extent of the Minimum Wages Act?"
    chunks = [create_general_chunk()]
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is True
    assert len(sufficiency.relevant_chunks) > 0
    
    print("✓ test_evidence_sufficient_general_query PASSED")


def test_evidence_sufficient_multiple_chunks():
    """Test sufficiency with multiple relevant chunks."""
    query = "What is the definition of employer?"
    chunks = [create_definition_chunk(), create_general_chunk()]
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is True
    assert len(sufficiency.relevant_chunks) >= 1
    
    print("✓ test_evidence_sufficient_multiple_chunks PASSED")


# ============================================================================
# TEST: is_evidence_sufficient - NEGATIVE CASES
# ============================================================================

def test_evidence_insufficient_no_chunks():
    """Test insufficiency when no chunks retrieved."""
    query = "What is the definition of employer?"
    chunks = []
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is False
    assert "no evidence" in sufficiency.reason.lower()
    assert len(sufficiency.relevant_chunks) == 0
    
    print("✓ test_evidence_insufficient_no_chunks PASSED")


def test_evidence_insufficient_definition_without_definitional_language():
    """Test insufficiency for definition query without definitional language."""
    query = "What is the definition of employer?"
    chunks = [create_general_chunk()]  # No definitional language
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is False
    assert "definitional language" in sufficiency.reason.lower()
    
    print("✓ test_evidence_insufficient_definition_without_definitional_language PASSED")


def test_evidence_insufficient_punishment_without_penalty_info():
    """Test insufficiency for punishment query without penalty information."""
    query = "What is the punishment for cheating?"
    chunks = [create_definition_chunk()]  # No punishment info
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is False
    assert "punishment" in sufficiency.reason.lower() or "penalty" in sufficiency.reason.lower()
    
    print("✓ test_evidence_insufficient_punishment_without_penalty_info PASSED")


def test_evidence_insufficient_irrelevant_chunks():
    """Test insufficiency when chunks are irrelevant."""
    query = "What is the definition of employer in the Minimum Wages Act?"
    chunks = [create_irrelevant_chunk()]
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is False
    assert len(sufficiency.relevant_chunks) == 0
    
    print("✓ test_evidence_insufficient_irrelevant_chunks PASSED")


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

def test_evidence_sufficient_mixed_relevant_irrelevant():
    """Test sufficiency with mix of relevant and irrelevant chunks."""
    query = "What is the definition of employer?"
    chunks = [create_definition_chunk(), create_irrelevant_chunk()]
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    # Should be sufficient because at least one chunk has definitional language
    assert sufficiency.is_sufficient is True
    
    print("✓ test_evidence_sufficient_mixed_relevant_irrelevant PASSED")


def test_evidence_sufficient_case_insensitive():
    """Test that sufficiency checking is case-insensitive."""
    query = "WHAT IS THE DEFINITION OF EMPLOYER?"
    chunks = [create_definition_chunk()]
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    assert sufficiency.is_sufficient is True
    
    print("✓ test_evidence_sufficient_case_insensitive PASSED")


def test_evidence_sufficient_partial_key_terms():
    """Test sufficiency with partial key term matches."""
    query = "What is the definition of employee?"  # Note: employee, not employer
    chunks = [create_definition_chunk()]  # Has "employer" and "employees"
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    # Should still be sufficient due to related terms
    assert sufficiency.is_sufficient is True
    
    print("✓ test_evidence_sufficient_partial_key_terms PASSED")


def test_evidence_insufficient_wrong_query_type():
    """Test insufficiency when chunk doesn't match query type requirements."""
    query = "What is the punishment for not paying minimum wages?"
    chunks = [create_definition_chunk()]  # Definition, not punishment
    
    sufficiency = is_evidence_sufficient(query, chunks)
    
    # Should be insufficient because punishment query needs punishment info
    assert sufficiency.is_sufficient is False
    
    print("✓ test_evidence_insufficient_wrong_query_type PASSED")


# ============================================================================
# TEST: QUERY TYPE SPECIFIC REQUIREMENTS
# ============================================================================

def test_definition_requires_definitional_language():
    """Test that definition queries require explicit definitional language."""
    query = "What is an employer?"
    
    # Chunk with definitional language
    chunk_with_def = RetrievedChunk(
        chunk_id="abc",
        semantic_id="Test_1_0",
        text="Employer means any person who employs workers.",
        score=0.9,
        act="Test Act",
        section="1"
    )
    
    # Chunk without definitional language
    chunk_without_def = RetrievedChunk(
        chunk_id="def",
        semantic_id="Test_2_0",
        text="The employer must pay minimum wages to workers.",
        score=0.9,
        act="Test Act",
        section="2"
    )
    
    # With definitional language - sufficient
    sufficiency1 = is_evidence_sufficient(query, [chunk_with_def])
    assert sufficiency1.is_sufficient is True
    
    # Without definitional language - insufficient
    sufficiency2 = is_evidence_sufficient(query, [chunk_without_def])
    assert sufficiency2.is_sufficient is False
    
    print("✓ test_definition_requires_definitional_language PASSED")


def test_punishment_requires_penalty_language():
    """Test that punishment queries require explicit penalty language."""
    query = "What is the punishment for Section 420?"
    
    # Chunk with penalty language
    chunk_with_penalty = RetrievedChunk(
        chunk_id="abc",
        semantic_id="IPC_420_0",
        text="Shall be punished with imprisonment for seven years and fine.",
        score=0.9,
        act="IPC",
        section="420"
    )
    
    # Chunk without penalty language
    chunk_without_penalty = RetrievedChunk(
        chunk_id="def",
        semantic_id="IPC_420_1",
        text="Section 420 deals with cheating and fraud.",
        score=0.9,
        act="IPC",
        section="420"
    )
    
    # With penalty language - sufficient
    sufficiency1 = is_evidence_sufficient(query, [chunk_with_penalty])
    assert sufficiency1.is_sufficient is True
    
    # Without penalty language - insufficient
    sufficiency2 = is_evidence_sufficient(query, [chunk_without_penalty])
    assert sufficiency2.is_sufficient is False
    
    print("✓ test_punishment_requires_penalty_language PASSED")


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all evidence sufficiency tests."""
    print("=" * 70)
    print("PHASE-3 EVIDENCE SUFFICIENCY - TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        # Query type classification
        test_classify_query_type_definition,
        test_classify_query_type_punishment,
        test_classify_query_type_procedure,
        test_classify_query_type_scope,
        
        # Sufficiency - positive cases
        test_evidence_sufficient_definition_query,
        test_evidence_sufficient_punishment_query,
        test_evidence_sufficient_general_query,
        test_evidence_sufficient_multiple_chunks,
        
        # Sufficiency - negative cases
        test_evidence_insufficient_no_chunks,
        test_evidence_insufficient_definition_without_definitional_language,
        test_evidence_insufficient_punishment_without_penalty_info,
        test_evidence_insufficient_irrelevant_chunks,
        
        # Edge cases
        test_evidence_sufficient_mixed_relevant_irrelevant,
        test_evidence_sufficient_case_insensitive,
        test_evidence_sufficient_partial_key_terms,
        test_evidence_insufficient_wrong_query_type,
        
        # Query type specific requirements
        test_definition_requires_definitional_language,
        test_punishment_requires_penalty_language,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
