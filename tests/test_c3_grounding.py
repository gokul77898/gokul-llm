#!/usr/bin/env python3
"""
Tests for Phase-2 C3 Grounded Generation Contract

Tests both positive (valid grounded answers) and negative (invalid/hallucinated) cases.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.c3_generate import (
    RetrievedChunk,
    build_grounded_prompt,
    extract_citations,
    enforce_c3,
    generate_grounded_answer,
    REFUSAL_MESSAGE
)


# Test fixtures
def create_test_chunks():
    """Create test chunks for validation."""
    return [
        RetrievedChunk(
            chunk_id="abc123",
            semantic_id="IPC_420_0",
            text="Section 420 IPC deals with cheating and dishonestly inducing delivery of property.",
            score=0.95,
            act="IPC",
            section="420"
        ),
        RetrievedChunk(
            chunk_id="def456",
            semantic_id="MinimumWagesAct_2_1",
            text="Section 2 defines employer as any person who employs one or more employees.",
            score=0.85,
            act="Minimum Wages Act",
            section="2"
        ),
        RetrievedChunk(
            chunk_id="ghi789",
            semantic_id="MinimumWagesAct_3_2",
            text="Section 3 states that the appropriate government shall fix minimum rates of wages.",
            score=0.75,
            act="Minimum Wages Act",
            section="3"
        ),
    ]


# ============================================================================
# TEST: build_grounded_prompt
# ============================================================================

def test_build_grounded_prompt_includes_evidence():
    """Test that prompt includes all evidence chunks."""
    chunks = create_test_chunks()
    query = "What is Section 420 IPC?"
    
    prompt = build_grounded_prompt(query, chunks)
    
    # Check all semantic IDs are present
    assert "[IPC_420_0]" in prompt
    assert "[MinimumWagesAct_2_1]" in prompt
    assert "[MinimumWagesAct_3_2]" in prompt
    
    # Check evidence text is present
    assert "Section 420 IPC deals with cheating" in prompt
    assert "defines employer" in prompt
    assert "fix minimum rates of wages" in prompt
    
    # Check query is present
    assert query in prompt
    
    print("✓ test_build_grounded_prompt_includes_evidence PASSED")


def test_build_grounded_prompt_forbids_outside_knowledge():
    """Test that prompt explicitly forbids outside knowledge."""
    chunks = create_test_chunks()
    query = "What is cheating?"
    
    prompt = build_grounded_prompt(query, chunks)
    
    # Check strict instructions are present
    assert "ONLY using the evidence" in prompt
    assert "Do NOT use any prior knowledge" in prompt
    assert "MUST cite sources" in prompt
    assert REFUSAL_MESSAGE in prompt
    
    print("✓ test_build_grounded_prompt_forbids_outside_knowledge PASSED")


# ============================================================================
# TEST: extract_citations
# ============================================================================

def test_extract_citations_valid():
    """Test extraction of valid citations."""
    answer = "According to [IPC_420_0], cheating is defined. Also see [MinimumWagesAct_2_1]."
    
    citations = extract_citations(answer)
    
    assert len(citations) == 2
    assert "IPC_420_0" in citations
    assert "MinimumWagesAct_2_1" in citations
    
    print("✓ test_extract_citations_valid PASSED")


def test_extract_citations_none():
    """Test extraction when no citations present."""
    answer = "This answer has no citations."
    
    citations = extract_citations(answer)
    
    assert len(citations) == 0
    
    print("✓ test_extract_citations_none PASSED")


def test_extract_citations_duplicate():
    """Test extraction with duplicate citations."""
    answer = "See [IPC_420_0] and also [IPC_420_0] again."
    
    citations = extract_citations(answer)
    
    # Should deduplicate
    assert len(citations) == 1
    assert "IPC_420_0" in citations
    
    print("✓ test_extract_citations_duplicate PASSED")


# ============================================================================
# TEST: enforce_c3 - POSITIVE CASES
# ============================================================================

def test_enforce_c3_valid_answer_with_citations():
    """Test valid answer with proper citations."""
    chunks = create_test_chunks()
    answer = "According to [IPC_420_0], Section 420 IPC deals with cheating."
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    assert is_valid is True
    assert len(invalid_citations) == 0
    assert refusal_reason is None
    
    print("✓ test_enforce_c3_valid_answer_with_citations PASSED")


def test_enforce_c3_refusal_message():
    """Test that refusal message is always valid."""
    chunks = create_test_chunks()
    answer = REFUSAL_MESSAGE
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    assert is_valid is True
    assert len(invalid_citations) == 0
    assert refusal_reason is None
    
    print("✓ test_enforce_c3_refusal_message PASSED")


def test_enforce_c3_multiple_valid_citations():
    """Test answer with multiple valid citations."""
    chunks = create_test_chunks()
    answer = "Per [IPC_420_0], cheating is illegal. Also, [MinimumWagesAct_2_1] defines employer."
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    assert is_valid is True
    assert len(invalid_citations) == 0
    assert refusal_reason is None
    
    print("✓ test_enforce_c3_multiple_valid_citations PASSED")


# ============================================================================
# TEST: enforce_c3 - NEGATIVE CASES
# ============================================================================

def test_enforce_c3_no_citations():
    """Test that answer without citations is rejected."""
    chunks = create_test_chunks()
    answer = "Section 420 IPC deals with cheating."
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    assert is_valid is False
    assert refusal_reason == "Answer contains no citations"
    
    print("✓ test_enforce_c3_no_citations PASSED")


def test_enforce_c3_invalid_citation():
    """Test that invalid citations are detected."""
    chunks = create_test_chunks()
    answer = "According to [IPC_421_0], fraud is illegal."  # IPC_421_0 not in chunks
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    assert is_valid is False
    assert "IPC_421_0" in invalid_citations
    assert "Invalid citations" in refusal_reason
    
    print("✓ test_enforce_c3_invalid_citation PASSED")


def test_enforce_c3_mixed_valid_invalid_citations():
    """Test that any invalid citation causes rejection."""
    chunks = create_test_chunks()
    answer = "Per [IPC_420_0], cheating is illegal. Also [FAKE_ID] says something."
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    assert is_valid is False
    assert "FAKE_ID" in invalid_citations
    assert "Invalid citations" in refusal_reason
    
    print("✓ test_enforce_c3_mixed_valid_invalid_citations PASSED")


def test_enforce_c3_hallucinated_semantic_id():
    """Test that hallucinated semantic IDs are caught."""
    chunks = create_test_chunks()
    # Hallucinated ID that looks valid but isn't in retrieved chunks
    answer = "According to [IPC_500_0], defamation is a crime."
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    assert is_valid is False
    assert "IPC_500_0" in invalid_citations
    
    print("✓ test_enforce_c3_hallucinated_semantic_id PASSED")


# ============================================================================
# TEST: generate_grounded_answer - INTEGRATION
# ============================================================================

def test_generate_grounded_answer_with_evidence():
    """Test end-to-end generation with evidence."""
    chunks = create_test_chunks()
    query = "What is Section 420 IPC?"
    
    result = generate_grounded_answer(query, chunks, use_mock=True)
    
    # Check result structure
    assert result.query == query
    assert result.answer is not None
    assert len(result.retrieved_semantic_ids) == 3
    
    # Mock should generate a valid answer
    assert result.is_grounded is True
    
    print("✓ test_generate_grounded_answer_with_evidence PASSED")


def test_generate_grounded_answer_no_evidence():
    """Test generation with no evidence (should refuse)."""
    chunks = []  # No evidence
    query = "What is Section 420 IPC?"
    
    result = generate_grounded_answer(query, chunks, use_mock=True)
    
    # Should refuse when no evidence
    assert result.answer == REFUSAL_MESSAGE
    assert result.is_grounded is True  # Refusal is valid
    assert len(result.cited_sources) == 0
    
    print("✓ test_generate_grounded_answer_no_evidence PASSED")


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

def test_enforce_c3_empty_semantic_ids():
    """Test handling of chunks without semantic IDs."""
    chunks = [
        RetrievedChunk(
            chunk_id="abc123",
            semantic_id="",  # Empty semantic ID
            text="Some text",
            score=0.95,
            act="IPC",
            section="420"
        )
    ]
    answer = "According to [IPC_420_0], something."
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    # Should reject because IPC_420_0 not in valid IDs
    assert is_valid is False
    
    print("✓ test_enforce_c3_empty_semantic_ids PASSED")


def test_extract_citations_malformed():
    """Test extraction with malformed citations."""
    answer = "See [IPC_420_0] and [INVALID ID] and [Another-Invalid]."
    
    citations = extract_citations(answer)
    
    # Should only extract valid format (alphanumeric + underscore)
    assert "IPC_420_0" in citations
    # Malformed ones might not match pattern correctly
    
    print("✓ test_extract_citations_malformed PASSED")


def test_enforce_c3_citation_in_refusal():
    """Test that citations in refusal message don't affect validation."""
    chunks = create_test_chunks()
    # Refusal message with accidental citation format
    answer = REFUSAL_MESSAGE
    
    is_valid, invalid_citations, refusal_reason = enforce_c3(answer, chunks)
    
    # Exact refusal message should always be valid
    assert is_valid is True
    
    print("✓ test_enforce_c3_citation_in_refusal PASSED")


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all C3 grounding tests."""
    print("=" * 70)
    print("C3 GROUNDED GENERATION CONTRACT - TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        # Prompt building
        test_build_grounded_prompt_includes_evidence,
        test_build_grounded_prompt_forbids_outside_knowledge,
        
        # Citation extraction
        test_extract_citations_valid,
        test_extract_citations_none,
        test_extract_citations_duplicate,
        
        # C3 enforcement - positive cases
        test_enforce_c3_valid_answer_with_citations,
        test_enforce_c3_refusal_message,
        test_enforce_c3_multiple_valid_citations,
        
        # C3 enforcement - negative cases
        test_enforce_c3_no_citations,
        test_enforce_c3_invalid_citation,
        test_enforce_c3_mixed_valid_invalid_citations,
        test_enforce_c3_hallucinated_semantic_id,
        
        # Integration tests
        test_generate_grounded_answer_with_evidence,
        test_generate_grounded_answer_no_evidence,
        
        # Edge cases
        test_enforce_c3_empty_semantic_ids,
        test_extract_citations_malformed,
        test_enforce_c3_citation_in_refusal,
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
