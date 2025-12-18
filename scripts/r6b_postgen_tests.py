#!/usr/bin/env python3
"""
Phase R6b Post-Generation Verification Tests

Tests that the post-generation verifier correctly catches:
1. Output without citations → REFUSE
2. Output with fake section → REFUSE
3. Output with wrong act → REFUSE
4. Output with fake court → REFUSE
5. Correct output → PASS

NO ML used. Deterministic verification only.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.postgen_verifier import PostGenerationVerifier, VerificationStatus


# ─────────────────────────────────────────────
# Test Definitions
# ─────────────────────────────────────────────

POSTGEN_TESTS = [
    # ─────────────────────────────────────────────
    # MUST REFUSE: Verification failures
    # ─────────────────────────────────────────────
    {
        "id": "T1_no_citation",
        "output": "Section 420 IPC deals with cheating. The punishment is up to 7 years.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "refuse",
        "expected_reason": "no_citation_in_output",
        "description": "Output without any citation markers"
    },
    {
        "id": "T2_fake_section",
        "output": "According to [1], Section 420 and Section 302 deal with crimes.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "refuse",
        "expected_reason": "hallucinated_section:302",
        "description": "Output mentions Section 302 not in evidence"
    },
    {
        "id": "T3_wrong_act",
        "output": "According to [1], the IT Act Section 420 applies here.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "refuse",
        "expected_reason": "hallucinated_act:it_act",
        "description": "Output mentions IT Act not in evidence"
    },
    {
        "id": "T4_fake_court",
        "output": "According to [1], the Supreme Court held in this case...",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "refuse",
        "expected_reason": "hallucinated_court:supreme_court",
        "description": "Output mentions Supreme Court not in evidence"
    },
    {
        "id": "T5_invalid_citation_number",
        "output": "According to [5], Section 420 deals with cheating.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "refuse",
        "expected_reason": "invalid_citation:[5]",
        "description": "Citation [5] doesn't exist (only 1 chunk)"
    },
    {
        "id": "T6_zero_citation",
        "output": "According to [0], Section 420 deals with cheating.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "refuse",
        "expected_reason": "invalid_citation:[0]",
        "description": "Citation [0] is invalid (1-indexed)"
    },
    # ─────────────────────────────────────────────
    # MUST PASS: Correct outputs
    # ─────────────────────────────────────────────
    {
        "id": "T7_correct_output",
        "output": "According to [1], Section 420 IPC deals with cheating. The punishment is up to 7 years.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "pass",
        "expected_reason": None,
        "description": "Correct output with valid citation"
    },
    {
        "id": "T8_multiple_citations",
        "output": "According to [1] and [2], Sections 420 and 415 IPC are related.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC"},
            {"chunk_id": "c2", "section": "415", "act": "IPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (IPC, Section 420, 1860)\nWhoever cheats...\n[2] (IPC, Section 415, 1860)\nCheating defined...\nEVIDENCE_END",
        "expected": "pass",
        "expected_reason": None,
        "description": "Correct output with multiple valid citations"
    },
    {
        "id": "T9_court_in_evidence",
        "output": "According to [1], the Supreme Court held that Section 420 applies.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "420", "act": "IPC", "court": "Supreme Court"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (Supreme Court, 2020, XYZ v State)\nThe court held...\nSOURCE: Case Law\nEVIDENCE_END",
        "expected": "pass",
        "expected_reason": None,
        "description": "Court mentioned is in evidence"
    },
    {
        "id": "T10_crpc_in_evidence",
        "output": "According to [1], Section 439 CrPC provides for bail.",
        "evidence_chunks": [
            {"chunk_id": "c1", "section": "439", "act": "CrPC"},
        ],
        "evidence_context": "EVIDENCE_START\n[1] (CrPC, Section 439, 1973)\nBail provisions...\nSOURCE: Bare Act\nEVIDENCE_END",
        "expected": "pass",
        "expected_reason": None,
        "description": "CrPC correctly referenced"
    },
]


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class TestOutcome:
    test_id: str
    result: TestResult
    reason: str


def run_tests() -> tuple[bool, List[TestOutcome]]:
    """Run all post-generation verification tests."""
    verifier = PostGenerationVerifier()
    outcomes = []
    
    for test in POSTGEN_TESTS:
        test_id = test["id"]
        output = test["output"]
        evidence_chunks = test["evidence_chunks"]
        evidence_context = test["evidence_context"]
        expected = test["expected"]
        expected_reason = test.get("expected_reason")
        
        # Run verification
        result = verifier.verify(
            output_text=output,
            evidence_chunks=evidence_chunks,
            evidence_context=evidence_context,
        )
        
        # Check result
        if expected == "pass":
            if result.status == VerificationStatus.PASS:
                outcomes.append(TestOutcome(test_id, TestResult.PASS, "Correctly passed"))
            else:
                outcomes.append(TestOutcome(test_id, TestResult.FAIL, f"Expected pass but got refuse: {result.refusal_reason}"))
        
        elif expected == "refuse":
            if result.status == VerificationStatus.REFUSE:
                # Check if reason matches
                if expected_reason and result.refusal_reason != expected_reason:
                    outcomes.append(TestOutcome(test_id, TestResult.FAIL, f"Wrong reason: expected '{expected_reason}', got '{result.refusal_reason}'"))
                else:
                    outcomes.append(TestOutcome(test_id, TestResult.PASS, f"Correctly refused: {result.refusal_reason}"))
            else:
                outcomes.append(TestOutcome(test_id, TestResult.FAIL, "Expected refuse but got pass"))
    
    all_passed = all(o.result == TestResult.PASS for o in outcomes)
    return all_passed, outcomes


def print_results(outcomes: List[TestOutcome]) -> None:
    """Print formatted test results."""
    print()
    print("PHASE R6b POST-GENERATION VERIFICATION TESTS")
    print("=" * 55)
    
    failed_tests = []
    
    for outcome in outcomes:
        test_id = outcome.test_id
        dots = "." * (40 - len(test_id))
        
        if outcome.result == TestResult.PASS:
            status = "PASS"
        else:
            status = "FAIL"
            failed_tests.append(test_id)
        
        print(f"{test_id} {dots} {status}")
        
        if outcome.result == TestResult.FAIL:
            print(f"    └─ {outcome.reason}")
    
    print("=" * 55)
    
    if not failed_tests:
        print("FINAL RESULT: PASS")
        print()
        print("✓ All unsafe outputs are overridden")
        print("✓ Zero hallucinated law passes through")
        print("✓ Refusals are structured")
    else:
        print("FINAL RESULT: FAIL")
        print(f"FAILED TESTS: {failed_tests}")
    
    print()


def main():
    """Run post-generation verification tests."""
    print("Running Phase R6b Post-Generation Verification Tests")
    
    all_passed, outcomes = run_tests()
    print_results(outcomes)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
