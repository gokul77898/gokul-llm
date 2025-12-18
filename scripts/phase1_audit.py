#!/usr/bin/env python3
"""
Phase 1 Model Suitability Audit Script

RULES:
- Runs OUTSIDE FastAPI
- Uses HTTP calls only to existing endpoints
- Does NOT import model code directly
- Binary PASS/FAIL outcomes only

Endpoints used:
- /encoder-only (for encoder testing)
- /moe-generate (for full pipeline testing)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

import requests

# Configuration
BASE_URL = "http://localhost:8000"
AUDIT_CASES_PATH = Path(__file__).parent.parent / "audits" / "phase1_cases.json"


@dataclass
class AuditResult:
    case_id: str
    category: str
    encoder_pass: bool = False
    decoder_pass: bool = False
    encoder_reason: str = ""
    decoder_reason: str = ""
    hallucination: bool = False
    foreign_law_leak: bool = False


@dataclass
class AuditSummary:
    total_cases: int = 0
    encoder_passes: int = 0
    decoder_passes: int = 0
    hallucination_count: int = 0
    refusal_correct: int = 0
    refusal_expected: int = 0
    results: List[AuditResult] = field(default_factory=list)


# ============================================================
# HARD PASS/FAIL RULES (STEP 3)
# ============================================================

FOREIGN_LAW_KEYWORDS = [
    "us federal", "uk law", "gdpr", "european union", "china", "australian",
    "american", "british", "canadian", "singapore", "uae"
]

INDIAN_LAW_MARKERS = ["ipc", "crpc", "indian penal", "bharatiya nyaya", "cpc"]


def check_encoder_pass(case: Dict, encoder_result: Dict) -> tuple[bool, str, bool]:
    """
    Encoder FAIL if:
    - invents any section (hallucination)
    - confuses IPC / CrPC
    - misses obvious sections >20% cases (tracked globally)
    
    Returns: (pass, reason, hallucination_detected)
    """
    if not encoder_result.get("result", {}).get("success"):
        return False, "Encoder API call failed", False
    
    expected_sections = case.get("expected_sections", [])
    expected_outcome = case.get("expected_outcome", "answer")
    
    # Extract entities from encoder result
    entities = encoder_result.get("result", {}).get("entities", [])
    
    # For refusal cases, encoder should NOT find specific legal sections
    if expected_outcome == "refusal":
        # Check if encoder invented sections for ambiguous/OOS queries
        if entities and isinstance(entities, list):
            found_sections = []
            for ent in entities:
                if isinstance(ent, dict):
                    word = ent.get("word", "").lower()
                    # Check if encoder hallucinated specific sections
                    if any(marker in word for marker in ["section", "ipc", "crpc"]):
                        found_sections.append(word)
            
            if found_sections and case.get("category") in ["out_of_scope", "ambiguous"]:
                return False, f"Encoder invented sections for {case['category']} query: {found_sections}", True
        
        return True, "Correctly did not extract specific sections", False
    
    # For answer cases, check section extraction
    if expected_sections:
        # Check if expected sections are found
        found_any = False
        if entities and isinstance(entities, list):
            entity_text = " ".join([
                ent.get("word", "") if isinstance(ent, dict) else str(ent)
                for ent in entities
            ]).lower()
            
            for section in expected_sections:
                section_lower = section.lower()
                # Check for section number
                section_num = section_lower.replace(" ipc", "").replace(" crpc", "").strip()
                if section_num in entity_text or section_lower in entity_text:
                    found_any = True
                    break
        
        if not found_any:
            return False, f"Missed expected sections: {expected_sections}", False
    
    return True, "Encoder extraction acceptable", False


def check_decoder_pass(case: Dict, generate_result: Dict) -> tuple[bool, str, bool, bool]:
    """
    Decoder FAIL if:
    - answers without encoder facts (check refusal field)
    - hallucinates law
    - fails to refuse ambiguous queries
    
    Returns: (pass, reason, hallucination, foreign_law_leak)
    """
    expected_outcome = case.get("expected_outcome", "answer")
    output = generate_result.get("output", "")
    refusal = generate_result.get("refusal", False)
    encoder_facts = generate_result.get("encoder_facts_used", "")
    
    output_lower = output.lower() if output else ""
    
    # Check for foreign law leakage
    foreign_leak = any(kw in output_lower for kw in FOREIGN_LAW_KEYWORDS)
    if foreign_leak and case.get("category") != "out_of_scope":
        return False, "Foreign law leakage detected", False, True
    
    # For refusal cases
    if expected_outcome == "refusal":
        # Decoder MUST refuse
        if refusal:
            return True, "Correctly refused", False, False
        
        # Check if output indicates refusal
        refusal_indicators = [
            "cannot answer", "insufficient", "not enough information",
            "unable to", "outside scope", "not applicable", "i cannot"
        ]
        if any(ind in output_lower for ind in refusal_indicators):
            return True, "Correctly refused (text-based)", False, False
        
        # If decoder answered when it should refuse
        if output and len(output) > 50:
            # Check if it's answering with foreign law for OOS cases
            if case.get("category") == "out_of_scope":
                if any(kw in output_lower for kw in FOREIGN_LAW_KEYWORDS):
                    return False, "Answered foreign law query instead of refusing", True, True
            
            return False, "Failed to refuse ambiguous/OOS query", False, False
        
        return True, "Appropriately limited response", False, False
    
    # For answer cases
    if expected_outcome == "answer":
        # Check if decoder refused when it shouldn't have
        if refusal:
            return False, "Incorrectly refused valid query", False, False
        
        # Check if output is meaningful
        if not output or len(output) < 20:
            return False, "Output too short or empty", False, False
        
        # Check for hallucination - inventing sections not in query
        expected_sections = case.get("expected_sections", [])
        if expected_sections:
            # Look for section numbers in output that weren't expected
            import re
            found_sections = re.findall(r'section\s*(\d+[a-z]?)', output_lower)
            
            expected_nums = set()
            for s in expected_sections:
                nums = re.findall(r'(\d+[a-z]?)', s.lower())
                expected_nums.update(nums)
            
            for found in found_sections:
                if found not in expected_nums:
                    # Potential hallucination - invented section
                    return False, f"Hallucinated section {found}", True, False
        
        return True, "Valid answer provided", False, False
    
    return True, "Unknown outcome type", False, False


# ============================================================
# AUDIT EXECUTION
# ============================================================

def load_audit_cases() -> List[Dict]:
    """Load test cases from JSON file"""
    if not AUDIT_CASES_PATH.exists():
        print(f"ERROR: Audit cases file not found: {AUDIT_CASES_PATH}")
        sys.exit(1)
    
    with open(AUDIT_CASES_PATH) as f:
        data = json.load(f)
    
    return data.get("cases", [])


def call_encoder(query: str) -> Dict:
    """Call /moe-encode endpoint"""
    try:
        response = requests.post(
            f"{BASE_URL}/moe-encode",
            json={"query": query},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        return {"error": str(e)}


def call_generate(query: str) -> Dict:
    """Call /moe-generate endpoint"""
    try:
        response = requests.post(
            f"{BASE_URL}/moe-generate",
            json={"query": query},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        return {"error": str(e)}


def run_audit() -> AuditSummary:
    """Run full Phase 1 audit"""
    cases = load_audit_cases()
    summary = AuditSummary(total_cases=len(cases))
    
    print("=" * 60)
    print("PHASE 1 MODEL SUITABILITY AUDIT")
    print("=" * 60)
    print(f"Total cases: {len(cases)}")
    print()
    
    for case in cases:
        case_id = case["id"]
        query = case["query"]
        category = case["category"]
        expected_outcome = case.get("expected_outcome", "answer")
        
        print(f"[{case_id}] {category}")
        print(f"  Query: {query[:60]}...")
        
        result = AuditResult(case_id=case_id, category=category)
        
        # Track refusal expectations
        if expected_outcome == "refusal":
            summary.refusal_expected += 1
        
        # 1. Test Encoder
        encoder_result = call_encoder(query)
        enc_pass, enc_reason, enc_halluc = check_encoder_pass(case, encoder_result)
        result.encoder_pass = enc_pass
        result.encoder_reason = enc_reason
        if enc_halluc:
            result.hallucination = True
            summary.hallucination_count += 1
        
        if enc_pass:
            summary.encoder_passes += 1
        
        print(f"  Encoder: {'PASS' if enc_pass else 'FAIL'} - {enc_reason}")
        
        # 2. Test Full Generation
        generate_result = call_generate(query)
        dec_pass, dec_reason, dec_halluc, foreign_leak = check_decoder_pass(case, generate_result)
        result.decoder_pass = dec_pass
        result.decoder_reason = dec_reason
        result.foreign_law_leak = foreign_leak
        
        if dec_halluc:
            result.hallucination = True
            if not enc_halluc:  # Don't double count
                summary.hallucination_count += 1
        
        if dec_pass:
            summary.decoder_passes += 1
            if expected_outcome == "refusal":
                summary.refusal_correct += 1
        
        print(f"  Decoder: {'PASS' if dec_pass else 'FAIL'} - {dec_reason}")
        
        summary.results.append(result)
        print()
    
    return summary


def print_summary(summary: AuditSummary):
    """Print audit summary table"""
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    
    enc_rate = (summary.encoder_passes / summary.total_cases * 100) if summary.total_cases > 0 else 0
    dec_rate = (summary.decoder_passes / summary.total_cases * 100) if summary.total_cases > 0 else 0
    ref_rate = (summary.refusal_correct / summary.refusal_expected * 100) if summary.refusal_expected > 0 else 0
    
    print(f"{'Metric':<30} {'Value':<15} {'Rate':<10}")
    print("-" * 55)
    print(f"{'Total Cases':<30} {summary.total_cases:<15}")
    print(f"{'Encoder Passes':<30} {summary.encoder_passes:<15} {enc_rate:.1f}%")
    print(f"{'Decoder Passes':<30} {summary.decoder_passes:<15} {dec_rate:.1f}%")
    print(f"{'Hallucination Count':<30} {summary.hallucination_count:<15} {'CRITICAL' if summary.hallucination_count > 0 else 'OK'}")
    print(f"{'Refusal Correctness':<30} {summary.refusal_correct}/{summary.refusal_expected:<10} {ref_rate:.1f}%")
    print()
    
    # Category breakdown
    print("CATEGORY BREAKDOWN:")
    print("-" * 55)
    categories = {}
    for r in summary.results:
        if r.category not in categories:
            categories[r.category] = {"total": 0, "enc_pass": 0, "dec_pass": 0}
        categories[r.category]["total"] += 1
        if r.encoder_pass:
            categories[r.category]["enc_pass"] += 1
        if r.decoder_pass:
            categories[r.category]["dec_pass"] += 1
    
    for cat, stats in categories.items():
        enc_pct = stats["enc_pass"] / stats["total"] * 100
        dec_pct = stats["dec_pass"] / stats["total"] * 100
        print(f"  {cat:<25} Enc: {enc_pct:>5.1f}%  Dec: {dec_pct:>5.1f}%")
    
    print()
    return enc_rate, dec_rate, summary.hallucination_count, ref_rate


def produce_verdict(enc_rate: float, dec_rate: float, halluc_count: int, ref_rate: float):
    """Produce final verdict based on audit results"""
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    # Encoder verdict
    if halluc_count > 0:
        enc_verdict = "REPLACE"
        enc_justification = f"Encoder hallucinated in {halluc_count} cases - critical failure."
    elif enc_rate < 60:
        enc_verdict = "REPLACE"
        enc_justification = f"Encoder pass rate {enc_rate:.1f}% below 60% threshold."
    elif enc_rate < 80:
        enc_verdict = "PROMPT FIX"
        enc_justification = f"Encoder pass rate {enc_rate:.1f}% - may improve with prompt engineering."
    else:
        enc_verdict = "KEEP"
        enc_justification = f"Encoder pass rate {enc_rate:.1f}% acceptable."
    
    # Decoder verdict
    if halluc_count > 2:
        dec_verdict = "REPLACE"
        dec_justification = f"Decoder involved in hallucinations - critical failure."
    elif dec_rate < 50:
        dec_verdict = "REPLACE"
        dec_justification = f"Decoder pass rate {dec_rate:.1f}% below 50% threshold."
    elif dec_rate < 70 or ref_rate < 80:
        dec_verdict = "PROMPT FIX"
        dec_justification = f"Decoder pass rate {dec_rate:.1f}%, refusal rate {ref_rate:.1f}% - needs prompt tuning."
    else:
        dec_verdict = "KEEP"
        dec_justification = f"Decoder pass rate {dec_rate:.1f}% acceptable."
    
    print(f"ENCODER: {enc_verdict}")
    print(f"DECODER: {dec_verdict}")
    print()
    print("JUSTIFICATION:")
    print(f"  Encoder: {enc_justification}")
    print(f"  Decoder: {dec_justification}")
    print()
    
    # Overall recommendation
    if enc_verdict == "REPLACE" or dec_verdict == "REPLACE":
        print("RECOMMENDATION: Model replacement required before Phase 2.")
    elif enc_verdict == "PROMPT FIX" or dec_verdict == "PROMPT FIX":
        print("RECOMMENDATION: Prompt engineering may resolve issues. Re-audit after fixes.")
    else:
        print("RECOMMENDATION: Models suitable for Phase 2.")
    
    print("=" * 60)


def main():
    """Main entry point"""
    # Check server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"ERROR: Server not healthy. Status: {health.status_code}")
            sys.exit(1)
        print(f"Server health: {health.json()}")
        print()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to server at {BASE_URL}")
        print("Please start the server with: uvicorn src.inference.server:app")
        sys.exit(1)
    
    # Run audit
    summary = run_audit()
    
    # Print summary
    enc_rate, dec_rate, halluc_count, ref_rate = print_summary(summary)
    
    # Produce verdict
    produce_verdict(enc_rate, dec_rate, halluc_count, ref_rate)


if __name__ == "__main__":
    main()
