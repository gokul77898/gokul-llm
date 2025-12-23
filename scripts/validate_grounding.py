#!/usr/bin/env python3
"""
Phase-2 RAG: Grounded Generation Validator

Validates that every answer sentence maps to retrieved chunk IDs.
Enforces refusal if evidence is missing or insufficient.

Usage:
    python scripts/validate_grounding.py --answer "text" --chunks chunk1,chunk2
    python scripts/validate_grounding.py --answer-file answer.txt --chunks-file chunks.json

This script ensures no hallucination by requiring evidence for every claim.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Sentence:
    """A sentence from the answer."""
    index: int
    text: str
    requires_evidence: bool  # False for meta-statements like "Based on the evidence..."


@dataclass
class GroundingResult:
    """Result of grounding validation."""
    sentence_index: int
    sentence_text: str
    is_grounded: bool
    supporting_chunks: List[str]
    confidence: float
    reason: str


@dataclass
class ValidationReport:
    """Overall validation report."""
    timestamp: str
    answer: str
    total_sentences: int
    sentences_requiring_evidence: int
    grounded_sentences: int
    ungrounded_sentences: int
    grounding_rate: float
    is_valid: bool
    should_refuse: bool
    refusal_reason: Optional[str]
    sentence_results: List[Dict]


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting (can be improved with NLP libraries)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def classify_sentence(sentence: str) -> bool:
    """
    Determine if a sentence requires evidence.
    
    Meta-statements (don't require evidence):
    - "Based on the evidence..."
    - "According to the retrieved documents..."
    - "I cannot answer this question..."
    - "The evidence does not contain..."
    
    Returns:
        True if sentence requires evidence, False if it's a meta-statement
    """
    meta_patterns = [
        r'^based on (the )?(evidence|documents|retrieved)',
        r'^according to (the )?(evidence|documents|retrieved)',
        r'^i (cannot|can\'t|am unable to)',
        r'^the (evidence|documents|retrieved).*(does not|doesn\'t|do not|don\'t)',
        r'^(unfortunately|regrettably)',
        r'^(however|but|although)',
        r'^this (answer|response) is (based|grounded)',
    ]
    
    sentence_lower = sentence.lower().strip()
    
    for pattern in meta_patterns:
        if re.match(pattern, sentence_lower):
            return False
    
    return True


def check_grounding(
    sentence: str,
    chunks: List[Dict],
    min_overlap_ratio: float = 0.3
) -> Tuple[bool, List[str], float, str]:
    """
    Check if a sentence is grounded in the provided chunks.
    
    Args:
        sentence: The sentence to check
        chunks: List of chunk dictionaries with 'chunk_id' and 'text'
        min_overlap_ratio: Minimum word overlap ratio to consider grounded
        
    Returns:
        (is_grounded, supporting_chunk_ids, confidence, reason)
    """
    # Extract key terms from sentence (simple word-based approach)
    sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    sentence_words = sentence_words - stop_words
    
    if not sentence_words:
        return False, [], 0.0, "No meaningful words in sentence"
    
    # Check each chunk for overlap
    supporting_chunks = []
    max_overlap = 0.0
    
    for chunk in chunks:
        chunk_text = chunk.get('text', '')
        chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))
        chunk_words = chunk_words - stop_words
        
        if not chunk_words:
            continue
        
        # Compute overlap
        overlap = len(sentence_words & chunk_words)
        overlap_ratio = overlap / len(sentence_words)
        
        if overlap_ratio >= min_overlap_ratio:
            supporting_chunks.append(chunk['chunk_id'])
            max_overlap = max(max_overlap, overlap_ratio)
    
    is_grounded = len(supporting_chunks) > 0
    
    if is_grounded:
        reason = f"Found {len(supporting_chunks)} supporting chunk(s) with {max_overlap:.2%} overlap"
    else:
        reason = f"No chunks with sufficient overlap (max: {max_overlap:.2%}, required: {min_overlap_ratio:.2%})"
    
    return is_grounded, supporting_chunks, max_overlap, reason


def validate_answer(
    answer: str,
    chunks: List[Dict],
    min_overlap_ratio: float = 0.3,
    require_all_grounded: bool = True
) -> ValidationReport:
    """
    Validate that an answer is properly grounded in evidence.
    
    Args:
        answer: The generated answer text
        chunks: List of retrieved chunks
        min_overlap_ratio: Minimum overlap to consider grounded
        require_all_grounded: If True, all sentences must be grounded
        
    Returns:
        ValidationReport with grounding analysis
    """
    # Split into sentences
    sentences = split_into_sentences(answer)
    
    # Classify and validate each sentence
    sentence_results = []
    grounded_count = 0
    ungrounded_count = 0
    sentences_requiring_evidence = 0
    
    for i, sentence_text in enumerate(sentences):
        requires_evidence = classify_sentence(sentence_text)
        
        if requires_evidence:
            sentences_requiring_evidence += 1
            is_grounded, supporting_chunks, confidence, reason = check_grounding(
                sentence_text, chunks, min_overlap_ratio
            )
            
            if is_grounded:
                grounded_count += 1
            else:
                ungrounded_count += 1
        else:
            # Meta-statement, automatically grounded
            is_grounded = True
            supporting_chunks = []
            confidence = 1.0
            reason = "Meta-statement (does not require evidence)"
        
        result = GroundingResult(
            sentence_index=i,
            sentence_text=sentence_text,
            is_grounded=is_grounded,
            supporting_chunks=supporting_chunks,
            confidence=confidence,
            reason=reason,
        )
        sentence_results.append(result)
    
    # Compute overall metrics
    grounding_rate = grounded_count / sentences_requiring_evidence if sentences_requiring_evidence > 0 else 1.0
    
    # Determine if valid
    if require_all_grounded:
        is_valid = ungrounded_count == 0
    else:
        is_valid = grounding_rate >= 0.8  # 80% threshold
    
    # Determine if should refuse
    should_refuse = not is_valid
    refusal_reason = None
    
    if should_refuse:
        if ungrounded_count > 0:
            refusal_reason = f"{ungrounded_count} sentence(s) lack supporting evidence"
        else:
            refusal_reason = f"Grounding rate {grounding_rate:.2%} below threshold"
    
    return ValidationReport(
        timestamp=datetime.utcnow().isoformat(),
        answer=answer,
        total_sentences=len(sentences),
        sentences_requiring_evidence=sentences_requiring_evidence,
        grounded_sentences=grounded_count,
        ungrounded_sentences=ungrounded_count,
        grounding_rate=grounding_rate,
        is_valid=is_valid,
        should_refuse=should_refuse,
        refusal_reason=refusal_reason,
        sentence_results=[asdict(r) for r in sentence_results],
    )


def print_validation_report(report: ValidationReport, verbose: bool = False) -> None:
    """Print validation report to console."""
    
    print()
    print("=" * 70)
    print("GROUNDED GENERATION VALIDATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print()
    print(f"Total Sentences: {report.total_sentences}")
    print(f"Sentences Requiring Evidence: {report.sentences_requiring_evidence}")
    print(f"Grounded Sentences: {report.grounded_sentences}")
    print(f"Ungrounded Sentences: {report.ungrounded_sentences}")
    print(f"Grounding Rate: {report.grounding_rate:.2%}")
    print()
    print(f"Valid: {report.is_valid}")
    print(f"Should Refuse: {report.should_refuse}")
    
    if report.refusal_reason:
        print(f"Refusal Reason: {report.refusal_reason}")
    
    print()
    print(f"Status: {'✓ PASS' if report.is_valid else '✗ FAIL'}")
    print("=" * 70)
    
    if verbose:
        print()
        print("SENTENCE-LEVEL RESULTS:")
        print("-" * 70)
        
        for result in report.sentence_results:
            status = "✓" if result['is_grounded'] else "✗"
            print(f"\n{status} Sentence {result['sentence_index'] + 1}:")
            print(f"   {result['sentence_text'][:80]}...")
            print(f"   Grounded: {result['is_grounded']}")
            
            if result['supporting_chunks']:
                print(f"   Supporting: {', '.join(result['supporting_chunks'][:3])}")
            
            print(f"   Reason: {result['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Phase-2 RAG: Grounded Generation Validator")
    parser.add_argument(
        "--answer",
        default=None,
        help="Answer text to validate"
    )
    parser.add_argument(
        "--answer-file",
        default=None,
        help="Path to file containing answer text"
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help="Comma-separated chunk IDs"
    )
    parser.add_argument(
        "--chunks-file",
        default=None,
        help="Path to JSON file with chunks (list of {chunk_id, text})"
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/rag/chunks",
        help="Directory containing chunk JSON files"
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.3,
        help="Minimum word overlap ratio (default: 0.3)"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial grounding (80%% threshold instead of 100%%)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed sentence-level results"
    )
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("Phase-2 RAG: Grounded Generation Validator")
    print("=" * 70)
    
    # Get answer text
    if args.answer:
        answer = args.answer
    elif args.answer_file:
        answer_path = Path(args.answer_file)
        if not answer_path.exists():
            log(f"ERROR: Answer file not found: {answer_path}")
            sys.exit(1)
        with open(answer_path, 'r') as f:
            answer = f.read().strip()
    else:
        log("ERROR: Must provide --answer or --answer-file")
        sys.exit(1)
    
    log(f"Answer length: {len(answer)} characters")
    
    # Get chunks
    chunks = []
    
    if args.chunks_file:
        chunks_file = Path(args.chunks_file)
        if not chunks_file.exists():
            log(f"ERROR: Chunks file not found: {chunks_file}")
            sys.exit(1)
        
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
    
    elif args.chunks:
        # Load chunks from directory
        chunks_dir = PROJECT_ROOT / args.chunks_dir
        chunk_ids = args.chunks.split(',')
        
        for chunk_id in chunk_ids:
            chunk_path = chunks_dir / f"{chunk_id.strip()}.json"
            if chunk_path.exists():
                with open(chunk_path, 'r') as f:
                    chunk_data = json.load(f)
                    chunks.append({
                        'chunk_id': chunk_data.get('chunk_id', chunk_id),
                        'text': chunk_data.get('text', ''),
                    })
            else:
                log(f"WARNING: Chunk not found: {chunk_id}")
    else:
        log("ERROR: Must provide --chunks or --chunks-file")
        sys.exit(1)
    
    log(f"Loaded {len(chunks)} chunks")
    
    # Validate
    print()
    log("Validating grounding...")
    
    report = validate_answer(
        answer=answer,
        chunks=chunks,
        min_overlap_ratio=args.min_overlap,
        require_all_grounded=not args.allow_partial,
    )
    
    # Print report
    print_validation_report(report, verbose=args.verbose)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print()
        log(f"Report saved to: {output_path}")
    
    # Exit with appropriate code
    print()
    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
