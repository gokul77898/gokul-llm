#!/usr/bin/env python3
"""
Phase-3 RAG: C3 Evidence-Sufficient Synthesis

Synthesizes answers from MULTIPLE chunks while preserving strict grounding.
Extends Phase-2 C3 with evidence sufficiency checking and multi-chunk citation.

Usage:
    python scripts/c3_synthesize.py "What is Section 420 IPC?"
    python scripts/c3_synthesize.py "What is the minimum wage?" --top-k 5
    python scripts/c3_synthesize.py "What is cheating?" --use-llm

This script enforces Phase-3 requirements:
1. Evidence sufficiency checking before generation
2. Multi-chunk citation enforcement
3. Claim coverage validation
4. Deterministic synthesis (temperature=0)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Phase-4: Local decoder interface
try:
    from src.decoders import get_decoder
    DECODER_AVAILABLE = True
except ImportError:
    DECODER_AVAILABLE = False


REFUSAL_MESSAGE = "I cannot answer based on the provided documents."


@dataclass
class RetrievedChunk:
    """Retrieved evidence chunk."""
    chunk_id: str
    semantic_id: str
    text: str
    score: float
    act: Optional[str]
    section: Optional[str]


@dataclass
class EvidenceSufficiency:
    """Evidence sufficiency assessment."""
    is_sufficient: bool
    reason: str
    relevant_chunks: List[str]
    query_type: str


@dataclass
class ClaimCoverage:
    """Coverage assessment for a single claim/sentence."""
    sentence: str
    is_covered: bool
    covering_chunks: List[str]
    reason: str


@dataclass
class SynthesizedAnswer:
    """Synthesized answer with validation."""
    query: str
    answer: str
    is_grounded: bool
    is_sufficient: bool
    cited_sources: List[str]
    invalid_citations: List[str]
    uncovered_claims: List[str]
    retrieved_semantic_ids: List[str]
    evidence_sufficiency: EvidenceSufficiency
    claim_coverage: List[ClaimCoverage]
    refusal_reason: Optional[str]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def retrieve_chunks(collection, query_text: str, top_k: int) -> List[RetrievedChunk]:
    """Retrieve chunks for a query."""
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved = []
    
    if results and results['ids'] and results['ids'][0]:
        ids = results['ids'][0]
        documents = results['documents'][0] if results.get('documents') else [None] * len(ids)
        metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(ids)
        distances = results['distances'][0] if results.get('distances') else [0.0] * len(ids)
        
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            score = 1.0 - (distances[i] if i < len(distances) else 0.0)
            text = documents[i] if i < len(documents) else ""
            
            retrieved.append(RetrievedChunk(
                chunk_id=chunk_id,
                semantic_id=meta.get('semantic_id', ''),
                text=text,
                score=score,
                act=meta.get('act'),
                section=meta.get('section'),
            ))
    
    return retrieved


def classify_query_type(query: str) -> str:
    """Classify query type for evidence sufficiency checking.
    
    Types:
    - definition: "What is...", "Define...", "What does X mean"
    - punishment: "What is the punishment", "penalty for"
    - procedure: "How to...", "What are the steps"
    - scope: "extent", "applicability"
    - general: other queries
    """
    query_lower = query.lower()
    
    # Check punishment first (more specific)
    if any(word in query_lower for word in ['punishment', 'penalty', 'sentence', 'imprisonment']):
        return 'punishment'
    # Check definition patterns
    elif any(pattern in query_lower for pattern in ['what is the definition', 'define ', 'definition of', 'what does', 'mean', 'meaning of', 'what is an', 'what is a ']):
        return 'definition'
    # Check procedure
    elif any(word in query_lower for word in ['how to', 'procedure', 'process', 'steps']):
        return 'procedure'
    # Check scope
    elif any(word in query_lower for word in ['extent', 'applicability', 'apply to', 'applies to', 'scope']):
        return 'scope'
    else:
        return 'general'


def is_evidence_sufficient(query: str, retrieved_chunks: List[RetrievedChunk]) -> EvidenceSufficiency:
    """Check if retrieved evidence is sufficient to answer the query.
    
    Rules:
    - At least 1 chunk must directly answer query
    - For definitions → explicit definitional language required
    - For punishments → explicit numeric/legal penalty required
    - Otherwise → check for relevant content
    
    Returns:
        EvidenceSufficiency with is_sufficient flag and reason
    """
    
    if not retrieved_chunks:
        return EvidenceSufficiency(
            is_sufficient=False,
            reason="No evidence chunks retrieved",
            relevant_chunks=[],
            query_type='unknown'
        )
    
    query_type = classify_query_type(query)
    query_lower = query.lower()
    relevant_chunks = []
    
    # Extract key terms from query (strip punctuation)
    key_terms = set()
    for word in query_lower.split():
        # Strip punctuation
        word_clean = word.strip('.,!?;:')
        if len(word_clean) > 3 and word_clean not in ['what', 'does', 'mean', 'the', 'is', 'are', 'for', 'with', 'from', 'this', 'that']:
            key_terms.add(word_clean)
    
    # Check each chunk for relevance
    for chunk in retrieved_chunks:
        chunk_text_lower = chunk.text.lower()
        
        # Check if chunk contains key terms
        term_matches = sum(1 for term in key_terms if term in chunk_text_lower)
        
        # More lenient matching: at least 1 key term OR high score
        if term_matches >= 1 or chunk.score >= 0.7:
            relevant_chunks.append(chunk.semantic_id)
    
    # Type-specific sufficiency checks
    if query_type == 'definition':
        # Check for definitional language
        for chunk in retrieved_chunks:
            chunk_text_lower = chunk.text.lower()
            if any(phrase in chunk_text_lower for phrase in [
                'means', 'defined as', 'refers to', 'is any', 'includes',
                'definition', 'shall mean'
            ]):
                if any(term in chunk_text_lower for term in key_terms):
                    return EvidenceSufficiency(
                        is_sufficient=True,
                        reason=f"Found definitional language in {chunk.semantic_id}",
                        relevant_chunks=relevant_chunks,
                        query_type=query_type
                    )
        
        return EvidenceSufficiency(
            is_sufficient=False,
            reason="No explicit definitional language found for definition query",
            relevant_chunks=relevant_chunks,
            query_type=query_type
        )
    
    elif query_type == 'punishment':
        # Check for penalty/punishment language
        for chunk in retrieved_chunks:
            chunk_text_lower = chunk.text.lower()
            has_penalty_language = any(phrase in chunk_text_lower for phrase in [
                'punishment', 'punished', 'penalty', 'imprisonment', 'fine', 'years',
                'shall be punished', 'liable to', 'sentenced'
            ])
            
            if has_penalty_language:
                # For punishment queries, having penalty language is sufficient
                # Don't require all key terms since query may ask about "Section 420"
                # but chunk may just describe the punishment without section number
                return EvidenceSufficiency(
                    is_sufficient=True,
                    reason=f"Found punishment/penalty language in {chunk.semantic_id}",
                    relevant_chunks=relevant_chunks,
                    query_type=query_type
                )
        
        return EvidenceSufficiency(
            is_sufficient=False,
            reason="No explicit punishment/penalty information found",
            relevant_chunks=relevant_chunks,
            query_type=query_type
        )
    
    else:
        # General sufficiency: at least one relevant chunk
        if relevant_chunks:
            return EvidenceSufficiency(
                is_sufficient=True,
                reason=f"Found {len(relevant_chunks)} relevant chunk(s)",
                relevant_chunks=relevant_chunks,
                query_type=query_type
            )
        else:
            return EvidenceSufficiency(
                is_sufficient=False,
                reason="No chunks contain sufficient relevant information",
                relevant_chunks=[],
                query_type=query_type
            )


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def extract_citations(answer_text: str) -> Set[str]:
    """Extract all citations from answer text.
    
    Citations are in format: [semantic_id]
    """
    pattern = r'\[([A-Za-z0-9_]+)\]'
    citations = re.findall(pattern, answer_text)
    return set(citations)


def check_claim_coverage(
    sentence: str,
    retrieved_chunks: List[RetrievedChunk],
    min_overlap: float = 0.3
) -> ClaimCoverage:
    """Check if a sentence/claim is covered by retrieved chunks.
    
    Coverage = semantic overlap + section match
    
    Args:
        sentence: The claim/sentence to check
        retrieved_chunks: Retrieved evidence chunks
        min_overlap: Minimum word overlap ratio (default 0.3)
    
    Returns:
        ClaimCoverage with is_covered flag and covering chunks
    """
    
    # Extract citations from sentence
    cited_ids = extract_citations(sentence)
    
    # If sentence has citations, check those chunks
    if cited_ids:
        covering_chunks = []
        for chunk in retrieved_chunks:
            if chunk.semantic_id in cited_ids:
                # Check semantic overlap
                sentence_words = set(sentence.lower().split())
                chunk_words = set(chunk.text.lower().split())
                
                # Remove common words and citations
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'per', 'both', 'are'}
                sentence_words -= stop_words
                chunk_words -= stop_words
                
                # Remove citation brackets from sentence words
                sentence_words = {w.strip('[]') for w in sentence_words if not w.startswith('[')}
                
                if sentence_words:
                    overlap = len(sentence_words & chunk_words) / len(sentence_words)
                    # For multi-citation sentences, be more lenient
                    effective_threshold = min_overlap * 0.7 if len(cited_ids) > 1 else min_overlap
                    if overlap >= effective_threshold:
                        covering_chunks.append(chunk.semantic_id)
        
        if covering_chunks:
            return ClaimCoverage(
                sentence=sentence,
                is_covered=True,
                covering_chunks=covering_chunks,
                reason=f"Covered by cited chunks: {', '.join(covering_chunks)}"
            )
        else:
            return ClaimCoverage(
                sentence=sentence,
                is_covered=False,
                covering_chunks=[],
                reason=f"Citations {cited_ids} present but insufficient semantic overlap"
            )
    
    # No citations - check if it's a meta-statement
    meta_phrases = [
        'based on', 'according to', 'the evidence', 'the documents',
        'i cannot', 'not present', 'does not contain'
    ]
    if any(phrase in sentence.lower() for phrase in meta_phrases):
        return ClaimCoverage(
            sentence=sentence,
            is_covered=True,
            covering_chunks=[],
            reason="Meta-statement (does not require evidence)"
        )
    
    # No citations and not meta-statement
    return ClaimCoverage(
        sentence=sentence,
        is_covered=False,
        covering_chunks=[],
        reason="No citations and not a meta-statement"
    )


def validate_multi_chunk_citations(
    answer_text: str,
    retrieved_chunks: List[RetrievedChunk]
) -> Tuple[bool, List[str], str]:
    """Validate that answer cites all chunks it uses information from.
    
    Rules:
    - If answer uses info from N chunks, must cite all N semantic_ids
    - No "primary source only" shortcuts
    
    Returns:
        (is_valid, missing_citations, reason)
    """
    
    # Get cited sources
    cited_ids = extract_citations(answer_text)
    
    # Check each sentence for coverage
    sentences = split_into_sentences(answer_text)
    used_chunks = set()
    
    for sentence in sentences:
        coverage = check_claim_coverage(sentence, retrieved_chunks)
        if coverage.is_covered and coverage.covering_chunks:
            used_chunks.update(coverage.covering_chunks)
    
    # Check if all used chunks are cited
    missing_citations = used_chunks - cited_ids
    
    if missing_citations:
        return False, list(missing_citations), f"Answer uses information from {missing_citations} but does not cite them"
    
    return True, [], "All used chunks are properly cited"


def build_synthesis_prompt(query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Build prompt for multi-chunk synthesis.
    
    Enforces:
    - Evidence-only answers
    - Multi-chunk citation
    - Deterministic synthesis
    - Legal-style summarization
    """
    
    # Build evidence section
    evidence_lines = []
    evidence_lines.append("EVIDENCE DOCUMENTS:")
    evidence_lines.append("")
    
    for chunk in retrieved_chunks:
        evidence_lines.append(f"[{chunk.semantic_id}]")
        if chunk.act:
            evidence_lines.append(f"Act: {chunk.act}")
        if chunk.section:
            evidence_lines.append(f"Section: {chunk.section}")
        evidence_lines.append(f"Text: {chunk.text}")
        evidence_lines.append("")
    
    evidence_section = "\n".join(evidence_lines)
    
    # Build strict synthesis prompt
    prompt = f"""You are a legal research assistant. You MUST follow these rules STRICTLY:

1. Answer ONLY using the evidence documents provided below
2. You MUST cite ALL sources you use with [semantic_id] format
3. If you use information from multiple documents, cite ALL of them
4. Every factual claim MUST have a citation
5. Use precise, legal-style language - no creative phrasing
6. Do NOT use any prior knowledge or information not in the evidence
7. If the answer is not present in the evidence, you MUST say EXACTLY:
   "{REFUSAL_MESSAGE}"

SYNTHESIS RULES:
- Combine information from multiple sources when relevant
- Cite each source you draw from: [ID1], [ID2], etc.
- Maintain factual accuracy - do not paraphrase creatively
- Use direct quotes or close paraphrases with citations

{evidence_section}

QUESTION: {query}

ANSWER (with citations for ALL sources used):"""
    
    return prompt


def generate_answer_mock(prompt: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Mock synthesis for testing (when no LLM available)."""
    
    if not retrieved_chunks:
        return REFUSAL_MESSAGE
    
    # Build a simple synthesized answer from multiple chunks
    if len(retrieved_chunks) >= 2:
        chunk1 = retrieved_chunks[0]
        chunk2 = retrieved_chunks[1]
        
        # Try to synthesize from multiple sources
        return f"According to [{chunk1.semantic_id}], {chunk1.text[:80]}... Additionally, [{chunk2.semantic_id}] states that {chunk2.text[:80]}..."
    else:
        chunk = retrieved_chunks[0]
        return f"Based on [{chunk.semantic_id}], {chunk.text[:100]}..."


def generate_answer_decoder(
    prompt: str,
    config: dict,
    temperature: float = 0.0
) -> str:
    """Generate answer using local decoder with deterministic synthesis.
    
    Phase-4: Uses local decoder interface instead of OpenAI.
    """
    
    if not DECODER_AVAILABLE:
        raise RuntimeError("Decoder not available. Check src/decoders installation.")
    
    try:
        decoder = get_decoder(config.get('decoder', {}))
        answer = decoder.generate(prompt, max_tokens=500, temperature=temperature)
        return answer.strip()
    
    except Exception as e:
        log(f"ERROR: Decoder generation failed: {e}")
        return REFUSAL_MESSAGE


def synthesize_answer(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    use_mock: bool = True,
    config: dict = None
) -> SynthesizedAnswer:
    """Synthesize answer from multiple chunks with full validation.
    
    Phase-3 synthesis pipeline:
    1. Check evidence sufficiency
    2. Generate synthesis (if sufficient)
    3. Validate multi-chunk citations
    4. Check claim coverage
    5. Return validated result
    
    Phase-4: Uses local decoder interface instead of OpenAI.
    """
    
    # Step 1: Check evidence sufficiency
    sufficiency = is_evidence_sufficient(query, retrieved_chunks)
    
    if not sufficiency.is_sufficient:
        # Force refusal if evidence insufficient
        return SynthesizedAnswer(
            query=query,
            answer=REFUSAL_MESSAGE,
            is_grounded=True,  # Refusal is valid
            is_sufficient=False,
            cited_sources=[],
            invalid_citations=[],
            uncovered_claims=[],
            retrieved_semantic_ids=[c.semantic_id for c in retrieved_chunks if c.semantic_id],
            evidence_sufficiency=sufficiency,
            claim_coverage=[],
            refusal_reason=sufficiency.reason
        )
    
    # Step 2: Build synthesis prompt
    prompt = build_synthesis_prompt(query, retrieved_chunks)
    
    # Step 3: Generate answer
    if use_mock:
        answer_text = generate_answer_mock(prompt, retrieved_chunks)
    else:
        if config is None:
            raise ValueError("Config required for decoder generation")
        answer_text = generate_answer_decoder(prompt, config, temperature=0.0)
    
    # Step 4: Extract citations
    cited_sources = list(extract_citations(answer_text))
    
    # Step 5: Validate citations are valid
    valid_semantic_ids = {c.semantic_id for c in retrieved_chunks if c.semantic_id}
    invalid_citations = list(set(cited_sources) - valid_semantic_ids)
    
    # Step 6: Validate multi-chunk citations
    citations_valid, missing_citations, citation_reason = validate_multi_chunk_citations(
        answer_text,
        retrieved_chunks
    )
    
    if missing_citations:
        invalid_citations.extend(missing_citations)
    
    # Step 7: Check claim coverage
    sentences = split_into_sentences(answer_text)
    claim_coverage = []
    uncovered_claims = []
    
    for sentence in sentences:
        coverage = check_claim_coverage(sentence, retrieved_chunks)
        claim_coverage.append(coverage)
        if not coverage.is_covered:
            uncovered_claims.append(sentence)
    
    # Step 8: Determine if answer is grounded
    is_grounded = (
        answer_text.strip() == REFUSAL_MESSAGE or
        (not invalid_citations and not uncovered_claims)
    )
    
    refusal_reason = None
    if not is_grounded:
        if invalid_citations:
            refusal_reason = f"Invalid citations: {', '.join(invalid_citations)}"
        elif uncovered_claims:
            refusal_reason = f"{len(uncovered_claims)} claim(s) lack coverage"
    
    return SynthesizedAnswer(
        query=query,
        answer=answer_text,
        is_grounded=is_grounded,
        is_sufficient=sufficiency.is_sufficient,
        cited_sources=cited_sources,
        invalid_citations=invalid_citations,
        uncovered_claims=uncovered_claims,
        retrieved_semantic_ids=[c.semantic_id for c in retrieved_chunks if c.semantic_id],
        evidence_sufficiency=sufficiency,
        claim_coverage=claim_coverage,
        refusal_reason=refusal_reason
    )


def print_synthesis_result(result: SynthesizedAnswer, verbose: bool = False) -> None:
    """Print synthesis result with validation details."""
    
    print()
    print("=" * 70)
    print("PHASE-3 C3 EVIDENCE-SUFFICIENT SYNTHESIS")
    print("=" * 70)
    print()
    print(f"Query: {result.query}")
    print()
    print(f"Answer:")
    print(f"{result.answer}")
    print()
    print("-" * 70)
    print("VALIDATION")
    print("-" * 70)
    print(f"Evidence Sufficient: {'✓ YES' if result.is_sufficient else '✗ NO'}")
    if not result.is_sufficient:
        print(f"  Reason: {result.evidence_sufficiency.reason}")
    print(f"Grounded: {'✓ YES' if result.is_grounded else '✗ NO'}")
    print(f"Citations: {len(result.cited_sources)}")
    
    if result.cited_sources:
        print(f"Cited Sources: {', '.join(result.cited_sources)}")
    
    if result.invalid_citations:
        print(f"⚠ Invalid Citations: {', '.join(result.invalid_citations)}")
    
    if result.uncovered_claims:
        print(f"⚠ Uncovered Claims: {len(result.uncovered_claims)}")
        if verbose:
            for claim in result.uncovered_claims:
                print(f"  - {claim}")
    
    if result.refusal_reason:
        print(f"⚠ Refusal Reason: {result.refusal_reason}")
    
    print(f"Retrieved Chunks: {len(result.retrieved_semantic_ids)}")
    
    if verbose and result.claim_coverage:
        print()
        print("-" * 70)
        print("CLAIM COVERAGE DETAILS")
        print("-" * 70)
        for i, coverage in enumerate(result.claim_coverage, 1):
            status = "✓" if coverage.is_covered else "✗"
            print(f"{status} Claim {i}: {coverage.sentence[:60]}...")
            print(f"  Covering: {', '.join(coverage.covering_chunks) if coverage.covering_chunks else 'None'}")
            print(f"  Reason: {coverage.reason}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Phase-3 RAG: C3 Evidence-Sufficient Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", help="Query text")
    parser.add_argument("--config", default="configs/phase1_rag.yaml", help="Config file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--use-llm", action="store_true", help="Use local decoder (Phase-4)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Check dependencies
    if not CHROMADB_AVAILABLE:
        log("ERROR: ChromaDB not available")
        sys.exit(1)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        log("ERROR: sentence-transformers not available")
        sys.exit(1)
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        log(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Phase-6: Health check on startup
    try:
        from src.runtime import health_check, print_health_check
        
        if not args.json:
            log("Running system health check...")
        
        health_result = health_check(config)
        
        if not args.json and not health_result.healthy:
            print_health_check(health_result, verbose=True)
            log("ERROR: System health check failed")
            sys.exit(1)
        
        if not args.json:
            log("Health check passed")
    
    except ImportError:
        log("WARNING: Health check not available (src.runtime not found)")
    
    # Print header
    if not args.json:
        print("=" * 70)
        print("Phase-3 RAG: C3 Evidence-Sufficient Synthesis")
        print("=" * 70)
        log(f"Query: {args.query}")
        log(f"Top-K: {args.top_k}")
        log(f"Mode: {'LLM' if args.use_llm else 'Mock'}")
    
    # Initialize ChromaDB
    chromadb_dir = PROJECT_ROOT / config['paths']['chromadb_dir']
    client = chromadb.PersistentClient(
        path=str(chromadb_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection_name = config['retrieval']['collection_name']
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        log(f"ERROR: Collection '{collection_name}' not found")
        sys.exit(1)
    
    if not args.json:
        log(f"Collection size: {collection.count()} chunks")
        log("Retrieving evidence...")
    
    # Retrieve chunks
    retrieved_chunks = retrieve_chunks(collection, args.query, args.top_k)
    
    if not args.json:
        log(f"Retrieved {len(retrieved_chunks)} chunks")
    
    # Phase-7: Log retrieval
    try:
        from src.observability import log_retrieval
        
        log_retrieval(
            query=args.query,
            retrieved_count=len(retrieved_chunks),
            semantic_ids=[c.semantic_id for c in retrieved_chunks if c.semantic_id],
            top_k=args.top_k,
            collection_name=collection_name
        )
    except ImportError:
        pass
    
    if not args.json:
        log("Synthesizing answer...")
    
    # Synthesize answer
    result = synthesize_answer(
        query=args.query,
        retrieved_chunks=retrieved_chunks,
        use_mock=not args.use_llm,
        config=config if args.use_llm else None
    )
    
    # Phase-7: Log synthesis and create audit record
    try:
        from src.observability import (
            log_synthesis,
            log_grounding_failure,
            create_audit_record,
            log_audit_record,
            print_audit_record,
            get_exit_code
        )
        
        # Log synthesis
        log_synthesis(
            query=args.query,
            phase="C3+",
            is_sufficient=result.is_sufficient,
            is_grounded=result.is_grounded,
            cited_sources=result.cited_sources,
            retrieved_semantic_ids=result.retrieved_semantic_ids,
            answer_length=len(result.answer)
        )
        
        # Log grounding failure if applicable
        if not result.is_grounded or not result.is_sufficient or result.invalid_citations or result.uncovered_claims:
            if not result.is_sufficient:
                failure_type = "insufficient_evidence"
            elif result.invalid_citations:
                failure_type = "invalid_citations"
            elif result.uncovered_claims:
                failure_type = "uncovered_claims"
            else:
                failure_type = "contract_violation"
            
            log_grounding_failure(
                query=args.query,
                phase="C3+",
                failure_type=failure_type,
                reason=result.refusal_reason or result.evidence_sufficiency.reason or "Grounding validation failed",
                retrieved_semantic_ids=result.retrieved_semantic_ids,
                cited_sources=result.cited_sources,
                invalid_citations=result.invalid_citations,
                uncovered_claims=result.uncovered_claims
            )
        
        # Create and log audit record
        audit_record = create_audit_record(
            query=args.query,
            retrieved_semantic_ids=result.retrieved_semantic_ids,
            cited_ids=result.cited_sources,
            refusal_reason=result.refusal_reason,
            phase="C3+",
            is_grounded=result.is_grounded,
            is_sufficient=result.is_sufficient,
            invalid_citations=result.invalid_citations if result.invalid_citations else None,
            uncovered_claims=result.uncovered_claims if result.uncovered_claims else None
        )
        
        log_audit_record(audit_record)
        
        if not args.json:
            print_audit_record(audit_record, verbose=args.verbose)
        
        # Determine exit code
        exit_code = get_exit_code(
            is_grounded=result.is_grounded,
            refusal_reason=result.refusal_reason,
            invalid_citations=result.invalid_citations,
            uncovered_claims=result.uncovered_claims
        )
    
    except ImportError:
        # Fallback if observability not available
        exit_code = 0 if result.is_grounded else 1
    
    # Output
    if args.json:
        output = {
            "query": result.query,
            "answer": result.answer,
            "is_grounded": result.is_grounded,
            "is_sufficient": result.is_sufficient,
            "cited_sources": result.cited_sources,
            "invalid_citations": result.invalid_citations,
            "uncovered_claims": result.uncovered_claims,
            "evidence_sufficiency": {
                "is_sufficient": result.evidence_sufficiency.is_sufficient,
                "reason": result.evidence_sufficiency.reason,
                "query_type": result.evidence_sufficiency.query_type
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print_synthesis_result(result, verbose=args.verbose)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
