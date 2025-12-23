#!/usr/bin/env python3
"""
Phase-2 RAG: C3 Grounded Generation Contract

Generates answers ONLY from retrieved evidence with strict validation.
No hallucination. No prior knowledge. Citations required.

Usage:
    python scripts/c3_generate.py "What is Section 420 IPC?"
    python scripts/c3_generate.py "What is Section 420 IPC?" --top-k 5
    python scripts/c3_generate.py "What is Section 420 IPC?" --model gpt-4

This script enforces the C3 contract:
1. Answers ONLY from retrieved evidence
2. All claims must be cited with [semantic_id]
3. Hard refusal if evidence insufficient
4. Post-generation validation
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

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
class GroundedAnswer:
    """Grounded answer with validation."""
    query: str
    answer: str
    is_grounded: bool
    cited_sources: List[str]
    invalid_citations: List[str]
    retrieved_semantic_ids: List[str]
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


def build_grounded_prompt(query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Build prompt that enforces grounded generation.
    
    The prompt MUST:
    - Enumerate chunks with semantic_id
    - Include ONLY retrieved text
    - Explicitly forbid outside knowledge
    - Require citations
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
    
    # Build strict instruction prompt
    prompt = f"""You are a legal research assistant. You MUST follow these rules STRICTLY:

1. Answer ONLY using the evidence documents provided below
2. You MUST cite sources using the format [semantic_id] (e.g., [IPC_420_0])
3. Every factual claim MUST have a citation
4. Do NOT use any prior knowledge or information not in the evidence
5. If the answer is not present in the evidence, you MUST say EXACTLY:
   "{REFUSAL_MESSAGE}"

{evidence_section}

QUESTION: {query}

ANSWER (with citations):"""
    
    return prompt


def extract_citations(answer_text: str) -> Set[str]:
    """Extract all citations from answer text.
    
    Citations are in format: [semantic_id]
    Examples: [IPC_420_0], [MinimumWagesAct_2_1]
    """
    pattern = r'\[([A-Za-z0-9_]+)\]'
    citations = re.findall(pattern, answer_text)
    return set(citations)


def enforce_c3(
    answer_text: str,
    retrieved_chunks: List[RetrievedChunk],
    require_citations: bool = True
) -> Tuple[bool, List[str], Optional[str]]:
    """Enforce C3 contract: validate answer is grounded in evidence.
    
    Returns:
        (is_valid, invalid_citations, refusal_reason)
    
    Rules:
    1. If answer is refusal message, it's valid
    2. Otherwise, must have citations
    3. All citations must reference retrieved chunks
    4. No invalid citations allowed
    """
    
    # Check if it's a refusal
    if answer_text.strip() == REFUSAL_MESSAGE:
        return True, [], None
    
    # Extract citations
    cited_sources = extract_citations(answer_text)
    
    # Get valid semantic IDs
    valid_semantic_ids = {chunk.semantic_id for chunk in retrieved_chunks if chunk.semantic_id}
    
    # Check for citations
    if require_citations and not cited_sources:
        return False, [], "Answer contains no citations"
    
    # Check all citations are valid
    invalid_citations = cited_sources - valid_semantic_ids
    if invalid_citations:
        return False, list(invalid_citations), f"Invalid citations: {', '.join(invalid_citations)}"
    
    # All checks passed
    return True, [], None


def generate_answer_mock(prompt: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Mock generation for testing (when no LLM available).
    
    This is a placeholder that demonstrates the expected format.
    In production, replace with actual LLM call.
    """
    
    # Check if we have relevant evidence
    if not retrieved_chunks:
        return REFUSAL_MESSAGE
    
    # Build a simple answer from first chunk
    chunk = retrieved_chunks[0]
    
    if chunk.act and "IPC" in chunk.act:
        return f"According to [{chunk.semantic_id}], {chunk.text[:100]}..."
    elif chunk.act and "Minimum Wages" in chunk.act:
        return f"Based on [{chunk.semantic_id}], {chunk.text[:100]}..."
    else:
        return REFUSAL_MESSAGE


def generate_answer_decoder(
    prompt: str,
    config: dict,
    temperature: float = 0.0
) -> str:
    """Generate answer using local decoder.
    
    Phase-4: Uses local decoder interface instead of OpenAI.
    
    Args:
        prompt: The grounded prompt
        config: Configuration dict with decoder settings
        temperature: Generation temperature (0.0 for deterministic)
    
    Returns:
        Generated answer text
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


def generate_grounded_answer(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    use_mock: bool = True,
    config: dict = None
) -> GroundedAnswer:
    """Generate and validate a grounded answer.
    
    This is the main C3 generation function that:
    1. Builds grounded prompt
    2. Generates answer (mock or decoder)
    3. Validates answer against C3 contract
    4. Returns validated result
    
    Phase-4: Uses local decoder interface instead of OpenAI.
    """
    
    # Build grounded prompt
    prompt = build_grounded_prompt(query, retrieved_chunks)
    
    # Generate answer
    if use_mock:
        answer_text = generate_answer_mock(prompt, retrieved_chunks)
    else:
        if config is None:
            raise ValueError("Config required for decoder generation")
        answer_text = generate_answer_decoder(prompt, config)
    
    # Extract citations
    cited_sources = list(extract_citations(answer_text))
    
    # Validate against C3 contract
    is_valid, invalid_citations, refusal_reason = enforce_c3(
        answer_text,
        retrieved_chunks,
        require_citations=True
    )
    
    # Get retrieved semantic IDs
    retrieved_semantic_ids = [c.semantic_id for c in retrieved_chunks if c.semantic_id]
    
    return GroundedAnswer(
        query=query,
        answer=answer_text,
        is_grounded=is_valid,
        cited_sources=cited_sources,
        invalid_citations=invalid_citations,
        retrieved_semantic_ids=retrieved_semantic_ids,
        refusal_reason=refusal_reason
    )


def print_grounded_answer(result: GroundedAnswer) -> None:
    """Print grounded answer with validation status."""
    
    print()
    print("=" * 70)
    print("C3 GROUNDED GENERATION RESULT")
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
    print(f"Grounded: {'✓ YES' if result.is_grounded else '✗ NO'}")
    print(f"Citations: {len(result.cited_sources)}")
    
    if result.cited_sources:
        print(f"Cited Sources: {', '.join(result.cited_sources)}")
    
    if result.invalid_citations:
        print(f"⚠ Invalid Citations: {', '.join(result.invalid_citations)}")
    
    if result.refusal_reason:
        print(f"⚠ Refusal Reason: {result.refusal_reason}")
    
    print(f"Retrieved Chunks: {len(result.retrieved_semantic_ids)}")
    print(f"Retrieved IDs: {', '.join(result.retrieved_semantic_ids[:5])}{'...' if len(result.retrieved_semantic_ids) > 5 else ''}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 RAG: C3 Grounded Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/c3_generate.py "What is Section 420 IPC?"
  python scripts/c3_generate.py "What is the minimum wage?" --top-k 3
  python scripts/c3_generate.py "What is cheating?" --use-llm --model gpt-4
        """
    )
    parser.add_argument(
        "query",
        help="Query text"
    )
    parser.add_argument(
        "--config",
        default="configs/phase1_rag.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use local decoder (Phase-4)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format"
    )
    args = parser.parse_args()
    
    # Check dependencies
    if not CHROMADB_AVAILABLE:
        log("ERROR: ChromaDB not available. Install with: pip install chromadb")
        sys.exit(1)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        log("ERROR: sentence-transformers not available. Install with: pip install sentence-transformers")
        sys.exit(1)
    
    if args.use_llm and not DECODER_AVAILABLE:
        log("ERROR: Decoder not available. Check src/decoders installation.")
        sys.exit(1)
    
    # Load configuration
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
        print("Phase-2 RAG: C3 Grounded Generation")
        print("=" * 70)
        log(f"Config: {args.config}")
        log(f"Query: {args.query}")
        log(f"Top-K: {args.top_k}")
        log(f"Mode: {'LLM' if args.use_llm else 'Mock'}")
    
    # Resolve paths
    chromadb_dir = PROJECT_ROOT / config['paths']['chromadb_dir']
    
    # Initialize ChromaDB
    if not args.json:
        log("Initializing ChromaDB...")
    
    client = chromadb.PersistentClient(
        path=str(chromadb_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection_name = config['retrieval']['collection_name']
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        log(f"ERROR: Collection '{collection_name}' not found. Run indexing first.")
        sys.exit(1)
    
    if not args.json:
        log(f"Loaded collection: {collection_name}")
        log(f"Collection size: {collection.count()} chunks")
    
    # Retrieve chunks
    if not args.json:
        log("Retrieving evidence chunks...")
    
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
    
    # Generate grounded answer
    if not args.json:
        log("Generating grounded answer...")
    
    result = generate_grounded_answer(
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
            phase="C2",
            is_sufficient=True,  # C2 doesn't check sufficiency
            is_grounded=result.is_grounded,
            cited_sources=result.cited_sources,
            retrieved_semantic_ids=result.retrieved_semantic_ids,
            answer_length=len(result.answer)
        )
        
        # Log grounding failure if applicable
        if not result.is_grounded or result.invalid_citations:
            failure_type = "invalid_citations" if result.invalid_citations else "contract_violation"
            log_grounding_failure(
                query=args.query,
                phase="C2",
                failure_type=failure_type,
                reason=result.refusal_reason or "Grounding validation failed",
                retrieved_semantic_ids=result.retrieved_semantic_ids,
                cited_sources=result.cited_sources,
                invalid_citations=result.invalid_citations
            )
        
        # Create and log audit record
        audit_record = create_audit_record(
            query=args.query,
            retrieved_semantic_ids=result.retrieved_semantic_ids,
            cited_ids=result.cited_sources,
            refusal_reason=result.refusal_reason,
            phase="C2",
            is_grounded=result.is_grounded,
            invalid_citations=result.invalid_citations if result.invalid_citations else None
        )
        
        log_audit_record(audit_record)
        
        if not args.json:
            print_audit_record(audit_record, verbose=args.json)
        
        # Determine exit code
        exit_code = get_exit_code(
            is_grounded=result.is_grounded,
            refusal_reason=result.refusal_reason,
            invalid_citations=result.invalid_citations
        )
    
    except ImportError:
        # Fallback if observability not available
        exit_code = 0 if result.is_grounded else 1
    
    # Output result
    if args.json:
        output = {
            "query": result.query,
            "answer": result.answer,
            "is_grounded": result.is_grounded,
            "cited_sources": result.cited_sources,
            "invalid_citations": result.invalid_citations,
            "retrieved_semantic_ids": result.retrieved_semantic_ids,
            "refusal_reason": result.refusal_reason,
        }
        print(json.dumps(output, indent=2))
    else:
        print_grounded_answer(result)
    
    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
