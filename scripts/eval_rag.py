#!/usr/bin/env python3
"""
Phase-2 RAG: Evaluation Script

Evaluates retrieval quality using test queries with known expected evidence.
Computes Recall@k, Precision@k, and reports missing/incorrect evidence.

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py --config configs/phase1_rag.yaml
    python scripts/eval_rag.py --top-k 10
    python scripts/eval_rag.py --output eval/results.json

This script is deterministic and suitable for CI usage.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@dataclass
class EvalQuery:
    """Evaluation query with expected evidence."""
    query_id: str
    query: str
    expected_chunks: List[str]
    expected_acts: List[str]
    category: str


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    chunk_id: str
    semantic_id: str
    score: float
    act: Optional[str]
    section: Optional[str]
    text_snippet: str


@dataclass
class QueryEvalResult:
    """Evaluation result for a single query."""
    query_id: str
    query: str
    category: str
    retrieved_count: int
    expected_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    recall: float
    precision: float
    retrieved_chunks: List[str]
    expected_chunks: List[str]
    missing_chunks: List[str]
    incorrect_chunks: List[str]


@dataclass
class EvalReport:
    """Overall evaluation report."""
    timestamp: str
    total_queries: int
    top_k: int
    mean_recall: float
    mean_precision: float
    mean_f1: float
    queries_with_perfect_recall: int
    queries_with_zero_recall: int
    pass_threshold: float
    passed: bool
    query_results: List[Dict]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def load_eval_queries(queries_file: Path) -> List[EvalQuery]:
    """Load evaluation queries from JSONL file."""
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            queries.append(EvalQuery(
                query_id=data['query_id'],
                query=data['query'],
                expected_chunks=data.get('expected_chunks', []),
                expected_acts=data.get('expected_acts', []),
                category=data.get('category', 'general'),
            ))
    return queries


def retrieve_chunks(collection, query_text: str, top_k: int) -> List[RetrievalResult]:
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
            
            # Get text snippet (first 100 chars)
            text_snippet = text[:100] + "..." if len(text) > 100 else text
            
            retrieved.append(RetrievalResult(
                chunk_id=chunk_id,
                semantic_id=meta.get('semantic_id', ''),
                score=score,
                act=meta.get('act'),
                section=meta.get('section'),
                text_snippet=text_snippet,
            ))
    
    return retrieved


def evaluate_query(
    eval_query: EvalQuery,
    retrieved: List[RetrievalResult]
) -> QueryEvalResult:
    """Evaluate retrieval for a single query using EXACT semantic_id matching."""
    
    # Get retrieved semantic IDs (NOT chunk_ids)
    retrieved_ids = {r.semantic_id for r in retrieved if r.semantic_id}
    expected_ids = set(eval_query.expected_chunks)
    
    # EXACT matching only - no fuzzy logic
    true_positives = len(retrieved_ids & expected_ids)
    false_positives = len(retrieved_ids - expected_ids)
    false_negatives = len(expected_ids - retrieved_ids)
    
    # Compute metrics
    recall = true_positives / len(expected_ids) if expected_ids else 0.0
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
    
    # Identify missing and incorrect semantic IDs
    missing_chunks = list(expected_ids - retrieved_ids)
    incorrect_chunks = list(retrieved_ids - expected_ids)
    
    return QueryEvalResult(
        query_id=eval_query.query_id,
        query=eval_query.query,
        category=eval_query.category,
        retrieved_count=len(retrieved_ids),
        expected_count=len(expected_ids),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        recall=recall,
        precision=precision,
        retrieved_chunks=list(retrieved_ids),
        expected_chunks=list(expected_ids),
        missing_chunks=missing_chunks,
        incorrect_chunks=incorrect_chunks,
    )


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def generate_report(
    query_results: List[QueryEvalResult],
    top_k: int,
    pass_threshold: float = 0.7
) -> EvalReport:
    """Generate overall evaluation report."""
    
    if not query_results:
        return EvalReport(
            timestamp=datetime.utcnow().isoformat(),
            total_queries=0,
            top_k=top_k,
            mean_recall=0.0,
            mean_precision=0.0,
            mean_f1=0.0,
            queries_with_perfect_recall=0,
            queries_with_zero_recall=0,
            pass_threshold=pass_threshold,
            passed=False,
            query_results=[],
        )
    
    # Compute aggregate metrics
    mean_recall = sum(r.recall for r in query_results) / len(query_results)
    mean_precision = sum(r.precision for r in query_results) / len(query_results)
    mean_f1 = compute_f1(mean_precision, mean_recall)
    
    queries_with_perfect_recall = sum(1 for r in query_results if r.recall == 1.0)
    queries_with_zero_recall = sum(1 for r in query_results if r.recall == 0.0)
    
    # Pass if mean recall >= threshold
    passed = mean_recall >= pass_threshold
    
    return EvalReport(
        timestamp=datetime.utcnow().isoformat(),
        total_queries=len(query_results),
        top_k=top_k,
        mean_recall=mean_recall,
        mean_precision=mean_precision,
        mean_f1=mean_f1,
        queries_with_perfect_recall=queries_with_perfect_recall,
        queries_with_zero_recall=queries_with_zero_recall,
        pass_threshold=pass_threshold,
        passed=passed,
        query_results=[asdict(r) for r in query_results],
    )


def print_report(report: EvalReport, verbose: bool = False) -> None:
    """Print evaluation report to console."""
    
    print()
    print("=" * 70)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Queries: {report.total_queries}")
    print(f"Top-K: {report.top_k}")
    print()
    print(f"Mean Recall@{report.top_k}:    {report.mean_recall:.4f}")
    print(f"Mean Precision@{report.top_k}: {report.mean_precision:.4f}")
    print(f"Mean F1 Score:      {report.mean_f1:.4f}")
    print()
    print(f"Perfect Recall (1.0): {report.queries_with_perfect_recall}/{report.total_queries}")
    print(f"Zero Recall (0.0):    {report.queries_with_zero_recall}/{report.total_queries}")
    print()
    print(f"Pass Threshold: {report.pass_threshold:.2f}")
    print(f"Status: {'✓ PASS' if report.passed else '✗ FAIL'}")
    print("=" * 70)
    
    if verbose:
        print()
        print("QUERY-LEVEL RESULTS:")
        print("-" * 70)
        
        for result in report.query_results:
            print(f"\nQuery ID: {result['query_id']}")
            print(f"Query: {result['query'][:60]}...")
            print(f"Category: {result['category']}")
            print(f"Recall: {result['recall']:.4f} | Precision: {result['precision']:.4f}")
            print(f"Retrieved: {result['retrieved_count']} | Expected: {result['expected_count']}")
            
            if result['missing_chunks']:
                print(f"Missing: {', '.join(result['missing_chunks'][:3])}")
            if result['incorrect_chunks']:
                print(f"Incorrect: {', '.join(result['incorrect_chunks'][:3])}")


def main():
    parser = argparse.ArgumentParser(description="Phase-2 RAG: Evaluation")
    parser.add_argument(
        "--config",
        default="configs/phase1_rag.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--queries",
        default="eval/queries.jsonl",
        help="Path to evaluation queries file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to retrieve (overrides config)"
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.7,
        help="Minimum mean recall to pass (default: 0.7)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed query-level results"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        log(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Print header
    print("=" * 70)
    print("Phase-2 RAG: Retrieval Evaluation")
    print("=" * 70)
    log(f"Config: {args.config}")
    log(f"Version: {config.get('version', 'unknown')}")
    
    # Resolve paths
    chromadb_dir = PROJECT_ROOT / config['paths']['chromadb_dir']
    queries_file = PROJECT_ROOT / args.queries
    
    # Get config
    encoder_config = config.get('encoder', {})
    encoder_model = encoder_config.get('model_name', 'BAAI/bge-large-en-v1.5')
    retrieval_config = config.get('retrieval', {})
    collection_name = retrieval_config.get('collection_name', 'legal_chunks')
    top_k = args.top_k or retrieval_config.get('top_k', 5)
    
    log(f"Queries file: {queries_file}")
    log(f"Top-K: {top_k}")
    log(f"Pass threshold: {args.pass_threshold}")
    
    # Check ChromaDB availability
    if not CHROMADB_AVAILABLE:
        log("ERROR: ChromaDB not installed. Run: pip install chromadb")
        sys.exit(1)
    
    # Check if ChromaDB exists
    if not chromadb_dir.exists():
        log("ERROR: ChromaDB not found. Run 'python scripts/index.py' first")
        sys.exit(1)
    
    # Check if queries file exists
    if not queries_file.exists():
        log(f"ERROR: Queries file not found: {queries_file}")
        sys.exit(1)
    
    # Load evaluation queries
    print()
    log("Loading evaluation queries...")
    eval_queries = load_eval_queries(queries_file)
    log(f"Loaded {len(eval_queries)} evaluation queries")
    
    # Initialize ChromaDB
    print()
    log("Initializing ChromaDB...")
    
    client = chromadb.PersistentClient(
        path=str(chromadb_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    
    # Create embedding function
    try:
        from chromadb.utils import embedding_functions
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=encoder_model
        )
        log("Embedding model loaded")
    except Exception as e:
        log(f"WARNING: Failed to load embedding model: {e}")
        embedding_fn = None
    
    # Get collection
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )
    except Exception as e:
        log(f"ERROR: Collection '{collection_name}' not found")
        sys.exit(1)
    
    collection_count = collection.count()
    log(f"Loaded collection with {collection_count} chunks")
    
    # Run evaluation
    print()
    log("Running evaluation...")
    
    query_results = []
    
    for i, eval_query in enumerate(eval_queries, 1):
        # Retrieve chunks
        retrieved = retrieve_chunks(collection, eval_query.query, top_k)
        
        # Evaluate using EXACT semantic_id matching
        result = evaluate_query(eval_query, retrieved)
        query_results.append(result)
        
        # Progress
        status = "✓" if result.recall >= args.pass_threshold else "✗"
        log(f"{status} [{i}/{len(eval_queries)}] {eval_query.query_id}: R={result.recall:.2f} P={result.precision:.2f}")
    
    # Generate report
    report = generate_report(query_results, top_k, args.pass_threshold)
    
    # Print report
    print_report(report, verbose=args.verbose)
    
    # Save report if requested
    if args.output:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print()
        log(f"Report saved to: {output_path}")
    
    # Exit with appropriate code
    print()
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
