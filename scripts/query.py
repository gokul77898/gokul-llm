#!/usr/bin/env python3
"""
Phase-1 RAG: Query Script

Queries the indexed chunks using semantic search and returns relevant results.
Supports both interactive and single-query modes.

Usage:
    python scripts/query.py "What is the minimum wage?"
    python scripts/query.py --interactive
    python scripts/query.py --config configs/phase1_rag.yaml "query text"

This script is stateless and can be run independently.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
class QueryResult:
    """Result from query."""
    chunk_id: str
    semantic_id: str
    text: str
    score: float
    act: Optional[str]
    section: Optional[str]
    doc_type: str
    year: Optional[int]
    citation: Optional[str]
    court: Optional[str]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def format_result(result: QueryResult, index: int) -> str:
    """Format a single retrieval result for display."""
    sep = '-' * 60
    lines = [
        f"\n{sep}",
        f"Result {index + 1} | Score: {result.score:.4f}",
        f"{sep}",
    ]
    
    if result.act:
        lines.append(f"Act: {result.act}")
    if result.section:
        lines.append(f"Section: {result.section}")
    if result.year:
        lines.append(f"Year: {result.year}")
    if result.citation:
        lines.append(f"Citation: {result.citation}")
    if result.court:
        lines.append(f"Court: {result.court}")
    
    lines.append(f"Type: {result.doc_type}")
    lines.append("")
    
    # Truncate text if too long
    text = result.text
    if len(text) > 500:
        text = text[:500] + "..."
    lines.append(text)
    
    return '\n'.join(lines)


def run_query(collection, query_text: str, top_k: int, filter_act: str = None) -> List[QueryResult]:
    """Execute a query and return results."""
    # Build where clause for filtering
    where = None
    if filter_act:
        where = {"act": {"$eq": filter_act}}
    
    # Query collection
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    
    # Convert to QueryResult objects
    query_results = []
    
    if results and results['ids'] and results['ids'][0]:
        ids = results['ids'][0]
        documents = results['documents'][0] if results.get('documents') else [None] * len(ids)
        metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(ids)
        distances = results['distances'][0] if results.get('distances') else [0.0] * len(ids)
        
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            # Convert distance to similarity score (ChromaDB uses L2/cosine distance)
            score = 1.0 - (distances[i] if i < len(distances) else 0.0)
            
            query_results.append(QueryResult(
                chunk_id=chunk_id,
                semantic_id=meta.get('semantic_id') or "",
                text=documents[i] if i < len(documents) else "",
                score=score,
                act=meta.get('act') or None,
                section=meta.get('section') or None,
                doc_type=meta.get('doc_type', 'unknown'),
                year=meta.get('year') or None,
                citation=meta.get('citation') or None,
                court=meta.get('court') or None,
            ))
    
    return query_results


def main():
    parser = argparse.ArgumentParser(description="Phase-1 RAG: Query")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
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
        default=None,
        help="Number of results to return (overrides config)"
    )
    parser.add_argument(
        "--filter-act",
        default=None,
        help="Filter results by act name"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        log(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Resolve paths
    chromadb_dir = PROJECT_ROOT / config['paths']['chromadb_dir']
    
    # Get encoder config
    encoder_config = config.get('encoder', {})
    encoder_model = encoder_config.get('model_name', 'BAAI/bge-large-en-v1.5')
    
    # Get retrieval config
    retrieval_config = config.get('retrieval', {})
    collection_name = retrieval_config.get('collection_name', 'legal_chunks')
    top_k = args.top_k or retrieval_config.get('top_k', 5)
    
    # Print header (unless JSON output)
    if not args.json:
        print("=" * 60)
        print("Phase-1 RAG: Query")
        print("=" * 60)
        log(f"Config: {args.config}")
        log(f"Version: {config.get('version', 'unknown')}")
        log(f"Encoder: {encoder_model}")
        log(f"Collection: {collection_name}")
        log(f"Top-K: {top_k}")
        if args.filter_act:
            log(f"Filter Act: {args.filter_act}")
    
    # Check ChromaDB availability
    if not CHROMADB_AVAILABLE:
        log("ERROR: ChromaDB not installed. Run: pip install chromadb")
        sys.exit(1)
    
    # Check if ChromaDB exists
    if not chromadb_dir.exists():
        log("ERROR: ChromaDB not found. Run 'python scripts/index.py' first")
        sys.exit(1)
    
    if not args.json:
        print()
        log("Loading retriever...")
    
    # Initialize ChromaDB client
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
    except Exception as e:
        if not args.json:
            log(f"WARNING: Failed to load embedding model: {e}")
        embedding_fn = None
    
    # Get collection
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )
    except Exception as e:
        log(f"ERROR: Collection '{collection_name}' not found. Run 'python scripts/index.py' first")
        sys.exit(1)
    
    collection_count = collection.count()
    
    if not args.json:
        log(f"Loaded {collection_count} chunks from index")
    
    if collection_count == 0:
        log("WARNING: Index is empty. Run 'python scripts/index.py' first")
        sys.exit(0)
    
    # Interactive mode
    if args.interactive:
        print()
        print("Interactive Query Mode (type 'quit' to exit)")
        print("=" * 60)
        
        while True:
            try:
                query_text = input("\nQuery> ").strip()
                
                if not query_text:
                    continue
                
                if query_text.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                results = run_query(collection, query_text, top_k, args.filter_act)
                
                if not results:
                    print("No results found.")
                    continue
                
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results):
                    print(format_result(result, i))
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break
        
        sys.exit(0)
    
    # Single query mode
    if not args.query:
        parser.print_help()
        print("\nERROR: Query text required (or use --interactive)")
        sys.exit(1)
    
    query_text = args.query
    
    if not args.json:
        print()
        log(f"Query: {query_text}")
        print()
    
    # Execute query
    results = run_query(collection, query_text, top_k, args.filter_act)
    
    # Output results
    if args.json:
        output = {
            "query": query_text,
            "top_k": top_k,
            "filter_act": args.filter_act,
            "result_count": len(results),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "semantic_id": r.semantic_id,
                    "score": r.score,
                    "text": r.text,
                    "act": r.act,
                    "section": r.section,
                    "doc_type": r.doc_type,
                    "year": r.year,
                    "citation": r.citation,
                    "court": r.court,
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        if not results:
            log("No results found.")
        else:
            log(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(format_result(result, i))
        
        print()
        print("=" * 60)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
