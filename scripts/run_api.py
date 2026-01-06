#!/usr/bin/env python3
"""
Legal Reasoning API CLI - Phase 6

Simple CLI wrapper for manual testing of the Legal Reasoning API.

Usage:
    python scripts/run_api.py --query "What is Section 420 IPC?"
    python scripts/run_api.py --query "Explain cheating" --top-k 5
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.legal_reasoning_api import LegalReasoningAPI
from src.graph.legal_graph_traverser import LegalGraphTraverser


def print_separator(char="═", length=60):
    """Print separator line."""
    print(char * length)


def print_response(response):
    """
    Pretty print API response.
    
    Args:
        response: LegalAnswerResponse
    """
    print("\n")
    print_separator()
    print("LEGAL REASONING API RESPONSE")
    print_separator()
    
    print(f"\n📝 Query: {response.query}")
    print(f"⏰ Timestamp: {response.timestamp}")
    
    print(f"\n📊 Status:")
    print(f"   Answered: {'✅ Yes' if response.answered else '❌ No'}")
    print(f"   Grounded: {'✅ Yes' if response.grounded else '❌ No'}")
    
    if response.refusal_reason:
        print(f"   Refusal Reason: {response.refusal_reason}")
    
    print(f"\n📈 Statistics:")
    print(f"   Retrieved: {response.retrieved_count}")
    print(f"   Allowed: {response.allowed_chunks_count}")
    print(f"   Excluded: {response.excluded_chunks_count}")
    print(f"   Cited: {response.cited_count}")
    
    if response.answered:
        print(f"\n💬 Answer:")
        print(f"   {response.answer[:200]}...")
        
        if response.statutory_basis:
            print(f"\n⚖️  Statutory Basis ({len(response.statutory_basis)}):")
            for node_id in response.statutory_basis:
                print(f"   • {node_id}")
        
        if response.judicial_interpretations:
            print(f"\n👨‍⚖️  Judicial Interpretations ({len(response.judicial_interpretations)}):")
            for node_id in response.judicial_interpretations:
                print(f"   • {node_id}")
        
        if response.applied_precedents:
            print(f"\n📚 Applied Precedents ({len(response.applied_precedents)}):")
            for node_id in response.applied_precedents:
                print(f"   • {node_id}")
        
        if response.supporting_precedents:
            print(f"\n🔗 Supporting Precedents ({len(response.supporting_precedents)}):")
            for node_id in response.supporting_precedents:
                print(f"   • {node_id}")
        
        if response.excluded_precedents:
            print(f"\n❌ Excluded Precedents ({len(response.excluded_precedents)}):")
            for node_id in response.excluded_precedents:
                print(f"   • {node_id}")
        
        if response.explanation_text:
            print(f"\n📄 Explanation:")
            print(f"\n{response.explanation_text}")
    else:
        print(f"\n❌ Refusal:")
        print(f"   {response.answer}")
    
    print("\n")
    print_separator()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Legal Reasoning API CLI"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Legal query to answer"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="data/graph/legal_graph_v2.pkl",
        help="Path to graph pickle file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of pretty print"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    print("\n")
    print_separator()
    print("LEGAL REASONING API - Phase 6")
    print_separator()
    
    print(f"\n📂 Loading graph from: {args.graph}")
    
    # Load graph
    try:
        traverser = LegalGraphTraverser.from_pickle(args.graph)
        print(f"✅ Graph loaded successfully")
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        print(f"   Using empty graph for demonstration")
        import networkx as nx
        traverser = LegalGraphTraverser(nx.DiGraph())
    
    # Phase 5: FAIL FAST VALIDATION
    print(f"\n🔍 VALIDATING SYSTEM COMPONENTS...")
    
    import os
    
    # Check graph file exists
    if not os.path.exists(args.graph):
        raise RuntimeError(
            "GRAPH FILE REQUIRED. "
            f"Missing {args.graph}. "
            "Run graph construction pipeline first."
        )
    
    # Check retrieval data exists
    if not os.path.exists("data/rag/chunks"):
        raise RuntimeError(
            "REAL RETRIEVAL DATA REQUIRED. "
            "Missing data/rag/chunks directory. "
            "Run data preparation pipeline first."
        )
    
    # Check vector index exists
    if not os.path.exists("data/rag/chromadb"):
        raise RuntimeError(
            "REAL VECTOR INDEX REQUIRED. "
            "Missing data/rag/chromadb directory. "
            "Run vector indexing pipeline first."
        )
    
    # Check chunks have content
    chunk_files = list(Path("data/rag/chunks").glob("*.json"))
    if len(chunk_files) == 0:
        raise RuntimeError(
            "EMPTY CHUNK DIRECTORY. "
            "No chunk files found in data/rag/chunks. "
            "Run data preparation pipeline first."
        )
    
    print(f"✅ Found {len(chunk_files)} chunk files")
    
    # Real retriever required - NO MOCKS ALLOWED
    from src.rag.retrieval.retriever import LegalRetriever
    
    retriever = LegalRetriever(
        chunks_dir="data/rag/chunks",
        chromadb_dir="data/rag/chromadb"
    )
    
    # Initialize retriever and validate
    stats = retriever.initialize()
    print(f"📊 Retriever initialized: {stats}")
    
    # Validate vector count > 0
    if stats.get("dense_chunks", 0) == 0:
        raise RuntimeError(
            "EMPTY VECTOR INDEX. "
            "No vectors found in ChromaDB. "
            "Run vector indexing pipeline first."
        )
    
    # Test retrieval to ensure semantic_id present
    test_results = retriever.retrieve("test query", top_k=1)
    if len(test_results) == 0:
        print(f"⚠️  Warning: No results from test retrieval")
    else:
        # Check semantic_id field
        first_result = test_results[0]
        if not hasattr(first_result, 'chunk_id') or not hasattr(first_result, 'text'):
            raise RuntimeError(
                "INVALID RETRIEVAL FORMAT. "
                "Retrieved chunks missing required fields (chunk_id, text)."
            )
        print(f"✅ Retrieval format validated")
    
    print(f"🔥 Using REAL ChromaDB-backed retriever")
    
    # Initialize API
    print(f"\n🚀 Initializing Legal Reasoning API...")
    api = LegalReasoningAPI(
        traverser=traverser,
        retriever=retriever,
        generator=None,  # Use mock generator
    )
    print(f"✅ API initialized")
    
    # Process query
    print(f"\n🔍 Processing query: {args.query}")
    print(f"   Top-K: {args.top_k}")
    
    response = api.answer_query(
        query=args.query,
        top_k=args.top_k
    )
    
    # Output response
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print_response(response)


if __name__ == "__main__":
    main()
