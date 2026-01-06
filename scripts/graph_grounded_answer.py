#!/usr/bin/env python3
"""
Graph-Aware Grounded Answer CLI

Phase 4: Graph-Aware Grounded Generation

Demonstrates the full pipeline:
- Retrieval
- Graph filtering
- Grounded generation
- Citation validation

Usage:
    python scripts/graph_grounded_answer.py --query "What is Section 420 IPC?"
    python scripts/graph_grounded_answer.py --query "Punishment for cheating?" --top-k 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.graph_grounded_generator import (
    GraphGroundedGenerator,
    GroundedAnswerResult,
)
from src.graph.graph_rag_filter import GraphRAGFilter
from src.graph.legal_graph_traverser import LegalGraphTraverser


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_banner() -> None:
    """Print CLI banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           GRAPH-AWARE GROUNDED ANSWER - Phase 4               ║
║           Graph-Constrained Grounded Generation               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def create_real_retriever():
    """Create real retriever - NO MOCKS ALLOWED."""
    # Check if required data exists
    import os
    if not os.path.exists("data/rag/chunks"):
        raise RuntimeError(
            "REAL RETRIEVAL DATA REQUIRED. "
            "Missing data/rag/chunks directory. "
            "Run data preparation pipeline first."
        )
    
    if not os.path.exists("data/rag/chromadb"):
        raise RuntimeError(
            "REAL VECTOR INDEX REQUIRED. "
            "Missing data/rag/chromadb directory. "
            "Run vector indexing pipeline first."
        )
    
    from src.rag.retrieval.retriever import LegalRetriever
    retriever = LegalRetriever(
        chunks_dir="data/rag/chunks",
        chromadb_dir="data/rag/chromadb"
    )
    
    # Initialize retriever
    stats = retriever.initialize()
    print(f"📊 Real retriever initialized: {stats}")
    
    return retriever


def print_pipeline_step(step: str, description: str) -> None:
    """Print pipeline step."""
    print(f"\n{'─' * 60}")
    print(f"  STEP {step}: {description}")
    print(f"{'─' * 60}")


def print_retrieved_chunks(chunks: list) -> None:
    """Print retrieved chunks."""
    print(f"\n  Retrieved {len(chunks)} chunks:\n")
    
    for i, chunk in enumerate(chunks[:5], 1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        doc_type = chunk.get("doc_type", "unknown")
        act = chunk.get("act", "N/A")
        section = chunk.get("section", "N/A")
        
        print(f"  {i}. {chunk_id}")
        print(f"     Type: {doc_type}, Act: {act}, Section: {section}")
    
    if len(chunks) > 5:
        print(f"\n  ... and {len(chunks) - 5} more")


def print_filter_result(filter_result) -> None:
    """Print graph filter result."""
    print(f"\n  📊 Filter Results:")
    print(f"     Allowed: {filter_result.total_allowed}")
    print(f"     Excluded: {filter_result.total_excluded}")
    
    if filter_result.total_excluded > 0:
        print(f"\n  ❌ Exclusions:")
        if filter_result.overruled_excluded > 0:
            print(f"     • Overruled cases: {filter_result.overruled_excluded}")
        if filter_result.section_mismatch_excluded > 0:
            print(f"     • Section mismatch: {filter_result.section_mismatch_excluded}")
        if filter_result.not_in_graph_excluded > 0:
            print(f"     • Not in graph: {filter_result.not_in_graph_excluded}")
    
    if filter_result.allowed_chunks:
        print(f"\n  ✅ Allowed Semantic IDs:")
        for chunk in filter_result.allowed_chunks[:5]:
            semantic_id = chunk.get("semantic_id", chunk.get("chunk_id", "N/A"))
            print(f"     • {semantic_id}")
        if len(filter_result.allowed_chunks) > 5:
            print(f"     ... and {len(filter_result.allowed_chunks) - 5} more")


def print_final_answer(result: GroundedAnswerResult) -> None:
    """Print final answer."""
    print(f"\n{'═' * 60}")
    print("  FINAL ANSWER")
    print(f"{'═' * 60}")
    
    print(f"\n  📝 Query: {result.query}")
    
    print(f"\n  📊 Statistics:")
    print(f"     Retrieved: {result.retrieved_count}")
    print(f"     Allowed: {result.allowed_chunks_count}")
    print(f"     Excluded: {result.excluded_chunks_count}")
    print(f"     Citations: {len(result.cited_semantic_ids)}")
    
    if result.cited_semantic_ids:
        print(f"\n  📚 Cited Semantic IDs:")
        for semantic_id in result.cited_semantic_ids:
            print(f"     • {semantic_id}")
    
    print(f"\n  ✅ Grounded: {'Yes' if result.grounded else 'No'}")
    print(f"  🤖 Method: {result.generation_method}")
    
    if result.refusal_reason:
        print(f"\n  ⚠️  Refusal Reason:")
        print(f"     {result.refusal_reason}")
    
    print(f"\n  💬 Answer:")
    # Wrap answer text
    answer_lines = result.answer.split('\n')
    for line in answer_lines:
        if line.strip():
            # Wrap long lines
            words = line.split()
            current_line = "     "
            for word in words:
                if len(current_line) + len(word) + 1 > 80:
                    print(current_line)
                    current_line = "     " + word
                else:
                    current_line += " " + word if current_line != "     " else word
            if current_line.strip():
                print(current_line)
        else:
            print()
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate graph-aware grounded answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic query
    python scripts/graph_grounded_answer.py \\
        --query "What is Section 420 IPC?"
    
    # With more chunks
    python scripts/graph_grounded_answer.py \\
        --query "What is the punishment for cheating?" \\
        --top-k 10
    
    # With custom graph
    python scripts/graph_grounded_answer.py \\
        --query "Explain Section 302 IPC" \\
        --graph-path data/graph/legal_graph_v2.pkl
    
    # Verbose mode
    python scripts/graph_grounded_answer.py \\
        --query "What is fraud?" \\
        --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to answer"
    )
    
    # Optional arguments
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph/legal_graph_v2.pkl",
        help="Path to graph pickle file"
    )
    parser.add_argument(
        "--mock-chunks",
        type=int,
        default=5,
        help="Number of mock chunks to generate (default: 5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    print_banner()
    
    print(f"📝 Query: {args.query}")
    print(f"📂 Graph: {args.graph_path}")
    print(f"🔢 Top-K: {args.top_k}")
    
    # Load graph
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        # Try v1 if v2 doesn't exist
        v1_path = graph_path.parent / "legal_graph_v1.pkl"
        if v1_path.exists():
            graph_path = v1_path
            print(f"📂 Using v1 graph: {graph_path}")
        else:
            print(f"❌ Graph not found: {args.graph_path}")
            print("   Run 'python scripts/build_legal_graph.py' first")
            sys.exit(1)
    
    try:
        traverser = LegalGraphTraverser.from_pickle(str(graph_path))
        print(f"✅ Graph loaded: {traverser.graph.number_of_nodes()} nodes, {traverser.graph.number_of_edges()} edges")
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        sys.exit(1)
    
    # Create components
    graph_filter = GraphRAGFilter(traverser)
    
    # Real retriever required - NO MOCKS ALLOWED
    real_retriever = create_real_retriever()
    
    # Create generator with real retriever
    generator = GraphGroundedGenerator(
        graph_filter=graph_filter,
        retriever=real_retriever,
        generator=None,
    )
    
    # Run pipeline with step-by-step output
    print(f"\n{'═' * 60}")
    print("  RUNNING PIPELINE")
    print(f"{'═' * 60}")
    
    # Step A: Retrieve
    print_pipeline_step("A", "Retrieval")
    retrieved_chunks = real_retriever.retrieve(args.query, args.top_k)
    print_retrieved_chunks(retrieved_chunks)
    
    # Step B: Filter
    print_pipeline_step("B", "Graph Filtering")
    filter_result = graph_filter.filter_chunks(args.query, retrieved_chunks)
    print_filter_result(filter_result)
    
    # Steps C-G: Generation
    print_pipeline_step("C-G", "Generation & Validation")
    result = generator.generate_answer(args.query, args.top_k)
    
    # Print final result
    print_final_answer(result)
    
    # Summary
    print(f"{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}\n")
    
    if result.refusal_reason:
        print("  ⚠️  Request was REFUSED")
        print(f"  Reason: {result.refusal_reason}")
    else:
        print("  ✅ Answer generated successfully")
        print(f"  Grounded: {'Yes' if result.grounded else 'No'}")
        print(f"  Citations: {len(result.cited_semantic_ids)}")
    
    print()


if __name__ == "__main__":
    main()
