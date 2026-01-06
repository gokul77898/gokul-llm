#!/usr/bin/env python3
"""
Graph Filter Debug CLI

Phase 3B: Graph-Constrained Retrieval

Debug tool to visualize how graph filtering affects retrieved chunks.

Usage:
    python scripts/graph_filter_debug.py \
        --query "Section 420 IPC" \
        --chunks-file test_chunks.json

    python scripts/graph_filter_debug.py \
        --query "What is the punishment for cheating?" \
        --mock-chunks 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.graph_rag_filter import GraphRAGFilter, GraphFilteredResult
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
║           GRAPH FILTER DEBUG - Phase 3B                       ║
║           Graph-Constrained Retrieval                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def create_mock_chunks(count: int) -> list:
    """Create mock chunks for testing."""
    chunks = []
    
    # Mock bare_act chunks
    for i in range(count // 2):
        chunks.append({
            "chunk_id": f"chunk_act_{i}",
            "doc_type": "bare_act",
            "act": "IPC",
            "section": str(420 + i),
            "text": f"Mock section {420 + i} text...",
        })
    
    # Mock case_law chunks
    for i in range(count // 2):
        chunks.append({
            "chunk_id": f"chunk_case_{i}",
            "doc_type": "case_law",
            "act": "IPC",
            "section": "420",
            "case_id": f"case_{i}",
            "citation": f"2020 SCC {100 + i}",
            "text": f"Mock case {i} text...",
        })
    
    return chunks


def load_chunks_from_file(file_path: str) -> list:
    """Load chunks from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "chunks" in data:
            return data["chunks"]
        elif "results" in data:
            return data["results"]
    
    return []


def print_chunks(chunks: list, title: str, max_display: int = 10) -> None:
    """Print chunk information."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    
    if not chunks:
        print("\n  (No chunks)")
        return
    
    print(f"\n  Total: {len(chunks)} chunks\n")
    
    for i, chunk in enumerate(chunks[:max_display], 1):
        chunk_id = chunk.get("chunk_id") or chunk.get("id", f"chunk_{i}")
        doc_type = chunk.get("doc_type", "unknown")
        act = chunk.get("act", "N/A")
        section = chunk.get("section", "N/A")
        
        print(f"  {i}. {chunk_id}")
        print(f"     Type: {doc_type}")
        print(f"     Act: {act}, Section: {section}")
        
        if doc_type == "case_law":
            citation = chunk.get("citation", chunk.get("case_id", "N/A"))
            print(f"     Citation: {citation}")
        
        text = chunk.get("text", "")
        if text:
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"     Text: {preview}")
        
        print()
    
    if len(chunks) > max_display:
        print(f"  ... and {len(chunks) - max_display} more\n")


def print_graph_traversal(
    filter_obj: GraphRAGFilter,
    query: str
) -> None:
    """Print graph traversal results for the query."""
    print(f"\n{'─' * 60}")
    print("  GRAPH TRAVERSAL RESULTS")
    print(f"{'─' * 60}")
    
    # Extract sections from query
    sections = filter_obj.extract_sections_from_query(query)
    
    if not sections:
        print("\n  No sections found in query")
        return
    
    print(f"\n  Sections in query: {len(sections)}\n")
    
    for act, section in sections:
        section_id = f"SECTION::{act.upper()}::{section}"
        
        print(f"  📄 {act} Section {section}")
        print(f"     Node ID: {section_id}")
        
        # Check if exists
        exists = filter_obj.traverser.node_exists(section_id)
        print(f"     Exists in graph: {'✅ Yes' if exists else '❌ No'}")
        
        if exists:
            # Get applicable cases
            result = filter_obj.traverser.get_applicable_cases(section_id)
            print(f"     Applicable cases: {len(result.node_ids)}")
            
            if result.node_ids:
                for case_id in result.node_ids[:3]:
                    print(f"       • {case_id}")
                if len(result.node_ids) > 3:
                    print(f"       ... and {len(result.node_ids) - 3} more")
            
            if result.overruled_excluded > 0:
                print(f"     Overruled excluded: {result.overruled_excluded}")
        
        print()


def print_filter_summary(result: GraphFilteredResult) -> None:
    """Print detailed filter summary."""
    print(f"\n{'─' * 60}")
    print("  FILTERING SUMMARY")
    print(f"{'─' * 60}")
    
    print(f"\n  📊 Statistics:")
    print(f"     Input chunks: {result.total_input}")
    print(f"     Allowed: {result.total_allowed} ({result.total_allowed/result.total_input*100:.1f}%)" if result.total_input > 0 else "     Allowed: 0")
    print(f"     Excluded: {result.total_excluded} ({result.total_excluded/result.total_input*100:.1f}%)" if result.total_input > 0 else "     Excluded: 0")
    
    if result.total_excluded > 0:
        print(f"\n  ❌ Exclusion Breakdown:")
        if result.overruled_excluded > 0:
            print(f"     • Overruled cases: {result.overruled_excluded}")
        if result.section_mismatch_excluded > 0:
            print(f"     • Section mismatch: {result.section_mismatch_excluded}")
        if result.not_in_graph_excluded > 0:
            print(f"     • Not in graph: {result.not_in_graph_excluded}")
    
    if result.exclusion_reasons:
        print(f"\n  📝 Exclusion Reasons:")
        for chunk_id, reason in list(result.exclusion_reasons.items())[:5]:
            print(f"     • {chunk_id}:")
            print(f"       {reason}")
        if len(result.exclusion_reasons) > 5:
            print(f"     ... and {len(result.exclusion_reasons) - 5} more")
    
    if result.graph_paths_used:
        print(f"\n  🔗 Graph Paths Used: {len(result.graph_paths_used)}")
        for i, path in enumerate(result.graph_paths_used[:3], 1):
            print(f"     {i}. {' → '.join(path)}")
        if len(result.graph_paths_used) > 3:
            print(f"     ... and {len(result.graph_paths_used) - 3} more")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Debug graph-constrained retrieval filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With mock chunks
    python scripts/graph_filter_debug.py \\
        --query "Section 420 IPC" \\
        --mock-chunks 10
    
    # With chunks from file
    python scripts/graph_filter_debug.py \\
        --query "What is the punishment for cheating?" \\
        --chunks-file test_chunks.json
    
    # With custom graph
    python scripts/graph_filter_debug.py \\
        --query "Section 302 IPC" \\
        --graph-path data/graph/legal_graph_v2.pkl \\
        --mock-chunks 5
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text to test"
    )
    
    # Chunk input
    parser.add_argument(
        "--chunks-file",
        type=str,
        help="Path to JSON file with chunks"
    )
    parser.add_argument(
        "--mock-chunks",
        type=int,
        help="Generate N mock chunks for testing"
    )
    
    # Graph input
    parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph/legal_graph_v2.pkl",
        help="Path to graph pickle file"
    )
    
    # Options
    parser.add_argument(
        "--max-display",
        type=int,
        default=10,
        help="Maximum chunks to display (default: 10)"
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
    
    # Load chunks
    if args.chunks_file:
        print(f"📂 Chunks: {args.chunks_file}")
        try:
            chunks = load_chunks_from_file(args.chunks_file)
            print(f"✅ Loaded {len(chunks)} chunks from file")
        except Exception as e:
            print(f"❌ Failed to load chunks: {e}")
            sys.exit(1)
    elif args.mock_chunks:
        print(f"🎭 Generating {args.mock_chunks} mock chunks")
        chunks = create_mock_chunks(args.mock_chunks)
    else:
        print("❌ Error: Must specify --chunks-file or --mock-chunks")
        sys.exit(1)
    
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
    
    # Create filter
    filter_obj = GraphRAGFilter(traverser)
    
    # Display input chunks
    print_chunks(chunks, "INPUT CHUNKS (Retrieved)", args.max_display)
    
    # Display graph traversal
    print_graph_traversal(filter_obj, args.query)
    
    # Run filtering
    print(f"\n{'═' * 60}")
    print("  RUNNING GRAPH FILTER")
    print(f"{'═' * 60}\n")
    
    result = filter_obj.filter_chunks(args.query, chunks)
    
    # Display results
    print_filter_summary(result)
    
    # Display allowed chunks
    print_chunks(result.allowed_chunks, "ALLOWED CHUNKS (After Filtering)", args.max_display)
    
    # Display excluded chunks
    if result.excluded_chunks:
        print_chunks(result.excluded_chunks, "EXCLUDED CHUNKS", args.max_display)
    
    # Final summary
    print(f"\n{'═' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'═' * 60}\n")
    
    print(f"  Query: {args.query}")
    print(f"  Input: {result.total_input} chunks")
    print(f"  Output: {result.total_allowed} chunks")
    print(f"  Filtered out: {result.total_excluded} chunks")
    
    if result.total_input > 0:
        retention_rate = (result.total_allowed / result.total_input) * 100
        print(f"  Retention rate: {retention_rate:.1f}%")
    
    print()


if __name__ == "__main__":
    main()
