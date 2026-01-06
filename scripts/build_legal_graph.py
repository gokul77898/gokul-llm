#!/usr/bin/env python3
"""
Build Legal Graph CLI

Phase 1: Legal Graph Foundation

Builds a directed graph of legal entities from existing RAG data.
NO LLMs, NO embeddings - pure deterministic graph construction.

Usage:
    python scripts/build_legal_graph.py
    python scripts/build_legal_graph.py --documents-dir data/rag/documents --chunks-dir data/rag/chunks
    python scripts/build_legal_graph.py --validate-only --graph-path data/graph/legal_graph_v1.pkl

Output:
    - data/graph/legal_graph_v1.pkl (pickle)
    - data/graph/legal_graph_v1.json (JSON for inspection)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.legal_graph_builder import (
    LegalGraphBuilder,
    NodeType,
    EdgeType,
    GraphStats,
    ValidationResult,
)


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
║           LEGAL GRAPH BUILDER - Phase 1                       ║
║           Legal Graph Foundation                              ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_stats(stats: GraphStats) -> None:
    """Print graph statistics."""
    print("\n" + "─" * 60)
    print("GRAPH STATISTICS")
    print("─" * 60)
    
    print(f"\n📊 Total Nodes: {stats.total_nodes}")
    print(f"📊 Total Edges: {stats.total_edges}")
    
    print("\n📦 Nodes by Type:")
    for node_type, count in stats.nodes_by_type.items():
        icon = {"act": "📜", "section": "📄", "case": "⚖️"}.get(node_type, "•")
        print(f"   {icon} {node_type.upper()}: {count}")
    
    print("\n🔗 Edges by Type:")
    for edge_type, count in stats.edges_by_type.items():
        print(f"   → {edge_type}: {count}")
    
    if stats.acts:
        print(f"\n📜 Acts Found ({len(stats.acts)}):")
        for act in stats.acts[:15]:
            print(f"   • {act}")
        if len(stats.acts) > 15:
            print(f"   ... and {len(stats.acts) - 15} more")
    
    print(f"\n⏱️  Build Time: {stats.build_time_seconds}s")
    print(f"📅 Build Timestamp: {stats.build_timestamp}")


def print_validation(result: ValidationResult) -> None:
    """Print validation results."""
    print("\n" + "─" * 60)
    print("VALIDATION RESULTS")
    print("─" * 60)
    
    if result.is_valid:
        print("\n✅ Graph is VALID")
    else:
        print("\n❌ Graph has ISSUES")
    
    if result.orphan_sections:
        print(f"\n⚠️  Orphan Sections ({len(result.orphan_sections)}):")
        for section in result.orphan_sections[:5]:
            print(f"   • {section}")
        if len(result.orphan_sections) > 5:
            print(f"   ... and {len(result.orphan_sections) - 5} more")
    
    if result.cases_without_act:
        print(f"\n⚠️  Cases without Act ({len(result.cases_without_act)}):")
        for case in result.cases_without_act[:5]:
            print(f"   • {case}")
        if len(result.cases_without_act) > 5:
            print(f"   ... and {len(result.cases_without_act) - 5} more")
    
    if result.cases_without_section:
        print(f"\nℹ️  Cases without Section ({len(result.cases_without_section)}):")
        print("   (This is informational, not an error)")
    
    if result.errors:
        print("\n❌ Errors:")
        for error in result.errors:
            print(f"   • {error}")
    
    if result.warnings and not result.errors:
        print(f"\n⚠️  Warnings: {len(result.warnings)}")


def verify_deterministic_rebuild(builder: LegalGraphBuilder) -> bool:
    """
    Verify that rebuilding produces the same graph.
    
    Returns:
        True if rebuild is deterministic
    """
    print("\n" + "─" * 60)
    print("DETERMINISTIC REBUILD CHECK")
    print("─" * 60)
    
    # Get current stats
    original_nodes = builder.graph.number_of_nodes()
    original_edges = builder.graph.number_of_edges()
    
    # Create a new builder and rebuild
    rebuild_builder = LegalGraphBuilder(
        documents_dir=str(builder.documents_dir),
        chunks_dir=str(builder.chunks_dir),
        output_dir=str(builder.output_dir),
    )
    rebuild_builder.build()
    
    rebuild_nodes = rebuild_builder.graph.number_of_nodes()
    rebuild_edges = rebuild_builder.graph.number_of_edges()
    
    is_deterministic = (
        original_nodes == rebuild_nodes and
        original_edges == rebuild_edges
    )
    
    if is_deterministic:
        print(f"\n✅ Rebuild is DETERMINISTIC")
        print(f"   Original: {original_nodes} nodes, {original_edges} edges")
        print(f"   Rebuild:  {rebuild_nodes} nodes, {rebuild_edges} edges")
    else:
        print(f"\n❌ Rebuild is NOT deterministic!")
        print(f"   Original: {original_nodes} nodes, {original_edges} edges")
        print(f"   Rebuild:  {rebuild_nodes} nodes, {rebuild_edges} edges")
    
    return is_deterministic


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Legal Graph from RAG data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build graph from default locations
    python scripts/build_legal_graph.py
    
    # Build with custom paths
    python scripts/build_legal_graph.py \\
        --documents-dir data/rag/documents \\
        --chunks-dir data/rag/chunks \\
        --output-dir data/graph
    
    # Validate existing graph
    python scripts/build_legal_graph.py \\
        --validate-only \\
        --graph-path data/graph/legal_graph_v1.pkl
    
    # Verbose output
    python scripts/build_legal_graph.py --verbose
        """
    )
    
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="data/rag/documents",
        help="Path to canonical documents directory"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="data/rag/chunks",
        help="Path to chunks directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/graph",
        help="Path for graph output"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version string for output files"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing graph, don't rebuild"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        help="Path to existing graph pickle (for --validate-only)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step"
    )
    parser.add_argument(
        "--skip-deterministic-check",
        action="store_true",
        help="Skip deterministic rebuild check"
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
    
    logger = logging.getLogger(__name__)
    
    # Validate-only mode
    if args.validate_only:
        if not args.graph_path:
            print("❌ Error: --graph-path required with --validate-only")
            sys.exit(1)
        
        if not Path(args.graph_path).exists():
            print(f"❌ Error: Graph file not found: {args.graph_path}")
            sys.exit(1)
        
        print(f"📂 Loading graph from: {args.graph_path}")
        builder = LegalGraphBuilder.load(args.graph_path)
        
        stats = builder.get_stats()
        print_stats(stats)
        
        result = builder.validate()
        print_validation(result)
        
        sys.exit(0 if result.is_valid else 1)
    
    # Build mode
    print(f"📂 Documents directory: {args.documents_dir}")
    print(f"📂 Chunks directory: {args.chunks_dir}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"📌 Version: {args.version}")
    
    # Check if directories exist
    docs_exist = Path(args.documents_dir).exists()
    chunks_exist = Path(args.chunks_dir).exists()
    
    if not docs_exist and not chunks_exist:
        print("\n⚠️  Warning: Neither documents nor chunks directory exists.")
        print("   Creating empty graph structure...")
        print("\n   To populate the graph, ensure data exists in:")
        print(f"   - {args.documents_dir}")
        print(f"   - {args.chunks_dir}")
    else:
        if docs_exist:
            doc_count = len(list(Path(args.documents_dir).glob("*.json")))
            print(f"   Found {doc_count} document files")
        else:
            print(f"   ⚠️  Documents directory not found")
        
        if chunks_exist:
            chunk_count = len([f for f in Path(args.chunks_dir).glob("*.json") if f.name != "index.json"])
            print(f"   Found {chunk_count} chunk files")
        else:
            print(f"   ⚠️  Chunks directory not found")
    
    # Build graph
    print("\n🔨 Building graph...")
    
    builder = LegalGraphBuilder(
        documents_dir=args.documents_dir,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
    )
    
    stats = builder.build()
    print_stats(stats)
    
    # Validation
    if not args.skip_validation:
        result = builder.validate()
        print_validation(result)
    
    # Deterministic check
    if not args.skip_deterministic_check and stats.total_nodes > 0:
        verify_deterministic_rebuild(builder)
    
    # Save graph
    print("\n" + "─" * 60)
    print("SAVING GRAPH")
    print("─" * 60)
    
    pickle_path, json_path = builder.save(version=args.version)
    
    print(f"\n✅ Graph saved successfully!")
    print(f"   📦 Pickle: {pickle_path}")
    print(f"   📄 JSON:   {json_path}")
    
    # Final summary
    print("\n" + "═" * 60)
    print("BUILD COMPLETE")
    print("═" * 60)
    print(f"\n📊 Summary:")
    print(f"   • Nodes: {stats.total_nodes}")
    print(f"   • Edges: {stats.total_edges}")
    print(f"   • Acts:  {len(stats.acts)}")
    print(f"   • Time:  {stats.build_time_seconds}s")
    
    if stats.total_nodes == 0:
        print("\n⚠️  Graph is empty. To populate:")
        print("   1. Add documents to data/rag/documents/")
        print("   2. Add chunks to data/rag/chunks/")
        print("   3. Re-run this script")
    
    print("\n✨ Ready for Phase 2!")
    print()


if __name__ == "__main__":
    main()
