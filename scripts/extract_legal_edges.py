#!/usr/bin/env python3
"""
Extract Legal Edges CLI

Phase 2: Legal Edge Extraction

Extracts legal relationships from document text using rule-based patterns.
NO LLMs, NO embeddings, NO ML inference - pure pattern matching.

Usage:
    python scripts/extract_legal_edges.py
    python scripts/extract_legal_edges.py --graph-path data/graph/legal_graph_v1.pkl
    python scripts/extract_legal_edges.py --chunks-dir data/rag/chunks --version v2

Output:
    - data/graph/legal_graph_v2.pkl (pickle)
    - data/graph/legal_graph_v2.json (JSON for inspection)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.legal_edge_extractor import (
    LegalEdgeExtractor,
    RelationType,
    ExtractionStats,
)
from src.graph.legal_graph_builder import LegalGraphBuilder


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
║           LEGAL EDGE EXTRACTOR - Phase 2                      ║
║           Rule-Based Relationship Extraction                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_stats(stats: ExtractionStats) -> None:
    """Print extraction statistics."""
    print("\n" + "─" * 60)
    print("EXTRACTION STATISTICS")
    print("─" * 60)
    
    print(f"\n📊 Processing:")
    print(f"   Chunks processed: {stats.total_chunks_processed}")
    print(f"   Sentences processed: {stats.total_sentences_processed}")
    print(f"   Chunks with edges: {stats.chunks_with_edges}")
    
    print(f"\n🔗 Edges Extracted: {stats.total_edges_extracted}")
    print(f"   Duplicates skipped: {stats.duplicate_edges_skipped}")
    
    if stats.edges_by_type:
        print("\n📈 Edges by Type:")
        
        # Case → Section relations
        section_types = [
            RelationType.INTERPRETS_SECTION.value,
            RelationType.APPLIES_SECTION.value,
            RelationType.DISTINGUISHES_SECTION.value,
        ]
        section_total = sum(stats.edges_by_type.get(t, 0) for t in section_types)
        
        if section_total > 0:
            print("\n   Case → Section:")
            for edge_type in section_types:
                count = stats.edges_by_type.get(edge_type, 0)
                if count > 0:
                    print(f"     → {edge_type}: {count}")
        
        # Case → Case relations
        case_types = [
            RelationType.CITES_CASE.value,
            RelationType.OVERRULES_CASE.value,
        ]
        case_total = sum(stats.edges_by_type.get(t, 0) for t in case_types)
        
        if case_total > 0:
            print("\n   Case → Case:")
            for edge_type in case_types:
                count = stats.edges_by_type.get(edge_type, 0)
                if count > 0:
                    print(f"     → {edge_type}: {count}")
    
    print(f"\n⏱️  Extraction Time: {stats.extraction_time_seconds}s")
    print(f"📅 Timestamp: {stats.extraction_timestamp}")


def print_graph_stats(builder: LegalGraphBuilder) -> None:
    """Print updated graph statistics."""
    stats = builder.get_stats()
    
    print("\n" + "─" * 60)
    print("UPDATED GRAPH STATISTICS")
    print("─" * 60)
    
    print(f"\n📊 Total Nodes: {stats.total_nodes}")
    print(f"📊 Total Edges: {stats.total_edges}")
    
    print("\n📦 Nodes by Type:")
    for node_type, count in stats.nodes_by_type.items():
        icon = {"act": "📜", "section": "📄", "case": "⚖️"}.get(node_type, "•")
        print(f"   {icon} {node_type.upper()}: {count}")
    
    print("\n🔗 Edges by Type:")
    for edge_type, count in sorted(stats.edges_by_type.items()):
        print(f"   → {edge_type}: {count}")


def print_validation_summary(extractor: LegalEdgeExtractor) -> None:
    """Print validation summary."""
    print("\n" + "─" * 60)
    print("VALIDATION SUMMARY")
    print("─" * 60)
    
    result = extractor.builder.validate()
    
    if result.is_valid:
        print("\n✅ Graph is VALID")
    else:
        print("\n⚠️  Graph has issues")
    
    if result.orphan_sections:
        print(f"\n   Orphan sections: {len(result.orphan_sections)}")
    
    if result.cases_without_act:
        print(f"   Cases without act: {len(result.cases_without_act)}")
    
    if result.cases_without_section:
        print(f"   Cases without section: {len(result.cases_without_section)}")
    
    # Source coverage
    edges = extractor.get_extracted_edges()
    if edges:
        unique_chunks = set(e.source_chunk_id for e in edges)
        print(f"\n📊 Source Coverage:")
        print(f"   Unique source chunks: {len(unique_chunks)}")
        print(f"   Total edges with provenance: {len(edges)}")


def verify_deterministic_rebuild(
    extractor: LegalEdgeExtractor,
    documents_dir: str,
    chunks_dir: str,
    graph_path: str,
    output_dir: str,
) -> bool:
    """
    Verify that re-extraction produces the same edges.
    
    Returns:
        True if deterministic
    """
    print("\n" + "─" * 60)
    print("DETERMINISTIC REBUILD CHECK")
    print("─" * 60)
    
    # Get current counts
    original_edges = extractor._stats.total_edges_extracted
    original_graph_edges = extractor.builder.graph.number_of_edges()
    
    # Create new extractor and re-extract
    rebuild_extractor = LegalEdgeExtractor(
        documents_dir=documents_dir,
        chunks_dir=chunks_dir,
        graph_path=graph_path,
        output_dir=output_dir,
    )
    rebuild_extractor.extract()
    
    rebuild_edges = rebuild_extractor._stats.total_edges_extracted
    
    is_deterministic = original_edges == rebuild_edges
    
    if is_deterministic:
        print(f"\n✅ Extraction is DETERMINISTIC")
        print(f"   Original: {original_edges} edges extracted")
        print(f"   Rebuild:  {rebuild_edges} edges extracted")
    else:
        print(f"\n❌ Extraction is NOT deterministic!")
        print(f"   Original: {original_edges} edges extracted")
        print(f"   Rebuild:  {rebuild_edges} edges extracted")
    
    return is_deterministic


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract legal relationships from document text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract edges using default paths
    python scripts/extract_legal_edges.py
    
    # Extract with custom paths
    python scripts/extract_legal_edges.py \\
        --graph-path data/graph/legal_graph_v1.pkl \\
        --chunks-dir data/rag/chunks \\
        --version v2
    
    # Skip deterministic check (faster)
    python scripts/extract_legal_edges.py --skip-deterministic-check
    
    # Verbose output
    python scripts/extract_legal_edges.py --verbose
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
        "--graph-path",
        type=str,
        default="data/graph/legal_graph_v1.pkl",
        help="Path to existing graph (Phase 1)"
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
        default="v2",
        help="Version string for output files"
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
    
    # Print configuration
    print(f"📂 Documents directory: {args.documents_dir}")
    print(f"📂 Chunks directory: {args.chunks_dir}")
    print(f"📂 Input graph: {args.graph_path}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"📌 Version: {args.version}")
    
    # Check paths
    graph_exists = Path(args.graph_path).exists()
    docs_exist = Path(args.documents_dir).exists()
    chunks_exist = Path(args.chunks_dir).exists()
    
    if graph_exists:
        print(f"\n✅ Found existing graph at {args.graph_path}")
    else:
        print(f"\n⚠️  Graph not found at {args.graph_path}")
        print("   Will create new graph")
    
    if not docs_exist and not chunks_exist:
        print("\n⚠️  Warning: Neither documents nor chunks directory exists.")
        print("   No edges will be extracted.")
    else:
        if docs_exist:
            doc_count = len(list(Path(args.documents_dir).glob("*.json")))
            print(f"   Found {doc_count} document files")
        if chunks_exist:
            chunk_count = len([
                f for f in Path(args.chunks_dir).glob("*.json")
                if f.name != "index.json"
            ])
            print(f"   Found {chunk_count} chunk files")
    
    # Create extractor
    print("\n🔨 Extracting legal relationships...")
    
    extractor = LegalEdgeExtractor(
        documents_dir=args.documents_dir,
        chunks_dir=args.chunks_dir,
        graph_path=args.graph_path,
        output_dir=args.output_dir,
    )
    
    # Run extraction
    stats = extractor.extract()
    print_stats(stats)
    
    # Print graph stats
    print_graph_stats(extractor.builder)
    
    # Validation
    if not args.skip_validation:
        print_validation_summary(extractor)
    
    # Deterministic check
    if not args.skip_deterministic_check and stats.total_edges_extracted > 0:
        verify_deterministic_rebuild(
            extractor,
            args.documents_dir,
            args.chunks_dir,
            args.graph_path,
            args.output_dir,
        )
    
    # Save graph
    print("\n" + "─" * 60)
    print("SAVING GRAPH")
    print("─" * 60)
    
    pickle_path, json_path = extractor.save(version=args.version)
    
    print(f"\n✅ Graph saved successfully!")
    print(f"   📦 Pickle: {pickle_path}")
    print(f"   📄 JSON:   {json_path}")
    
    # Final summary
    print("\n" + "═" * 60)
    print("EXTRACTION COMPLETE")
    print("═" * 60)
    
    graph_stats = extractor.builder.get_stats()
    print(f"\n📊 Summary:")
    print(f"   • Edges extracted: {stats.total_edges_extracted}")
    print(f"   • Graph nodes: {graph_stats.total_nodes}")
    print(f"   • Graph edges: {graph_stats.total_edges}")
    print(f"   • Time: {stats.extraction_time_seconds}s")
    
    if stats.total_edges_extracted == 0:
        print("\n⚠️  No edges extracted. To populate:")
        print("   1. Add case_law documents to data/rag/documents/")
        print("   2. Add case_law chunks to data/rag/chunks/")
        print("   3. Re-run this script")
    
    print("\n✨ Ready for Phase 3 (Graph Traversal)!")
    print()


if __name__ == "__main__":
    main()
