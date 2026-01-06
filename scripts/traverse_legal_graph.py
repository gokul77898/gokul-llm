#!/usr/bin/env python3
"""
Traverse Legal Graph CLI

Phase 3A: Graph Traversal API

READ-ONLY graph traversal utilities for the legal knowledge graph.
NO LLMs, NO embeddings, NO modifications to graph.

Usage:
    python scripts/traverse_legal_graph.py --section IPC::420 --relations applies interprets
    python scripts/traverse_legal_graph.py --act IPC --list-sections
    python scripts/traverse_legal_graph.py --case "2020_SCC_123" --precedents --depth 3
    python scripts/traverse_legal_graph.py --section IPC::302 --related-sections

Output:
    - Console-print traversal paths
    - NO JSON generation
    - NO persistence changes
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.legal_graph_traverser import (
    LegalGraphTraverser,
    TraversalRelation,
    TraversalResult,
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
║           LEGAL GRAPH TRAVERSER - Phase 3A                    ║
║           READ-ONLY Graph Traversal API                       ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_graph_stats(traverser: LegalGraphTraverser) -> None:
    """Print graph statistics."""
    stats = traverser.get_graph_stats()
    
    print("─" * 50)
    print("  GRAPH STATISTICS")
    print("─" * 50)
    
    print(f"\n  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']}")
    
    if stats['nodes_by_type']:
        print("\n  Nodes by Type:")
        for node_type, count in sorted(stats['nodes_by_type'].items()):
            icon = {"act": "📜", "section": "📄", "case": "⚖️"}.get(node_type, "•")
            print(f"    {icon} {node_type}: {count}")
    
    if stats['edges_by_type']:
        print("\n  Edges by Type:")
        for edge_type, count in sorted(stats['edges_by_type'].items()):
            print(f"    → {edge_type}: {count}")
    
    print()


def handle_list_sections(traverser: LegalGraphTraverser, act_id: str) -> None:
    """Handle --list-sections command."""
    print(f"\n📜 Sections for Act: {act_id}")
    print("─" * 50)
    
    sections = traverser.get_sections_for_act(act_id)
    
    if sections:
        print(f"\n  Found {len(sections)} sections:\n")
        for section in sections:
            print(f"    • {section}")
    else:
        print("\n  No sections found.")
        print(f"  (Act '{act_id}' may not exist in the graph)")
    
    print()


def handle_cases_for_section(
    traverser: LegalGraphTraverser,
    section_id: str,
    relations: list
) -> None:
    """Handle --section with --relations command."""
    # Parse relation types
    relation_map = {
        "applies": TraversalRelation.APPLIES_SECTION.value,
        "interprets": TraversalRelation.INTERPRETS_SECTION.value,
        "distinguishes": TraversalRelation.DISTINGUISHES_SECTION.value,
        "mentions": TraversalRelation.MENTIONS_SECTION.value,
    }
    
    if relations:
        relation_types = []
        for r in relations:
            r_lower = r.lower()
            if r_lower in relation_map:
                relation_types.append(relation_map[r_lower])
            elif r.upper() in [t.value for t in TraversalRelation]:
                relation_types.append(r.upper())
            else:
                print(f"  ⚠️  Unknown relation: {r}")
    else:
        relation_types = None  # All relations
    
    print(f"\n⚖️  Cases for Section: {section_id}")
    if relation_types:
        print(f"   Relations: {', '.join(relation_types)}")
    print("─" * 50)
    
    cases = traverser.get_cases_for_section(section_id, relation_types)
    
    if cases:
        print(f"\n  Found {len(cases)} cases:\n")
        for case in cases:
            overruled = " (OVERRULED)" if traverser._is_overruled(case) else ""
            print(f"    • {case}{overruled}")
    else:
        print("\n  No cases found.")
    
    print()


def handle_applicable_cases(traverser: LegalGraphTraverser, section_id: str) -> None:
    """Handle --applicable-cases command."""
    print(f"\n✅ Applicable Cases for Section: {section_id}")
    print("   (Excludes overruled cases)")
    print("─" * 50)
    
    result = traverser.get_applicable_cases(section_id)
    
    if result.node_ids:
        print(f"\n  Found {len(result.node_ids)} applicable cases:\n")
        for case in result.node_ids:
            print(f"    • {case}")
        
        if result.overruled_excluded > 0:
            print(f"\n  ⚠️  Excluded {result.overruled_excluded} overruled case(s)")
    else:
        print("\n  No applicable cases found.")
    
    print()


def handle_precedent_chain(
    traverser: LegalGraphTraverser,
    case_id: str,
    depth: int
) -> None:
    """Handle --precedents command."""
    print(f"\n📚 Precedent Chain for Case: {case_id}")
    print(f"   Depth: {depth}")
    print("─" * 50)
    
    result = traverser.get_precedent_chain(case_id, depth)
    
    if result.node_ids:
        print(f"\n  Found {len(result.node_ids)} cited cases:\n")
        for case in result.node_ids:
            overruled = " (OVERRULED)" if traverser._is_overruled(case) else ""
            print(f"    • {case}{overruled}")
        
        if result.paths:
            print(f"\n  Citation Paths ({len(result.paths)}):\n")
            for i, path in enumerate(result.paths[:10], 1):
                print(f"    {i}. {' → '.join(path)}")
            if len(result.paths) > 10:
                print(f"    ... and {len(result.paths) - 10} more")
        
        print(f"\n  Max depth reached: {result.depth_reached}")
        
        if result.cycles_avoided > 0:
            print(f"  Cycles avoided: {result.cycles_avoided}")
    else:
        print("\n  No precedents found.")
    
    print()


def handle_related_sections(
    traverser: LegalGraphTraverser,
    section_id: str,
    depth: int
) -> None:
    """Handle --related-sections command."""
    print(f"\n🔗 Related Sections for: {section_id}")
    print(f"   (via shared cases, depth: {depth})")
    print("─" * 50)
    
    result = traverser.get_related_sections(section_id, depth)
    
    if result.node_ids:
        print(f"\n  Found {len(result.node_ids)} related sections:\n")
        for section in result.node_ids:
            print(f"    • {section}")
        
        if result.paths:
            print(f"\n  Connection Paths ({len(result.paths)}):\n")
            for i, path in enumerate(result.paths[:10], 1):
                print(f"    {i}. {' → '.join(path)}")
            if len(result.paths) > 10:
                print(f"    ... and {len(result.paths) - 10} more")
        
        if result.cycles_avoided > 0:
            print(f"\n  Cycles avoided: {result.cycles_avoided}")
    else:
        print("\n  No related sections found.")
    
    print()


def handle_node_info(traverser: LegalGraphTraverser, node_id: str) -> None:
    """Handle --info command."""
    print(f"\nℹ️  Node Info: {node_id}")
    print("─" * 50)
    
    info = traverser.get_node_info(node_id)
    
    if info:
        print(f"\n  Node Type: {info.get('node_type', 'unknown')}")
        print("\n  Attributes:")
        for key, value in sorted(info.items()):
            if key != "node_type":
                print(f"    • {key}: {value}")
        
        # Show connections
        normalized = traverser._normalize_node_id(node_id)
        
        # Outgoing edges
        out_edges = list(traverser.graph.successors(normalized))
        if out_edges:
            print(f"\n  Outgoing edges ({len(out_edges)}):")
            for target in out_edges[:5]:
                edge_type = traverser._get_edge_type(normalized, target)
                print(f"    → {edge_type} → {target}")
            if len(out_edges) > 5:
                print(f"    ... and {len(out_edges) - 5} more")
        
        # Incoming edges
        in_edges = list(traverser.graph.predecessors(normalized))
        if in_edges:
            print(f"\n  Incoming edges ({len(in_edges)}):")
            for source in in_edges[:5]:
                edge_type = traverser._get_edge_type(source, normalized)
                print(f"    ← {edge_type} ← {source}")
            if len(in_edges) > 5:
                print(f"    ... and {len(in_edges) - 5} more")
    else:
        print(f"\n  Node not found: {node_id}")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Traverse legal knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List sections for an act
    python scripts/traverse_legal_graph.py --act IPC --list-sections
    
    # Get cases for a section with specific relations
    python scripts/traverse_legal_graph.py \\
        --section IPC::420 \\
        --relations applies interprets
    
    # Get applicable cases (excludes overruled)
    python scripts/traverse_legal_graph.py \\
        --section IPC::302 \\
        --applicable-cases
    
    # Get precedent chain for a case
    python scripts/traverse_legal_graph.py \\
        --case "2020_SCC_123" \\
        --precedents \\
        --depth 3
    
    # Get related sections via shared cases
    python scripts/traverse_legal_graph.py \\
        --section IPC::420 \\
        --related-sections
    
    # Get node info
    python scripts/traverse_legal_graph.py --info "ACT::IPC"
    
    # Show graph stats
    python scripts/traverse_legal_graph.py --stats
        """
    )
    
    # Graph input
    parser.add_argument(
        "--graph-path",
        type=str,
        default="data/graph/legal_graph_v2.pkl",
        help="Path to graph pickle file"
    )
    
    # Query targets
    parser.add_argument(
        "--act",
        type=str,
        help="Act ID to query (e.g., IPC, CrPC)"
    )
    parser.add_argument(
        "--section",
        type=str,
        help="Section ID to query (e.g., IPC::420, SECTION::IPC::420)"
    )
    parser.add_argument(
        "--case",
        type=str,
        help="Case ID to query"
    )
    
    # Query types
    parser.add_argument(
        "--list-sections",
        action="store_true",
        help="List sections for an act (requires --act)"
    )
    parser.add_argument(
        "--relations",
        nargs="+",
        help="Relation types to filter (applies, interprets, distinguishes, mentions)"
    )
    parser.add_argument(
        "--applicable-cases",
        action="store_true",
        help="Get applicable cases excluding overruled (requires --section)"
    )
    parser.add_argument(
        "--precedents",
        action="store_true",
        help="Get precedent chain (requires --case)"
    )
    parser.add_argument(
        "--related-sections",
        action="store_true",
        help="Get related sections via shared cases (requires --section)"
    )
    parser.add_argument(
        "--info",
        type=str,
        help="Get info about a specific node"
    )
    
    # Options
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Maximum traversal depth (default: 2)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics"
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
    
    # Check graph exists
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        # Try v1 if v2 doesn't exist
        v1_path = graph_path.parent / "legal_graph_v1.pkl"
        if v1_path.exists():
            graph_path = v1_path
            print(f"📂 Using graph: {graph_path}")
        else:
            print(f"❌ Graph not found: {args.graph_path}")
            print("   Run 'python scripts/build_legal_graph.py' first")
            sys.exit(1)
    else:
        print(f"📂 Using graph: {graph_path}")
    
    # Load graph
    try:
        traverser = LegalGraphTraverser.from_pickle(str(graph_path))
        print(f"✅ Graph loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        sys.exit(1)
    
    # Handle commands
    if args.stats:
        print_graph_stats(traverser)
        return
    
    if args.info:
        handle_node_info(traverser, args.info)
        return
    
    if args.act and args.list_sections:
        handle_list_sections(traverser, args.act)
        return
    
    if args.section:
        if args.applicable_cases:
            handle_applicable_cases(traverser, args.section)
        elif args.related_sections:
            handle_related_sections(traverser, args.section, args.depth)
        else:
            handle_cases_for_section(traverser, args.section, args.relations)
        return
    
    if args.case and args.precedents:
        handle_precedent_chain(traverser, args.case, args.depth)
        return
    
    # No specific command - show help
    if not any([args.act, args.section, args.case, args.info, args.stats]):
        print("No query specified. Use --help for usage information.\n")
        print_graph_stats(traverser)
        print("Available commands:")
        print("  --act IPC --list-sections        List sections for an act")
        print("  --section IPC::420               Get cases for a section")
        print("  --section IPC::420 --applicable-cases")
        print("                                   Get applicable cases (excludes overruled)")
        print("  --section IPC::420 --related-sections")
        print("                                   Get related sections via shared cases")
        print("  --case CASE_ID --precedents      Get precedent chain")
        print("  --info NODE_ID                   Get node info")
        print("  --stats                          Show graph statistics")
        print()


if __name__ == "__main__":
    main()
