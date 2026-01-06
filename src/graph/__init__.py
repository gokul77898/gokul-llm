"""
Legal Graph Module - Phases 1, 2, 3A

This module provides graph-based representation of legal documents
using NetworkX for deterministic graph building and traversal.

Phase 1: Graph Foundation (Nodes + Basic Edges)
Phase 2: Legal Edge Extraction (Rule-based relationship extraction)
Phase 3A: Graph Traversal API (READ-ONLY traversal utilities)

NO LLMs, NO embeddings - pure data structure and pattern matching.
"""

from .legal_graph_builder import (
    LegalGraphBuilder,
    NodeType,
    EdgeType,
    GraphStats,
    ValidationResult,
)

from .legal_edge_extractor import (
    LegalEdgeExtractor,
    RelationType,
    ExtractedEdge,
    ExtractionStats,
    LegalPatterns,
)

from .legal_graph_traverser import (
    LegalGraphTraverser,
    TraversalRelation,
    TraversalResult,
)

from .graph_rag_filter import (
    GraphRAGFilter,
    GraphFilteredResult,
)

__all__ = [
    # Phase 1
    "LegalGraphBuilder",
    "NodeType",
    "EdgeType",
    "GraphStats",
    "ValidationResult",
    # Phase 2
    "LegalEdgeExtractor",
    "RelationType",
    "ExtractedEdge",
    "ExtractionStats",
    "LegalPatterns",
    # Phase 3A
    "LegalGraphTraverser",
    "TraversalRelation",
    "TraversalResult",
    # Phase 3B
    "GraphRAGFilter",
    "GraphFilteredResult",
]
