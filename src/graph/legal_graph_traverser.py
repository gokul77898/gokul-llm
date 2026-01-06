"""
Legal Graph Traverser - Phase 3A: Graph Traversal API

READ-ONLY graph traversal utilities for the legal knowledge graph.
NO LLMs, NO embeddings, NO modifications to graph.

Provides deterministic traversal methods:
- get_sections_for_act(act_id)
- get_cases_for_section(section_id, relation_types)
- get_precedent_chain(case_id, depth)
- get_applicable_cases(section_id) - excludes overruled
- get_related_sections(section_id) - via shared cases
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

import networkx as nx

from .legal_graph_builder import LegalGraphBuilder, NodeType, EdgeType
from .legal_edge_extractor import RelationType

logger = logging.getLogger(__name__)


class TraversalRelation(str, Enum):
    """Relations used in traversal queries."""
    # Phase 1 relations
    HAS_SECTION = "HAS_SECTION"
    BELONGS_TO_ACT = "BELONGS_TO_ACT"
    MENTIONS_SECTION = "MENTIONS_SECTION"
    
    # Phase 2 relations
    INTERPRETS_SECTION = "INTERPRETS_SECTION"
    APPLIES_SECTION = "APPLIES_SECTION"
    DISTINGUISHES_SECTION = "DISTINGUISHES_SECTION"
    CITES_CASE = "CITES_CASE"
    OVERRULES_CASE = "OVERRULES_CASE"


@dataclass
class TraversalResult:
    """Result of a graph traversal operation."""
    node_ids: List[str] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    depth_reached: int = 0
    cycles_avoided: int = 0
    overruled_excluded: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": self.node_ids,
            "paths": self.paths,
            "depth_reached": self.depth_reached,
            "cycles_avoided": self.cycles_avoided,
            "overruled_excluded": self.overruled_excluded,
        }


class LegalGraphTraverser:
    """
    READ-ONLY graph traversal for legal knowledge graphs.
    
    All methods:
    - Return node IDs only (no text)
    - Are deterministic (sorted outputs)
    - Avoid cycles
    - Support depth limits
    
    NO modifications to the graph.
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize traverser with a NetworkX graph.
        
        Args:
            graph: NetworkX directed graph from LegalGraphBuilder
        """
        self.graph = graph
        
        # Cache overruled cases for efficient lookup
        self._overruled_cases: Optional[Set[str]] = None
    
    @classmethod
    def from_pickle(cls, pickle_path: str) -> 'LegalGraphTraverser':
        """
        Load graph from pickle and create traverser.
        
        Args:
            pickle_path: Path to graph pickle file
            
        Returns:
            LegalGraphTraverser instance
        """
        builder = LegalGraphBuilder.load(pickle_path)
        return cls(builder.graph)
    
    # ─────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────
    
    def _get_node_type(self, node_id: str) -> Optional[str]:
        """Get the type of a node."""
        if node_id not in self.graph:
            return None
        return self.graph.nodes[node_id].get("node_type")
    
    def _is_act(self, node_id: str) -> bool:
        """Check if node is an Act."""
        return self._get_node_type(node_id) == NodeType.ACT.value
    
    def _is_section(self, node_id: str) -> bool:
        """Check if node is a Section."""
        return self._get_node_type(node_id) == NodeType.SECTION.value
    
    def _is_case(self, node_id: str) -> bool:
        """Check if node is a Case."""
        return self._get_node_type(node_id) == NodeType.CASE.value
    
    def _get_edge_type(self, source: str, target: str) -> Optional[str]:
        """Get the edge type between two nodes."""
        if not self.graph.has_edge(source, target):
            return None
        return self.graph.edges[source, target].get("edge_type")
    
    def _normalize_node_id(self, node_id: str, node_type: str = None) -> str:
        """
        Normalize a node ID to canonical format.
        
        Handles cases like:
        - "IPC_420" -> "SECTION::IPC::420"
        - "420" with act="IPC" -> "SECTION::IPC::420"
        """
        # Already in canonical format
        if node_id.startswith("ACT::") or node_id.startswith("SECTION::") or node_id.startswith("CASE::"):
            return node_id
        
        # Try to find matching node
        for n in self.graph.nodes():
            # Check if node_id is a suffix match
            if n.endswith(f"::{node_id}") or n.endswith(f"::{node_id.upper()}"):
                return n
            # Check normalized form
            if node_id.upper().replace("_", "::") in n:
                return n
        
        # Return as-is if no match found
        return node_id
    
    def _get_overruled_cases(self) -> Set[str]:
        """
        Get set of all overruled case IDs.
        
        A case is overruled if another case has an OVERRULES_CASE edge to it.
        """
        if self._overruled_cases is not None:
            return self._overruled_cases
        
        overruled = set()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == TraversalRelation.OVERRULES_CASE.value:
                overruled.add(v)  # v is the overruled case
        
        self._overruled_cases = overruled
        return overruled
    
    def _is_overruled(self, case_id: str) -> bool:
        """Check if a case has been overruled."""
        return case_id in self._get_overruled_cases()
    
    # ─────────────────────────────────────────────
    # TRAVERSAL METHODS
    # ─────────────────────────────────────────────
    
    def get_sections_for_act(self, act_id: str) -> List[str]:
        """
        Get all sections belonging to an act.
        
        Traverses: Act --HAS_SECTION--> Section
        
        Args:
            act_id: Act node ID (e.g., "ACT::IPC" or "IPC")
            
        Returns:
            Sorted list of section node IDs
        """
        # Normalize act_id
        if not act_id.startswith("ACT::"):
            act_id = f"ACT::{LegalGraphBuilder.normalize_act_name(act_id)}"
        
        if act_id not in self.graph:
            logger.debug(f"Act not found: {act_id}")
            return []
        
        sections = []
        
        for successor in self.graph.successors(act_id):
            edge_type = self._get_edge_type(act_id, successor)
            if edge_type == TraversalRelation.HAS_SECTION.value:
                if self._is_section(successor):
                    sections.append(successor)
        
        return sorted(sections)
    
    def get_cases_for_section(
        self,
        section_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get all cases related to a section.
        
        Traverses: Case --[relation]--> Section (reverse direction)
        
        Args:
            section_id: Section node ID
            relation_types: Optional list of relation types to filter
                           (default: all case-section relations)
            
        Returns:
            Sorted list of case node IDs
        """
        # Normalize section_id
        section_id = self._normalize_node_id(section_id)
        
        if section_id not in self.graph:
            logger.debug(f"Section not found: {section_id}")
            return []
        
        # Default relation types
        if relation_types is None:
            relation_types = [
                TraversalRelation.MENTIONS_SECTION.value,
                TraversalRelation.INTERPRETS_SECTION.value,
                TraversalRelation.APPLIES_SECTION.value,
                TraversalRelation.DISTINGUISHES_SECTION.value,
            ]
        else:
            # Normalize relation types
            relation_types = [r.upper() if isinstance(r, str) else r.value for r in relation_types]
        
        cases = []
        
        # Look at predecessors (cases point TO sections)
        for predecessor in self.graph.predecessors(section_id):
            edge_type = self._get_edge_type(predecessor, section_id)
            if edge_type in relation_types:
                if self._is_case(predecessor):
                    cases.append(predecessor)
        
        return sorted(cases)
    
    def get_precedent_chain(
        self,
        case_id: str,
        depth: int = 2
    ) -> TraversalResult:
        """
        Get chain of precedent cases (cases cited by this case).
        
        Traverses: Case --CITES_CASE--> Case (recursive)
        
        Args:
            case_id: Starting case node ID
            depth: Maximum traversal depth (default: 2)
            
        Returns:
            TraversalResult with cited cases and paths
        """
        result = TraversalResult()
        
        # Normalize case_id
        if not case_id.startswith("CASE::"):
            case_id = f"CASE::{case_id.upper().replace(' ', '_')}"
        
        if case_id not in self.graph:
            logger.debug(f"Case not found: {case_id}")
            return result
        
        visited: Set[str] = set()
        all_cases: Set[str] = set()
        
        def traverse(current: str, current_depth: int, path: List[str]):
            if current_depth > depth:
                return
            
            if current in visited:
                result.cycles_avoided += 1
                return
            
            visited.add(current)
            
            # Find cited cases
            for successor in self.graph.successors(current):
                edge_type = self._get_edge_type(current, successor)
                if edge_type == TraversalRelation.CITES_CASE.value:
                    if self._is_case(successor):
                        all_cases.add(successor)
                        new_path = path + [successor]
                        result.paths.append(new_path)
                        result.depth_reached = max(result.depth_reached, current_depth)
                        
                        # Recurse
                        traverse(successor, current_depth + 1, new_path)
        
        traverse(case_id, 1, [case_id])
        
        result.node_ids = sorted(all_cases)
        return result
    
    def get_applicable_cases(self, section_id: str) -> TraversalResult:
        """
        Get cases that interpret or apply a section, EXCLUDING overruled cases.
        
        Includes:
        - INTERPRETS_SECTION
        - APPLIES_SECTION
        
        Excludes:
        - Cases that have been OVERRULED
        
        Args:
            section_id: Section node ID
            
        Returns:
            TraversalResult with applicable cases
        """
        result = TraversalResult()
        
        # Normalize section_id
        section_id = self._normalize_node_id(section_id)
        
        if section_id not in self.graph:
            logger.debug(f"Section not found: {section_id}")
            return result
        
        # Get overruled cases
        overruled = self._get_overruled_cases()
        
        applicable_relations = [
            TraversalRelation.INTERPRETS_SECTION.value,
            TraversalRelation.APPLIES_SECTION.value,
        ]
        
        cases = []
        
        for predecessor in self.graph.predecessors(section_id):
            edge_type = self._get_edge_type(predecessor, section_id)
            if edge_type in applicable_relations:
                if self._is_case(predecessor):
                    if predecessor in overruled:
                        result.overruled_excluded += 1
                        logger.debug(f"Excluding overruled case: {predecessor}")
                    else:
                        cases.append(predecessor)
                        result.paths.append([predecessor, section_id])
        
        result.node_ids = sorted(cases)
        return result
    
    def get_related_sections(
        self,
        section_id: str,
        max_depth: int = 2
    ) -> TraversalResult:
        """
        Get sections related to a given section via shared cases.
        
        Traversal pattern:
        Section <--[case relation]-- Case --[case relation]--> Other Section
        
        Args:
            section_id: Starting section node ID
            max_depth: Maximum traversal depth (default: 2)
            
        Returns:
            TraversalResult with related sections
        """
        result = TraversalResult()
        
        # Normalize section_id
        section_id = self._normalize_node_id(section_id)
        
        if section_id not in self.graph:
            logger.debug(f"Section not found: {section_id}")
            return result
        
        visited_sections: Set[str] = {section_id}
        related_sections: Set[str] = set()
        
        case_relations = [
            TraversalRelation.MENTIONS_SECTION.value,
            TraversalRelation.INTERPRETS_SECTION.value,
            TraversalRelation.APPLIES_SECTION.value,
            TraversalRelation.DISTINGUISHES_SECTION.value,
        ]
        
        def find_related(current_section: str, current_depth: int, path: List[str]):
            if current_depth > max_depth:
                return
            
            # Find cases that reference this section
            for predecessor in self.graph.predecessors(current_section):
                edge_type = self._get_edge_type(predecessor, current_section)
                if edge_type in case_relations and self._is_case(predecessor):
                    # Find other sections this case references
                    for successor in self.graph.successors(predecessor):
                        succ_edge_type = self._get_edge_type(predecessor, successor)
                        if succ_edge_type in case_relations and self._is_section(successor):
                            if successor not in visited_sections:
                                visited_sections.add(successor)
                                related_sections.add(successor)
                                new_path = path + [predecessor, successor]
                                result.paths.append(new_path)
                                result.depth_reached = max(result.depth_reached, current_depth)
                                
                                # Recurse to find more related sections
                                if current_depth < max_depth:
                                    find_related(successor, current_depth + 1, new_path)
                            elif successor in related_sections:
                                result.cycles_avoided += 1
        
        find_related(section_id, 1, [section_id])
        
        result.node_ids = sorted(related_sections)
        return result
    
    # ─────────────────────────────────────────────
    # ADDITIONAL UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_act_for_section(self, section_id: str) -> Optional[str]:
        """
        Get the act that a section belongs to.
        
        Traverses: Act --HAS_SECTION--> Section (reverse)
        
        Args:
            section_id: Section node ID
            
        Returns:
            Act node ID or None
        """
        section_id = self._normalize_node_id(section_id)
        
        if section_id not in self.graph:
            return None
        
        for predecessor in self.graph.predecessors(section_id):
            edge_type = self._get_edge_type(predecessor, section_id)
            if edge_type == TraversalRelation.HAS_SECTION.value:
                if self._is_act(predecessor):
                    return predecessor
        
        return None
    
    def get_cases_citing(self, case_id: str) -> List[str]:
        """
        Get cases that cite a given case.
        
        Traverses: Other Case --CITES_CASE--> This Case (reverse)
        
        Args:
            case_id: Case node ID
            
        Returns:
            Sorted list of citing case node IDs
        """
        if not case_id.startswith("CASE::"):
            case_id = f"CASE::{case_id.upper().replace(' ', '_')}"
        
        if case_id not in self.graph:
            return []
        
        citing_cases = []
        
        for predecessor in self.graph.predecessors(case_id):
            edge_type = self._get_edge_type(predecessor, case_id)
            if edge_type == TraversalRelation.CITES_CASE.value:
                if self._is_case(predecessor):
                    citing_cases.append(predecessor)
        
        return sorted(citing_cases)
    
    def get_overruling_case(self, case_id: str) -> Optional[str]:
        """
        Get the case that overruled a given case (if any).
        
        Traverses: Overruling Case --OVERRULES_CASE--> This Case (reverse)
        
        Args:
            case_id: Case node ID
            
        Returns:
            Overruling case node ID or None
        """
        if not case_id.startswith("CASE::"):
            case_id = f"CASE::{case_id.upper().replace(' ', '_')}"
        
        if case_id not in self.graph:
            return None
        
        for predecessor in self.graph.predecessors(case_id):
            edge_type = self._get_edge_type(predecessor, case_id)
            if edge_type == TraversalRelation.OVERRULES_CASE.value:
                return predecessor
        
        return None
    
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        normalized = self._normalize_node_id(node_id)
        return normalized in self.graph
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node attributes dict or None
        """
        normalized = self._normalize_node_id(node_id)
        
        if normalized not in self.graph:
            return None
        
        return dict(self.graph.nodes[normalized])
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get basic graph statistics."""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": {},
            "edges_by_type": {},
        }
        
        # Count nodes by type
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            stats["nodes_by_type"][node_type] = stats["nodes_by_type"].get(node_type, 0) + 1
        
        # Count edges by type
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("edge_type", "unknown")
            stats["edges_by_type"][edge_type] = stats["edges_by_type"].get(edge_type, 0) + 1
        
        return stats
    
    # ─────────────────────────────────────────────
    # PRINTING METHODS
    # ─────────────────────────────────────────────
    
    def print_traversal_result(
        self,
        result: TraversalResult,
        title: str = "Traversal Result"
    ) -> None:
        """Print a traversal result to console."""
        print(f"\n{'─' * 50}")
        print(f"  {title}")
        print(f"{'─' * 50}")
        
        print(f"\n  Found: {len(result.node_ids)} nodes")
        
        if result.node_ids:
            print("\n  Nodes:")
            for node_id in result.node_ids:
                print(f"    • {node_id}")
        
        if result.paths:
            print(f"\n  Paths ({len(result.paths)}):")
            for i, path in enumerate(result.paths[:10], 1):  # Show first 10
                print(f"    {i}. {' → '.join(path)}")
            if len(result.paths) > 10:
                print(f"    ... and {len(result.paths) - 10} more")
        
        if result.depth_reached > 0:
            print(f"\n  Max depth reached: {result.depth_reached}")
        
        if result.cycles_avoided > 0:
            print(f"  Cycles avoided: {result.cycles_avoided}")
        
        if result.overruled_excluded > 0:
            print(f"  Overruled cases excluded: {result.overruled_excluded}")
        
        print()
    
    def print_sections(self, act_id: str) -> None:
        """Print sections for an act."""
        sections = self.get_sections_for_act(act_id)
        
        print(f"\n{'─' * 50}")
        print(f"  Sections for: {act_id}")
        print(f"{'─' * 50}")
        
        if sections:
            print(f"\n  Found: {len(sections)} sections")
            for section in sections:
                print(f"    • {section}")
        else:
            print("\n  No sections found")
        
        print()
    
    def print_cases(
        self,
        section_id: str,
        relation_types: Optional[List[str]] = None
    ) -> None:
        """Print cases for a section."""
        cases = self.get_cases_for_section(section_id, relation_types)
        
        print(f"\n{'─' * 50}")
        print(f"  Cases for: {section_id}")
        if relation_types:
            print(f"  Relations: {', '.join(relation_types)}")
        print(f"{'─' * 50}")
        
        if cases:
            print(f"\n  Found: {len(cases)} cases")
            for case in cases:
                overruled = " (OVERRULED)" if self._is_overruled(case) else ""
                print(f"    • {case}{overruled}")
        else:
            print("\n  No cases found")
        
        print()
