"""
Precedent Path Extractor - Phase 5A

Extracts structured precedent explanations from grounded generation results.

STRUCTURED EXPLANATIONS ONLY.
NO NATURAL LANGUAGE GENERATION.
NO LLM USAGE.
NO NEW GRAPH TRAVERSAL.

Uses ONLY:
- grounded_result.cited_semantic_ids
- grounded_result.graph_paths_used
- traverser (READ-ONLY for node info)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any, Set

from ..graph.legal_graph_traverser import LegalGraphTraverser
from ..generation.graph_grounded_generator import GroundedAnswerResult

logger = logging.getLogger(__name__)


@dataclass
class PrecedentExplanation:
    """
    Structured precedent explanation.
    
    NO natural language - only structured data.
    """
    cited_semantic_id: str
    node_id: str
    node_type: Literal["ACT", "SECTION", "CASE"]
    authority_level: Literal["statute", "supreme_court", "high_court"]
    relation_chain: List[str] = field(default_factory=list)
    graph_path: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cited_semantic_id": self.cited_semantic_id,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "authority_level": self.authority_level,
            "relation_chain": self.relation_chain,
            "graph_path": self.graph_path,
        }


class PrecedentPathExtractor:
    """
    Extracts structured precedent explanations from grounded results.
    
    NO new graph traversal.
    NO LLM calls.
    NO natural language generation.
    
    Uses ONLY data already in GroundedAnswerResult.
    """
    
    # Authority level priority (higher = more authoritative)
    AUTHORITY_PRIORITY = {
        "statute": 3,
        "supreme_court": 2,
        "high_court": 1,
    }
    
    def __init__(self, traverser: LegalGraphTraverser):
        """
        Initialize extractor.
        
        Args:
            traverser: LegalGraphTraverser (READ-ONLY)
        """
        self.traverser = traverser
    
    # ─────────────────────────────────────────────
    # NODE CLASSIFICATION
    # ─────────────────────────────────────────────
    
    def _classify_node_type(self, node_id: str) -> Literal["ACT", "SECTION", "CASE"]:
        """
        Classify node type from node ID.
        
        Args:
            node_id: Graph node ID
            
        Returns:
            Node type
        """
        if node_id.startswith("ACT::"):
            return "ACT"
        elif node_id.startswith("SECTION::"):
            return "SECTION"
        elif node_id.startswith("CASE::"):
            return "CASE"
        else:
            # Default to CASE for unknown
            logger.warning(f"Unknown node type for {node_id}, defaulting to CASE")
            return "CASE"
    
    def _classify_authority_level(
        self,
        node_id: str,
        node_type: Literal["ACT", "SECTION", "CASE"]
    ) -> Literal["statute", "supreme_court", "high_court"]:
        """
        Classify authority level.
        
        Rules:
        - ACT::* → statute
        - SECTION::* → statute
        - CASE::* Supreme Court → supreme_court
        - CASE::* High Court → high_court
        
        Args:
            node_id: Graph node ID
            node_type: Node type
            
        Returns:
            Authority level
        """
        if node_type in ["ACT", "SECTION"]:
            return "statute"
        
        # For cases, check court in node attributes
        if node_type == "CASE":
            node_info = self.traverser.get_node_info(node_id)
            if node_info:
                court = node_info.get("court", "").lower()
                
                # Check for Supreme Court
                if "supreme" in court or "sc" in court:
                    return "supreme_court"
                
                # Check for High Court
                if "high" in court or "hc" in court:
                    return "high_court"
            
            # Default to high_court for cases without court info
            return "high_court"
        
        # Default
        return "high_court"
    
    # ─────────────────────────────────────────────
    # SEMANTIC ID TO NODE ID MAPPING
    # ─────────────────────────────────────────────
    
    def _map_semantic_id_to_node_id(
        self,
        semantic_id: str,
        graph_paths: List[List[str]]
    ) -> str:
        """
        Map semantic ID to graph node ID.
        
        Uses graph_paths_used to find the node ID.
        
        Args:
            semantic_id: Semantic ID from cited_semantic_ids
            graph_paths: Graph paths from grounded_result
            
        Returns:
            Graph node ID or semantic_id if not found
        """
        # Try to find semantic_id in graph paths
        # This is a heuristic - we look for nodes that might match
        
        # First, check if semantic_id is already a node ID
        if self.traverser.node_exists(semantic_id):
            return semantic_id
        
        # Try to extract from semantic_id pattern
        # Common patterns: "semantic_act_0", "semantic_case_1", "chunk_act_0", etc.
        
        # If semantic_id contains act/section info, try to construct node ID
        if "act" in semantic_id.lower():
            # Try to find ACT or SECTION nodes in paths
            for path in graph_paths:
                for node_id in path:
                    if node_id.startswith("ACT::") or node_id.startswith("SECTION::"):
                        return node_id
        
        if "case" in semantic_id.lower():
            # Try to find CASE nodes in paths
            for path in graph_paths:
                for node_id in path:
                    if node_id.startswith("CASE::"):
                        return node_id
        
        # If we can't map it, return the semantic_id itself
        # This allows the system to still produce output
        logger.debug(f"Could not map semantic_id {semantic_id} to node_id, using as-is")
        return semantic_id
    
    # ─────────────────────────────────────────────
    # PATH EXTRACTION
    # ─────────────────────────────────────────────
    
    def _extract_relation_chain(self, graph_path: List[str]) -> List[str]:
        """
        Extract relation chain from graph path.
        
        Args:
            graph_path: List of node IDs in path
            
        Returns:
            List of relation types between nodes
        """
        if len(graph_path) < 2:
            return []
        
        relations = []
        for i in range(len(graph_path) - 1):
            source = graph_path[i]
            target = graph_path[i + 1]
            
            # Get edge data
            if self.traverser.graph.has_edge(source, target):
                edge_data = self.traverser.graph.get_edge_data(source, target)
                edge_type = edge_data.get("edge_type", "UNKNOWN")
                relations.append(edge_type)
            else:
                relations.append("UNKNOWN")
        
        return relations
    
    def _find_paths_for_node(
        self,
        node_id: str,
        all_paths: List[List[str]]
    ) -> List[List[str]]:
        """
        Find all paths that contain the given node.
        
        Args:
            node_id: Node ID to search for
            all_paths: All graph paths
            
        Returns:
            List of paths containing the node
        """
        matching_paths = []
        for path in all_paths:
            if node_id in path:
                matching_paths.append(path)
        
        return matching_paths
    
    # ─────────────────────────────────────────────
    # MAIN EXTRACTION METHOD
    # ─────────────────────────────────────────────
    
    def extract(
        self,
        grounded_result: GroundedAnswerResult
    ) -> List[PrecedentExplanation]:
        """
        Extract structured precedent explanations.
        
        Uses ONLY:
        - grounded_result.cited_semantic_ids
        - grounded_result.graph_paths_used
        
        NO new graph traversal.
        NO LLM calls.
        
        Args:
            grounded_result: Result from graph-grounded generation
            
        Returns:
            List of structured precedent explanations
        """
        logger.info(f"Extracting precedents from {len(grounded_result.cited_semantic_ids)} citations")
        
        # Handle empty input
        if not grounded_result.cited_semantic_ids:
            logger.info("No cited semantic IDs, returning empty list")
            return []
        
        explanations = []
        seen_node_ids: Set[str] = set()  # For deduplication
        
        # Process each cited semantic ID
        for semantic_id in grounded_result.cited_semantic_ids:
            # Map semantic ID to node ID
            node_id = self._map_semantic_id_to_node_id(
                semantic_id,
                grounded_result.graph_paths_used
            )
            
            # Skip if already processed (deduplication)
            if node_id in seen_node_ids:
                logger.debug(f"Skipping duplicate node_id: {node_id}")
                continue
            
            seen_node_ids.add(node_id)
            
            # Classify node
            node_type = self._classify_node_type(node_id)
            authority_level = self._classify_authority_level(node_id, node_type)
            
            # Find paths containing this node
            relevant_paths = self._find_paths_for_node(
                node_id,
                grounded_result.graph_paths_used
            )
            
            # Use first path if available, otherwise empty
            if relevant_paths:
                graph_path = relevant_paths[0]
                relation_chain = self._extract_relation_chain(graph_path)
            else:
                graph_path = [node_id]
                relation_chain = []
            
            # Create explanation
            explanation = PrecedentExplanation(
                cited_semantic_id=semantic_id,
                node_id=node_id,
                node_type=node_type,
                authority_level=authority_level,
                relation_chain=relation_chain,
                graph_path=graph_path,
            )
            
            explanations.append(explanation)
        
        # Sort by authority level (statute > supreme_court > high_court)
        explanations.sort(
            key=lambda x: self.AUTHORITY_PRIORITY.get(x.authority_level, 0),
            reverse=True
        )
        
        logger.info(f"Extracted {len(explanations)} precedent explanations")
        
        return explanations
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_explanation_stats(
        self,
        explanations: List[PrecedentExplanation]
    ) -> Dict[str, Any]:
        """Get statistics about explanations."""
        if not explanations:
            return {
                "total": 0,
                "by_type": {},
                "by_authority": {},
            }
        
        by_type = {}
        by_authority = {}
        
        for exp in explanations:
            # Count by type
            by_type[exp.node_type] = by_type.get(exp.node_type, 0) + 1
            
            # Count by authority
            by_authority[exp.authority_level] = by_authority.get(exp.authority_level, 0) + 1
        
        return {
            "total": len(explanations),
            "by_type": by_type,
            "by_authority": by_authority,
        }
    
    def print_explanations(
        self,
        explanations: List[PrecedentExplanation]
    ) -> None:
        """Print explanations to console."""
        print("\n" + "═" * 60)
        print("PRECEDENT EXPLANATIONS")
        print("═" * 60)
        
        if not explanations:
            print("\n  No precedent explanations")
            return
        
        stats = self.get_explanation_stats(explanations)
        
        print(f"\n📊 Statistics:")
        print(f"   Total: {stats['total']}")
        print(f"   By Type: {stats['by_type']}")
        print(f"   By Authority: {stats['by_authority']}")
        
        print(f"\n📚 Explanations:\n")
        
        for i, exp in enumerate(explanations, 1):
            print(f"  {i}. {exp.node_id}")
            print(f"     Semantic ID: {exp.cited_semantic_id}")
            print(f"     Type: {exp.node_type}")
            print(f"     Authority: {exp.authority_level}")
            
            if exp.relation_chain:
                print(f"     Relations: {' → '.join(exp.relation_chain)}")
            
            if exp.graph_path and len(exp.graph_path) > 1:
                print(f"     Path: {' → '.join(exp.graph_path[:3])}")
                if len(exp.graph_path) > 3:
                    print(f"           ... ({len(exp.graph_path)} nodes total)")
            
            print()
