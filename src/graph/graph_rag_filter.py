"""
Graph-RAG Filter - Phase 3B: Graph-Constrained Retrieval

Filtering layer that uses graph traversal to constrain which RAG chunks
are allowed to pass downstream.

NO retrieval changes, NO graph mutation, NO LLMs.
Pure deterministic filtering based on graph connectivity.

Filtering Rules:
A. If chunk refers to a SECTION: Allow only if section exists in graph
B. If chunk refers to a CASE: Exclude if overruled or not connected
C. If query mentions a SECTION: Only allow chunks linked to that section
D. If multiple sections retrieved: Keep only graph-connected sections
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from .legal_graph_traverser import LegalGraphTraverser, TraversalRelation
from .legal_graph_builder import LegalGraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class GraphFilteredResult:
    """
    Result of graph-based chunk filtering.
    
    Provides full explainability for why chunks were allowed or excluded.
    """
    allowed_chunks: List[Dict[str, Any]] = field(default_factory=list)
    excluded_chunks: List[Dict[str, Any]] = field(default_factory=list)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)  # chunk_id -> reason
    graph_paths_used: List[List[str]] = field(default_factory=list)
    
    # Statistics
    total_input: int = 0
    total_allowed: int = 0
    total_excluded: int = 0
    overruled_excluded: int = 0
    section_mismatch_excluded: int = 0
    not_in_graph_excluded: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_chunks": self.allowed_chunks,
            "excluded_chunks": self.excluded_chunks,
            "exclusion_reasons": self.exclusion_reasons,
            "graph_paths_used": self.graph_paths_used,
            "stats": {
                "total_input": self.total_input,
                "total_allowed": self.total_allowed,
                "total_excluded": self.total_excluded,
                "overruled_excluded": self.overruled_excluded,
                "section_mismatch_excluded": self.section_mismatch_excluded,
                "not_in_graph_excluded": self.not_in_graph_excluded,
            }
        }


class GraphRAGFilter:
    """
    Graph-constrained filtering for RAG chunks.
    
    Uses LegalGraphTraverser (READ-ONLY) to filter chunks based on
    graph connectivity and legal validity.
    
    NO modifications to retrieval, graph, or embeddings.
    """
    
    def __init__(self, traverser: LegalGraphTraverser):
        """
        Initialize filter with a graph traverser.
        
        Args:
            traverser: LegalGraphTraverser instance (READ-ONLY)
        """
        self.traverser = traverser
        
        # Pattern to extract section references from query
        self.section_pattern = re.compile(
            r'(?:Section|Sec\.|S\.)\s*(\d+(?:\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE
        )
        
        # Pattern to extract act names from query
        self.act_pattern = re.compile(
            r'(IPC|CrPC|CPC|Indian\s+Penal\s+Code|'
            r'Code\s+of\s+Criminal\s+Procedure|'
            r'Evidence\s+Act|Contract\s+Act|'
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act)',
            re.IGNORECASE
        )
    
    # ─────────────────────────────────────────────
    # QUERY PARSING
    # ─────────────────────────────────────────────
    
    def extract_sections_from_query(self, query: str) -> List[Tuple[str, str]]:
        """
        Extract (act, section) pairs from query.
        
        Args:
            query: User query text
            
        Returns:
            List of (act_name, section_number) tuples
        """
        results = []
        
        # Find all section references
        section_matches = self.section_pattern.finditer(query)
        
        for match in section_matches:
            section = match.group(1).strip()
            
            # Try to find the act name near this section reference
            start = max(0, match.start() - 100)
            end = min(len(query), match.end() + 100)
            context = query[start:end]
            
            act_match = self.act_pattern.search(context)
            if act_match:
                act = act_match.group(1).strip()
            else:
                # Default to IPC if no act specified
                act = "IPC"
            
            results.append((act, section))
        
        return results
    
    # ─────────────────────────────────────────────
    # CHUNK VALIDATION
    # ─────────────────────────────────────────────
    
    def _get_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """Get chunk identifier."""
        return chunk.get("chunk_id") or chunk.get("id") or str(hash(str(chunk)))
    
    def _get_chunk_section_id(self, chunk: Dict[str, Any]) -> Optional[str]:
        """
        Get the section node ID for a chunk.
        
        Args:
            chunk: Chunk dictionary with metadata
            
        Returns:
            Section node ID or None
        """
        act = chunk.get("act")
        section = chunk.get("section")
        
        if not act or not section:
            return None
        
        # Normalize to graph node ID format
        section_id = LegalGraphBuilder.make_section_id(act, section)
        return section_id
    
    def _get_chunk_case_id(self, chunk: Dict[str, Any]) -> Optional[str]:
        """
        Get the case node ID for a chunk.
        
        Args:
            chunk: Chunk dictionary with metadata
            
        Returns:
            Case node ID or None
        """
        # Try different fields for case identifier
        case_id = (
            chunk.get("case_id") or
            chunk.get("citation") or
            chunk.get("doc_id") if chunk.get("doc_type") == "case_law" else None
        )
        
        if not case_id:
            return None
        
        # Normalize to graph node ID format
        return LegalGraphBuilder.make_case_id(case_id)
    
    def _is_section_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk refers to a section."""
        return chunk.get("doc_type") == "bare_act" and chunk.get("section") is not None
    
    def _is_case_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk refers to a case."""
        return chunk.get("doc_type") == "case_law"
    
    # ─────────────────────────────────────────────
    # FILTERING RULES
    # ─────────────────────────────────────────────
    
    def _check_section_exists(self, chunk: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Rule A: If chunk refers to a SECTION, allow only if section exists in graph.
        
        Returns:
            (should_allow, reason_if_excluded)
        """
        section_id = self._get_chunk_section_id(chunk)
        
        if not section_id:
            # No section reference, can't apply this rule
            return True, None
        
        if self.traverser.node_exists(section_id):
            return True, None
        else:
            return False, f"Section not in graph: {section_id}"
    
    def _check_case_validity(
        self,
        chunk: Dict[str, Any],
        queried_sections: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Rule B: If chunk refers to a CASE:
        - Exclude if case is overruled
        - Exclude if not connected to queried section
        
        Returns:
            (should_allow, reason_if_excluded)
        """
        case_id = self._get_chunk_case_id(chunk)
        
        if not case_id:
            # No case reference, can't apply this rule
            return True, None
        
        # Check if case exists in graph
        if not self.traverser.node_exists(case_id):
            return False, f"Case not in graph: {case_id}"
        
        # Check if case is overruled
        if self.traverser._is_overruled(case_id):
            return False, f"Case is overruled: {case_id}"
        
        # If query mentions specific sections, check connectivity
        if queried_sections:
            # Check if case is connected to any queried section
            connected = False
            for section_id in queried_sections:
                # Get cases for this section
                cases = self.traverser.get_cases_for_section(section_id)
                if case_id in cases:
                    connected = True
                    break
            
            if not connected:
                return False, f"Case not connected to queried sections: {case_id}"
        
        return True, None
    
    def _check_section_connectivity(
        self,
        chunk: Dict[str, Any],
        queried_sections: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Rule C: If query mentions a SECTION, only allow chunks linked to that section
        via APPLIES / INTERPRETS edges.
        
        Returns:
            (should_allow, reason_if_excluded)
        """
        if not queried_sections:
            # No specific sections queried, can't apply this rule
            return True, None
        
        chunk_section_id = self._get_chunk_section_id(chunk)
        
        if not chunk_section_id:
            # Chunk doesn't reference a section, can't apply this rule
            return True, None
        
        # Check if chunk's section matches or is connected to queried sections
        if chunk_section_id in queried_sections:
            # Direct match
            return True, None
        
        # Check if connected via shared cases
        for queried_section in queried_sections:
            result = self.traverser.get_related_sections(queried_section, max_depth=1)
            if chunk_section_id in result.node_ids:
                return True, None
        
        return False, f"Section not connected to queried sections: {chunk_section_id}"
    
    def _check_multi_section_connectivity(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Set[str]:
        """
        Rule D: If multiple sections retrieved, keep only graph-connected sections.
        
        Returns:
            Set of section IDs that are connected
        """
        # Extract all section IDs from chunks
        section_ids = set()
        for chunk in chunks:
            section_id = self._get_chunk_section_id(chunk)
            if section_id:
                section_ids.add(section_id)
        
        if len(section_ids) <= 1:
            # Only one section or no sections, all are "connected"
            return section_ids
        
        # Build connectivity graph
        connected_sections = set()
        section_list = list(section_ids)
        
        # Start with first section
        if section_list:
            connected_sections.add(section_list[0])
            
            # Find sections connected to the first one
            for section_id in section_list[1:]:
                # Check if connected via shared cases or related sections
                result = self.traverser.get_related_sections(section_list[0], max_depth=2)
                if section_id in result.node_ids:
                    connected_sections.add(section_id)
                else:
                    # Check reverse direction
                    result = self.traverser.get_related_sections(section_id, max_depth=2)
                    if section_list[0] in result.node_ids:
                        connected_sections.add(section_id)
        
        return connected_sections
    
    # ─────────────────────────────────────────────
    # MAIN FILTERING METHOD
    # ─────────────────────────────────────────────
    
    def filter_chunks(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> GraphFilteredResult:
        """
        Filter retrieved chunks using graph constraints.
        
        Args:
            query: User query text
            retrieved_chunks: List of chunk dictionaries from retrieval
            
        Returns:
            GraphFilteredResult with allowed/excluded chunks and reasons
        """
        result = GraphFilteredResult()
        result.total_input = len(retrieved_chunks)
        
        # If no graph or no chunks, allow all (fallback mode)
        if self.traverser.graph.number_of_nodes() == 0:
            logger.info("Empty graph, allowing all chunks (fallback mode)")
            result.allowed_chunks = retrieved_chunks
            result.total_allowed = len(retrieved_chunks)
            return result
        
        if not retrieved_chunks:
            return result
        
        # Parse query to extract section references
        queried_sections_raw = self.extract_sections_from_query(query)
        queried_sections = [
            LegalGraphBuilder.make_section_id(act, section)
            for act, section in queried_sections_raw
        ]
        
        logger.debug(f"Queried sections: {queried_sections}")
        
        # Rule D: Check multi-section connectivity first
        connected_sections = self._check_multi_section_connectivity(retrieved_chunks)
        logger.debug(f"Connected sections: {connected_sections}")
        
        # Filter each chunk
        for chunk in retrieved_chunks:
            chunk_id = self._get_chunk_id(chunk)
            should_allow = True
            exclusion_reason = None
            
            # Rule A: Check if section exists in graph
            if self._is_section_chunk(chunk):
                allow, reason = self._check_section_exists(chunk)
                if not allow:
                    should_allow = False
                    exclusion_reason = reason
                    result.not_in_graph_excluded += 1
            
            # Rule B: Check case validity (overruled, connectivity)
            if should_allow and self._is_case_chunk(chunk):
                allow, reason = self._check_case_validity(chunk, queried_sections)
                if not allow:
                    should_allow = False
                    exclusion_reason = reason
                    if "overruled" in reason.lower():
                        result.overruled_excluded += 1
                    else:
                        result.section_mismatch_excluded += 1
            
            # Rule C: Check section connectivity to query
            if should_allow and queried_sections:
                allow, reason = self._check_section_connectivity(chunk, queried_sections)
                if not allow:
                    should_allow = False
                    exclusion_reason = reason
                    result.section_mismatch_excluded += 1
            
            # Rule D: Check if section is in connected set
            if should_allow:
                chunk_section_id = self._get_chunk_section_id(chunk)
                if chunk_section_id and connected_sections:
                    if chunk_section_id not in connected_sections:
                        should_allow = False
                        exclusion_reason = f"Section not in connected set: {chunk_section_id}"
                        result.section_mismatch_excluded += 1
            
            # Record result
            if should_allow:
                result.allowed_chunks.append(chunk)
                result.total_allowed += 1
            else:
                result.excluded_chunks.append(chunk)
                result.exclusion_reasons[chunk_id] = exclusion_reason or "Unknown reason"
                result.total_excluded += 1
        
        # Collect graph paths used (for explainability)
        for section_id in queried_sections:
            if self.traverser.node_exists(section_id):
                # Get applicable cases for this section
                applicable_result = self.traverser.get_applicable_cases(section_id)
                result.graph_paths_used.extend(applicable_result.paths)
        
        logger.info(
            f"Filtered {result.total_input} chunks: "
            f"{result.total_allowed} allowed, {result.total_excluded} excluded"
        )
        
        return result
    
    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def get_filter_stats(self, result: GraphFilteredResult) -> Dict[str, Any]:
        """Get filtering statistics."""
        return {
            "total_input": result.total_input,
            "total_allowed": result.total_allowed,
            "total_excluded": result.total_excluded,
            "allow_rate": result.total_allowed / result.total_input if result.total_input > 0 else 0,
            "exclusion_breakdown": {
                "overruled": result.overruled_excluded,
                "section_mismatch": result.section_mismatch_excluded,
                "not_in_graph": result.not_in_graph_excluded,
            }
        }
    
    def print_filter_result(self, result: GraphFilteredResult) -> None:
        """Print filter result to console."""
        print("\n" + "─" * 60)
        print("GRAPH FILTER RESULT")
        print("─" * 60)
        
        print(f"\n📊 Summary:")
        print(f"   Input chunks: {result.total_input}")
        print(f"   Allowed: {result.total_allowed}")
        print(f"   Excluded: {result.total_excluded}")
        
        if result.total_input > 0:
            allow_rate = (result.total_allowed / result.total_input) * 100
            print(f"   Allow rate: {allow_rate:.1f}%")
        
        if result.total_excluded > 0:
            print(f"\n❌ Exclusion Breakdown:")
            if result.overruled_excluded > 0:
                print(f"   • Overruled cases: {result.overruled_excluded}")
            if result.section_mismatch_excluded > 0:
                print(f"   • Section mismatch: {result.section_mismatch_excluded}")
            if result.not_in_graph_excluded > 0:
                print(f"   • Not in graph: {result.not_in_graph_excluded}")
        
        if result.allowed_chunks:
            print(f"\n✅ Allowed Chunks ({len(result.allowed_chunks)}):")
            for chunk in result.allowed_chunks[:5]:
                chunk_id = self._get_chunk_id(chunk)
                section = chunk.get("section", "N/A")
                act = chunk.get("act", "N/A")
                print(f"   • {chunk_id} - {act} Section {section}")
            if len(result.allowed_chunks) > 5:
                print(f"   ... and {len(result.allowed_chunks) - 5} more")
        
        if result.excluded_chunks:
            print(f"\n❌ Excluded Chunks ({len(result.excluded_chunks)}):")
            for chunk in result.excluded_chunks[:5]:
                chunk_id = self._get_chunk_id(chunk)
                reason = result.exclusion_reasons.get(chunk_id, "Unknown")
                print(f"   • {chunk_id}: {reason}")
            if len(result.excluded_chunks) > 5:
                print(f"   ... and {len(result.excluded_chunks) - 5} more")
        
        print()
