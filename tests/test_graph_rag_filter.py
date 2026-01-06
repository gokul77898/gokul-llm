"""
Tests for Graph-RAG Filter - Phase 3B

Tests filtering logic, overrule exclusion, section connectivity,
and deterministic ordering.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

from src.graph.graph_rag_filter import GraphRAGFilter, GraphFilteredResult
from src.graph.legal_graph_traverser import LegalGraphTraverser, TraversalRelation
from src.graph.legal_graph_builder import LegalGraphBuilder


class TestGraphRAGFilterSetup(unittest.TestCase):
    """Test filter initialization."""
    
    def test_init_with_traverser(self):
        """Test initialization with traverser."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        filter_obj = GraphRAGFilter(traverser)
        
        self.assertIsNotNone(filter_obj.traverser)
        self.assertEqual(filter_obj.traverser, traverser)


class TestQueryParsing(unittest.TestCase):
    """Test query parsing for section extraction."""
    
    def setUp(self):
        """Set up filter."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        self.filter = GraphRAGFilter(traverser)
    
    def test_extract_section_basic(self):
        """Test basic section extraction."""
        query = "What is Section 420 IPC?"
        sections = self.filter.extract_sections_from_query(query)
        
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0], ("IPC", "420"))
    
    def test_extract_section_with_subsection(self):
        """Test section with subsection."""
        query = "Explain Section 302(1) of IPC"
        sections = self.filter.extract_sections_from_query(query)
        
        self.assertEqual(len(sections), 1)
        act, section = sections[0]
        self.assertEqual(act, "IPC")
        self.assertIn("302", section)
    
    def test_extract_multiple_sections(self):
        """Test multiple sections in query."""
        query = "Compare Section 420 and Section 421 of IPC"
        sections = self.filter.extract_sections_from_query(query)
        
        self.assertEqual(len(sections), 2)
    
    def test_extract_section_different_act(self):
        """Test section with different act."""
        query = "What is Section 154 CrPC?"
        sections = self.filter.extract_sections_from_query(query)
        
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "CrPC")
    
    def test_no_section_in_query(self):
        """Test query without section."""
        query = "What is the punishment for cheating?"
        sections = self.filter.extract_sections_from_query(query)
        
        self.assertEqual(len(sections), 0)


class TestOverruleExclusion(unittest.TestCase):
    """Test overruled case exclusion."""
    
    def setUp(self):
        """Set up graph with overruled case."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add section
        self.builder.add_has_section_edge("IPC", "420")
        
        # Add cases
        self.builder.add_case_node("case_valid")
        self.builder.add_case_node("case_overruled")
        self.builder.add_case_node("case_overruler")
        
        # Both cases apply section
        self.builder.graph.add_edge(
            "CASE::CASE_VALID", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULED", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        # Overrule edge
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULER", "CASE::CASE_OVERRULED",
            edge_type=TraversalRelation.OVERRULES_CASE.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_overruled_case_excluded(self):
        """Test that overruled cases are excluded."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "case_law",
                "case_id": "case_valid",
                "act": "IPC",
                "section": "420",
            },
            {
                "chunk_id": "chunk2",
                "doc_type": "case_law",
                "case_id": "case_overruled",
                "act": "IPC",
                "section": "420",
            },
        ]
        
        result = self.filter.filter_chunks("Section 420 IPC", chunks)
        
        # Only valid case should be allowed
        self.assertEqual(result.total_allowed, 1)
        self.assertEqual(result.total_excluded, 1)
        self.assertEqual(result.overruled_excluded, 1)
        
        # Check allowed chunk
        self.assertEqual(result.allowed_chunks[0]["case_id"], "case_valid")
        
        # Check exclusion reason
        self.assertIn("overruled", result.exclusion_reasons["chunk2"].lower())


class TestSectionExistence(unittest.TestCase):
    """Test section existence checking."""
    
    def setUp(self):
        """Set up graph with sections."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add only section 420
        self.builder.add_has_section_edge("IPC", "420")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_section_exists_allowed(self):
        """Test that chunks with existing sections are allowed."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Section 420 text",
            }
        ]
        
        result = self.filter.filter_chunks("Section 420 IPC", chunks)
        
        self.assertEqual(result.total_allowed, 1)
        self.assertEqual(result.total_excluded, 0)
    
    def test_section_not_exists_excluded(self):
        """Test that chunks with non-existent sections are excluded."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "999",  # Doesn't exist in graph
                "text": "Section 999 text",
            }
        ]
        
        result = self.filter.filter_chunks("Section 999 IPC", chunks)
        
        self.assertEqual(result.total_allowed, 0)
        self.assertEqual(result.total_excluded, 1)
        self.assertEqual(result.not_in_graph_excluded, 1)


class TestSectionConnectivity(unittest.TestCase):
    """Test section connectivity filtering."""
    
    def setUp(self):
        """Set up graph with connected sections."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add sections
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "421")
        self.builder.add_has_section_edge("IPC", "302")
        
        # Add case that connects 420 and 421
        self.builder.add_case_node("case1")
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::421",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        # 302 is not connected to 420
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_connected_section_allowed(self):
        """Test that connected sections are allowed."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
            },
            {
                "chunk_id": "chunk2",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "421",
            },
        ]
        
        result = self.filter.filter_chunks("Section 420 IPC", chunks)
        
        # Both should be allowed (connected via case1)
        self.assertEqual(result.total_allowed, 2)


class TestNoGraphFallback(unittest.TestCase):
    """Test fallback behavior when graph is empty."""
    
    def test_empty_graph_allows_all(self):
        """Test that empty graph allows all chunks (fallback mode)."""
        # Empty graph
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        filter_obj = GraphRAGFilter(traverser)
        
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
            },
            {
                "chunk_id": "chunk2",
                "doc_type": "case_law",
                "case_id": "case1",
            },
        ]
        
        result = filter_obj.filter_chunks("Section 420 IPC", chunks)
        
        # All chunks should be allowed
        self.assertEqual(result.total_allowed, 2)
        self.assertEqual(result.total_excluded, 0)


class TestDeterministicOrdering(unittest.TestCase):
    """Test that filtering is deterministic."""
    
    def setUp(self):
        """Set up graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "421")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_deterministic_filtering(self):
        """Test that filtering produces same results."""
        chunks = [
            {"chunk_id": "chunk1", "doc_type": "bare_act", "act": "IPC", "section": "420"},
            {"chunk_id": "chunk2", "doc_type": "bare_act", "act": "IPC", "section": "421"},
            {"chunk_id": "chunk3", "doc_type": "bare_act", "act": "IPC", "section": "999"},
        ]
        
        result1 = self.filter.filter_chunks("Section 420 IPC", chunks)
        result2 = self.filter.filter_chunks("Section 420 IPC", chunks)
        
        # Should produce same results
        self.assertEqual(result1.total_allowed, result2.total_allowed)
        self.assertEqual(result1.total_excluded, result2.total_excluded)
        
        # Allowed chunks should be in same order
        allowed_ids1 = [c["chunk_id"] for c in result1.allowed_chunks]
        allowed_ids2 = [c["chunk_id"] for c in result2.allowed_chunks]
        self.assertEqual(allowed_ids1, allowed_ids2)


class TestCaseConnectivity(unittest.TestCase):
    """Test case connectivity to queried sections."""
    
    def setUp(self):
        """Set up graph with cases and sections."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add sections
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "302")
        
        # Add cases
        self.builder.add_case_node("case_connected")
        self.builder.add_case_node("case_not_connected")
        
        # case_connected applies section 420
        self.builder.graph.add_edge(
            "CASE::CASE_CONNECTED", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        # case_not_connected applies section 302 (different section)
        self.builder.graph.add_edge(
            "CASE::CASE_NOT_CONNECTED", "SECTION::IPC::302",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_connected_case_allowed(self):
        """Test that cases connected to queried section are allowed."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "case_law",
                "case_id": "case_connected",
                "act": "IPC",
                "section": "420",
            }
        ]
        
        result = self.filter.filter_chunks("Section 420 IPC", chunks)
        
        self.assertEqual(result.total_allowed, 1)
    
    def test_not_connected_case_excluded(self):
        """Test that cases not connected to queried section are excluded."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "case_law",
                "case_id": "case_not_connected",
                "act": "IPC",
                "section": "302",
            }
        ]
        
        result = self.filter.filter_chunks("Section 420 IPC", chunks)
        
        # Should be excluded because not connected to Section 420
        self.assertEqual(result.total_allowed, 0)
        self.assertEqual(result.total_excluded, 1)


class TestMixedChunks(unittest.TestCase):
    """Test filtering with mixed chunk types."""
    
    def setUp(self):
        """Set up graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add section
        self.builder.add_has_section_edge("IPC", "420")
        
        # Add valid case
        self.builder.add_case_node("case_valid")
        self.builder.graph.add_edge(
            "CASE::CASE_VALID", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        # Add overruled case
        self.builder.add_case_node("case_overruled")
        self.builder.add_case_node("case_overruler")
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULED", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULER", "CASE::CASE_OVERRULED",
            edge_type=TraversalRelation.OVERRULES_CASE.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_mixed_chunks_filtering(self):
        """Test filtering with mix of bare_act and case_law chunks."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
            },
            {
                "chunk_id": "chunk2",
                "doc_type": "case_law",
                "case_id": "case_valid",
                "act": "IPC",
                "section": "420",
            },
            {
                "chunk_id": "chunk3",
                "doc_type": "case_law",
                "case_id": "case_overruled",
                "act": "IPC",
                "section": "420",
            },
        ]
        
        result = self.filter.filter_chunks("What is cheating?", chunks)  # No section in query
        
        # Should allow: bare_act 420, case_valid
        # Should exclude: case_overruled (overruled)
        self.assertEqual(result.total_allowed, 2)
        self.assertEqual(result.total_excluded, 1)
        self.assertEqual(result.overruled_excluded, 1)
    
    def test_not_in_graph_excluded(self):
        """Test that chunks with sections not in graph are excluded."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "999",  # Doesn't exist
            },
        ]
        
        result = self.filter.filter_chunks("What is section 999?", chunks)
        
        self.assertEqual(result.total_allowed, 0)
        self.assertEqual(result.total_excluded, 1)
        self.assertEqual(result.not_in_graph_excluded, 1)


class TestFilterStats(unittest.TestCase):
    """Test filter statistics."""
    
    def test_get_filter_stats(self):
        """Test statistics calculation."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        filter_obj = GraphRAGFilter(traverser)
        
        result = GraphFilteredResult()
        result.total_input = 10
        result.total_allowed = 7
        result.total_excluded = 3
        result.overruled_excluded = 1
        result.section_mismatch_excluded = 2
        
        stats = filter_obj.get_filter_stats(result)
        
        self.assertEqual(stats["total_input"], 10)
        self.assertEqual(stats["total_allowed"], 7)
        self.assertEqual(stats["total_excluded"], 3)
        self.assertAlmostEqual(stats["allow_rate"], 0.7)
        self.assertEqual(stats["exclusion_breakdown"]["overruled"], 1)
        self.assertEqual(stats["exclusion_breakdown"]["section_mismatch"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
