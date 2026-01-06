"""
Tests for Precedent Path Extractor - Phase 5A

Tests deterministic output, no new graph traversal,
no missing cited sources, and constraint compliance.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

from src.explanation.precedent_extractor import (
    PrecedentPathExtractor,
    PrecedentExplanation,
)
from src.generation.graph_grounded_generator import GroundedAnswerResult
from src.graph.legal_graph_traverser import LegalGraphTraverser, TraversalRelation
from src.graph.legal_graph_builder import LegalGraphBuilder


class TestPrecedentExtractorSetup(unittest.TestCase):
    """Test extractor initialization."""
    
    def test_init_with_traverser(self):
        """Test initialization with traverser."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        extractor = PrecedentPathExtractor(traverser)
        
        self.assertIsNotNone(extractor.traverser)
        self.assertEqual(extractor.traverser, traverser)


class TestEmptyInput(unittest.TestCase):
    """Test empty input handling."""
    
    def setUp(self):
        """Set up extractor."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        self.extractor = PrecedentPathExtractor(traverser)
    
    def test_empty_cited_ids(self):
        """Test that empty cited_semantic_ids returns empty list."""
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=[],
            graph_paths_used=[],
        )
        
        explanations = self.extractor.extract(result)
        
        self.assertEqual(len(explanations), 0)
    
    def test_empty_graph_paths(self):
        """Test with empty graph paths."""
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["sem1"],
            graph_paths_used=[],
        )
        
        explanations = self.extractor.extract(result)
        
        # Should still produce explanation, just without path info
        self.assertEqual(len(explanations), 1)


class TestNodeClassification(unittest.TestCase):
    """Test node type and authority classification."""
    
    def setUp(self):
        """Set up extractor with graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add nodes
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_case_node("case1")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.extractor = PrecedentPathExtractor(self.traverser)
    
    def test_classify_act_node(self):
        """Test ACT node classification."""
        node_type = self.extractor._classify_node_type("ACT::IPC")
        self.assertEqual(node_type, "ACT")
        
        authority = self.extractor._classify_authority_level("ACT::IPC", node_type)
        self.assertEqual(authority, "statute")
    
    def test_classify_section_node(self):
        """Test SECTION node classification."""
        node_type = self.extractor._classify_node_type("SECTION::IPC::420")
        self.assertEqual(node_type, "SECTION")
        
        authority = self.extractor._classify_authority_level("SECTION::IPC::420", node_type)
        self.assertEqual(authority, "statute")
    
    def test_classify_case_node(self):
        """Test CASE node classification."""
        node_type = self.extractor._classify_node_type("CASE::CASE1")
        self.assertEqual(node_type, "CASE")
        
        # Default authority for cases without court info
        authority = self.extractor._classify_authority_level("CASE::CASE1", node_type)
        self.assertIn(authority, ["supreme_court", "high_court"])


class TestDeduplication(unittest.TestCase):
    """Test deduplication by node_id."""
    
    def setUp(self):
        """Set up extractor."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        self.extractor = PrecedentPathExtractor(traverser)
    
    def test_duplicate_node_ids_deduplicated(self):
        """Test that duplicate node_ids are deduplicated."""
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["ACT::IPC", "ACT::IPC", "SECTION::IPC::420"],
            graph_paths_used=[
                ["ACT::IPC", "SECTION::IPC::420"],
            ],
        )
        
        explanations = self.extractor.extract(result)
        
        # Should have 2 unique explanations (ACT::IPC and SECTION::IPC::420)
        self.assertEqual(len(explanations), 2)
        
        # Check node_ids are unique
        node_ids = [exp.node_id for exp in explanations]
        self.assertEqual(len(node_ids), len(set(node_ids)))


class TestAuthoritySorting(unittest.TestCase):
    """Test sorting by authority level."""
    
    def setUp(self):
        """Set up extractor with graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add nodes
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_case_node("case1")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.extractor = PrecedentPathExtractor(self.traverser)
    
    def test_statute_before_case(self):
        """Test that statutes are sorted before cases."""
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["CASE::CASE1", "SECTION::IPC::420"],
            graph_paths_used=[
                ["SECTION::IPC::420"],
                ["CASE::CASE1"],
            ],
        )
        
        explanations = self.extractor.extract(result)
        
        # Statute should come first
        self.assertEqual(explanations[0].authority_level, "statute")


class TestDeterministicOutput(unittest.TestCase):
    """Test that output is deterministic."""
    
    def setUp(self):
        """Set up extractor with graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_case_node("case1")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.extractor = PrecedentPathExtractor(self.traverser)
    
    def test_deterministic_extraction(self):
        """Test that same input produces same output."""
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["SECTION::IPC::420", "CASE::CASE1"],
            graph_paths_used=[
                ["SECTION::IPC::420"],
                ["CASE::CASE1"],
            ],
        )
        
        explanations1 = self.extractor.extract(result)
        explanations2 = self.extractor.extract(result)
        
        # Should produce same number of explanations
        self.assertEqual(len(explanations1), len(explanations2))
        
        # Should have same node_ids in same order
        node_ids1 = [exp.node_id for exp in explanations1]
        node_ids2 = [exp.node_id for exp in explanations2]
        self.assertEqual(node_ids1, node_ids2)
        
        # Should have same authority levels
        authorities1 = [exp.authority_level for exp in explanations1]
        authorities2 = [exp.authority_level for exp in explanations2]
        self.assertEqual(authorities1, authorities2)


class TestNoNewGraphTraversal(unittest.TestCase):
    """Test that no new graph traversal is performed."""
    
    def setUp(self):
        """Set up extractor with graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add section and case
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_case_node("case1")
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.extractor = PrecedentPathExtractor(self.traverser)
    
    def test_uses_only_provided_paths(self):
        """Test that extractor uses only provided paths."""
        # Provide limited paths
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["SECTION::IPC::420"],
            graph_paths_used=[
                ["SECTION::IPC::420"],  # Only this path provided
            ],
        )
        
        explanations = self.extractor.extract(result)
        
        # Should only have explanation for provided path
        self.assertEqual(len(explanations), 1)
        self.assertEqual(explanations[0].node_id, "SECTION::IPC::420")
        
        # Should use the provided path, not discover new ones
        self.assertEqual(explanations[0].graph_path, ["SECTION::IPC::420"])


class TestNoMissingCitedSources(unittest.TestCase):
    """Test that all cited sources are included."""
    
    def setUp(self):
        """Set up extractor."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        self.extractor = PrecedentPathExtractor(traverser)
    
    def test_all_cited_ids_processed(self):
        """Test that all cited_semantic_ids are processed."""
        cited_ids = ["sem1", "sem2", "sem3"]
        
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=cited_ids,
            graph_paths_used=[],
        )
        
        explanations = self.extractor.extract(result)
        
        # Should have explanation for each cited ID
        self.assertEqual(len(explanations), len(cited_ids))
        
        # Check all cited IDs are present
        cited_ids_in_explanations = [exp.cited_semantic_id for exp in explanations]
        for cited_id in cited_ids:
            self.assertIn(cited_id, cited_ids_in_explanations)


class TestRelationChainExtraction(unittest.TestCase):
    """Test relation chain extraction from paths."""
    
    def setUp(self):
        """Set up extractor with graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add nodes and edges
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_case_node("case1")
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.extractor = PrecedentPathExtractor(self.traverser)
    
    def test_extract_relation_chain(self):
        """Test extraction of relation chain."""
        path = ["CASE::CASE1", "SECTION::IPC::420"]
        
        relations = self.extractor._extract_relation_chain(path)
        
        # Should have one relation
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0], TraversalRelation.APPLIES_SECTION.value)
    
    def test_empty_path_no_relations(self):
        """Test that empty path produces no relations."""
        relations = self.extractor._extract_relation_chain([])
        self.assertEqual(len(relations), 0)
    
    def test_single_node_no_relations(self):
        """Test that single node produces no relations."""
        relations = self.extractor._extract_relation_chain(["SECTION::IPC::420"])
        self.assertEqual(len(relations), 0)


class TestPrecedentExplanationDataclass(unittest.TestCase):
    """Test PrecedentExplanation dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        explanation = PrecedentExplanation(
            cited_semantic_id="sem1",
            node_id="SECTION::IPC::420",
            node_type="SECTION",
            authority_level="statute",
            relation_chain=["APPLIES_SECTION"],
            graph_path=["CASE::CASE1", "SECTION::IPC::420"],
        )
        
        result = explanation.to_dict()
        
        # Check all fields present
        self.assertIn("cited_semantic_id", result)
        self.assertIn("node_id", result)
        self.assertIn("node_type", result)
        self.assertIn("authority_level", result)
        self.assertIn("relation_chain", result)
        self.assertIn("graph_path", result)
        
        # Check values
        self.assertEqual(result["cited_semantic_id"], "sem1")
        self.assertEqual(result["node_id"], "SECTION::IPC::420")
        self.assertEqual(result["node_type"], "SECTION")
        self.assertEqual(result["authority_level"], "statute")


class TestUtilityMethods(unittest.TestCase):
    """Test utility methods."""
    
    def setUp(self):
        """Set up extractor."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        self.extractor = PrecedentPathExtractor(traverser)
    
    def test_get_explanation_stats_empty(self):
        """Test stats for empty list."""
        stats = self.extractor.get_explanation_stats([])
        
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["by_type"], {})
        self.assertEqual(stats["by_authority"], {})
    
    def test_get_explanation_stats(self):
        """Test stats calculation."""
        explanations = [
            PrecedentExplanation(
                cited_semantic_id="sem1",
                node_id="SECTION::IPC::420",
                node_type="SECTION",
                authority_level="statute",
            ),
            PrecedentExplanation(
                cited_semantic_id="sem2",
                node_id="CASE::CASE1",
                node_type="CASE",
                authority_level="high_court",
            ),
        ]
        
        stats = self.extractor.get_explanation_stats(explanations)
        
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["by_type"]["SECTION"], 1)
        self.assertEqual(stats["by_type"]["CASE"], 1)
        self.assertEqual(stats["by_authority"]["statute"], 1)
        self.assertEqual(stats["by_authority"]["high_court"], 1)


class TestOverruledCasesNeverAppear(unittest.TestCase):
    """Test that overruled cases never appear in explanations."""
    
    def setUp(self):
        """Set up extractor with graph containing overruled case."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add section
        self.builder.add_has_section_edge("IPC", "420")
        
        # Add valid and overruled cases
        self.builder.add_case_node("case_valid")
        self.builder.add_case_node("case_overruled")
        self.builder.add_case_node("case_overruler")
        
        self.builder.graph.add_edge(
            "CASE::CASE_VALID", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULED", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULER", "CASE::CASE_OVERRULED",
            edge_type=TraversalRelation.OVERRULES_CASE.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.extractor = PrecedentPathExtractor(self.traverser)
    
    def test_overruled_case_not_in_explanations(self):
        """Test that overruled cases are not in explanations."""
        # Grounded result should NOT include overruled case
        # (it should have been filtered out by GraphRAGFilter)
        result = GroundedAnswerResult(
            answer="Test answer",
            cited_semantic_ids=["CASE::CASE_VALID"],  # Only valid case
            graph_paths_used=[
                ["CASE::CASE_VALID", "SECTION::IPC::420"],
            ],
        )
        
        explanations = self.extractor.extract(result)
        
        # Should only have valid case
        self.assertEqual(len(explanations), 1)
        self.assertEqual(explanations[0].node_id, "CASE::CASE_VALID")
        
        # Overruled case should not appear
        node_ids = [exp.node_id for exp in explanations]
        self.assertNotIn("CASE::CASE_OVERRULED", node_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
