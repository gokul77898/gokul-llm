"""
Tests for Legal Graph Traverser - Phase 3A

Tests traversal correctness, cycle prevention, overrule exclusion,
and deterministic output.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

from src.graph.legal_graph_traverser import (
    LegalGraphTraverser,
    TraversalRelation,
    TraversalResult,
)
from src.graph.legal_graph_builder import LegalGraphBuilder, NodeType


class TestTraverserSetup(unittest.TestCase):
    """Test traverser initialization."""
    
    def test_init_with_graph(self):
        """Test initialization with NetworkX graph."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        
        self.assertIsNotNone(traverser.graph)
        self.assertEqual(traverser.graph.number_of_nodes(), 0)
    
    def test_from_pickle(self):
        """Test loading from pickle file."""
        # Create a temporary graph
        temp_dir = tempfile.mkdtemp()
        output_dir = Path(temp_dir) / "graph"
        output_dir.mkdir(parents=True)
        
        builder = LegalGraphBuilder(
            documents_dir=str(Path(temp_dir) / "docs"),
            chunks_dir=str(Path(temp_dir) / "chunks"),
            output_dir=str(output_dir),
        )
        builder.add_act_node("IPC")
        builder._stats = builder._calculate_stats()
        pickle_path, _ = builder.save(version="test")
        
        # Load via traverser
        traverser = LegalGraphTraverser.from_pickle(str(pickle_path))
        
        self.assertIn("ACT::IPC", traverser.graph)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestGetSectionsForAct(unittest.TestCase):
    """Test get_sections_for_act method."""
    
    def setUp(self):
        """Set up test graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add IPC with sections
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "302")
        self.builder.add_has_section_edge("IPC", "376")
        
        # Add CrPC with sections
        self.builder.add_has_section_edge("CrPC", "154")
        self.builder.add_has_section_edge("CrPC", "161")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_get_sections_basic(self):
        """Test basic section retrieval."""
        sections = self.traverser.get_sections_for_act("IPC")
        
        self.assertEqual(len(sections), 3)
        self.assertIn("SECTION::IPC::420", sections)
        self.assertIn("SECTION::IPC::302", sections)
        self.assertIn("SECTION::IPC::376", sections)
    
    def test_get_sections_with_prefix(self):
        """Test with ACT:: prefix."""
        sections = self.traverser.get_sections_for_act("ACT::IPC")
        
        self.assertEqual(len(sections), 3)
    
    def test_get_sections_different_act(self):
        """Test sections for different act."""
        sections = self.traverser.get_sections_for_act("CrPC")
        
        self.assertEqual(len(sections), 2)
        self.assertIn("SECTION::CRPC::154", sections)
    
    def test_get_sections_nonexistent_act(self):
        """Test with non-existent act."""
        sections = self.traverser.get_sections_for_act("UNKNOWN_ACT")
        
        self.assertEqual(len(sections), 0)
    
    def test_get_sections_sorted(self):
        """Test that results are sorted."""
        sections = self.traverser.get_sections_for_act("IPC")
        
        self.assertEqual(sections, sorted(sections))


class TestGetCasesForSection(unittest.TestCase):
    """Test get_cases_for_section method."""
    
    def setUp(self):
        """Set up test graph with cases."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add section
        self.builder.add_has_section_edge("IPC", "420")
        
        # Add cases with different relations
        case1 = self.builder.add_case_node("case1")
        case2 = self.builder.add_case_node("case2")
        case3 = self.builder.add_case_node("case3")
        
        section_id = "SECTION::IPC::420"
        
        # Case1 applies section
        self.builder.graph.add_edge(
            case1, section_id,
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        # Case2 interprets section
        self.builder.graph.add_edge(
            case2, section_id,
            edge_type=TraversalRelation.INTERPRETS_SECTION.value
        )
        
        # Case3 distinguishes section
        self.builder.graph.add_edge(
            case3, section_id,
            edge_type=TraversalRelation.DISTINGUISHES_SECTION.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_get_all_cases(self):
        """Test getting all cases (no filter)."""
        cases = self.traverser.get_cases_for_section("SECTION::IPC::420")
        
        self.assertEqual(len(cases), 3)
    
    def test_get_cases_filtered_applies(self):
        """Test filtering by APPLIES_SECTION."""
        cases = self.traverser.get_cases_for_section(
            "SECTION::IPC::420",
            relation_types=[TraversalRelation.APPLIES_SECTION.value]
        )
        
        self.assertEqual(len(cases), 1)
        self.assertIn("CASE::CASE1", cases)
    
    def test_get_cases_filtered_multiple(self):
        """Test filtering by multiple relations."""
        cases = self.traverser.get_cases_for_section(
            "SECTION::IPC::420",
            relation_types=[
                TraversalRelation.APPLIES_SECTION.value,
                TraversalRelation.INTERPRETS_SECTION.value,
            ]
        )
        
        self.assertEqual(len(cases), 2)
    
    def test_get_cases_nonexistent_section(self):
        """Test with non-existent section."""
        cases = self.traverser.get_cases_for_section("SECTION::IPC::999")
        
        self.assertEqual(len(cases), 0)
    
    def test_get_cases_sorted(self):
        """Test that results are sorted."""
        cases = self.traverser.get_cases_for_section("SECTION::IPC::420")
        
        self.assertEqual(cases, sorted(cases))


class TestGetPrecedentChain(unittest.TestCase):
    """Test get_precedent_chain method."""
    
    def setUp(self):
        """Set up test graph with citation chain."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Create citation chain: A -> B -> C -> D
        self.builder.add_case_node("case_a")
        self.builder.add_case_node("case_b")
        self.builder.add_case_node("case_c")
        self.builder.add_case_node("case_d")
        
        self.builder.graph.add_edge(
            "CASE::CASE_A", "CASE::CASE_B",
            edge_type=TraversalRelation.CITES_CASE.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_B", "CASE::CASE_C",
            edge_type=TraversalRelation.CITES_CASE.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_C", "CASE::CASE_D",
            edge_type=TraversalRelation.CITES_CASE.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_precedent_chain_depth_1(self):
        """Test precedent chain with depth 1."""
        result = self.traverser.get_precedent_chain("case_a", depth=1)
        
        self.assertEqual(len(result.node_ids), 1)
        self.assertIn("CASE::CASE_B", result.node_ids)
    
    def test_precedent_chain_depth_2(self):
        """Test precedent chain with depth 2."""
        result = self.traverser.get_precedent_chain("case_a", depth=2)
        
        self.assertEqual(len(result.node_ids), 2)
        self.assertIn("CASE::CASE_B", result.node_ids)
        self.assertIn("CASE::CASE_C", result.node_ids)
    
    def test_precedent_chain_depth_3(self):
        """Test precedent chain with depth 3."""
        result = self.traverser.get_precedent_chain("case_a", depth=3)
        
        self.assertEqual(len(result.node_ids), 3)
        self.assertIn("CASE::CASE_D", result.node_ids)
    
    def test_precedent_chain_paths(self):
        """Test that paths are recorded."""
        result = self.traverser.get_precedent_chain("case_a", depth=2)
        
        self.assertGreater(len(result.paths), 0)
    
    def test_precedent_chain_nonexistent(self):
        """Test with non-existent case."""
        result = self.traverser.get_precedent_chain("nonexistent", depth=2)
        
        self.assertEqual(len(result.node_ids), 0)


class TestCyclePrevention(unittest.TestCase):
    """Test cycle prevention in traversals."""
    
    def setUp(self):
        """Set up test graph with cycles."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Create cycle: A -> B -> C -> A
        self.builder.add_case_node("case_a")
        self.builder.add_case_node("case_b")
        self.builder.add_case_node("case_c")
        
        self.builder.graph.add_edge(
            "CASE::CASE_A", "CASE::CASE_B",
            edge_type=TraversalRelation.CITES_CASE.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_B", "CASE::CASE_C",
            edge_type=TraversalRelation.CITES_CASE.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_C", "CASE::CASE_A",
            edge_type=TraversalRelation.CITES_CASE.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_cycle_avoided_in_precedent_chain(self):
        """Test that cycles are avoided in precedent chain."""
        result = self.traverser.get_precedent_chain("case_a", depth=10)
        
        # Should find B, C, and A (via C->A edge) but not infinite loop
        # The key is that traversal terminates and doesn't loop infinitely
        self.assertLessEqual(len(result.node_ids), 3)
        
        # B and C should definitely be found
        self.assertIn("CASE::CASE_B", result.node_ids)
        self.assertIn("CASE::CASE_C", result.node_ids)
    
    def test_no_infinite_loop(self):
        """Test that traversal terminates."""
        # This should complete without hanging
        result = self.traverser.get_precedent_chain("case_a", depth=100)
        
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.node_ids), 3)


class TestOverruleExclusion(unittest.TestCase):
    """Test overrule exclusion in get_applicable_cases."""
    
    def setUp(self):
        """Set up test graph with overruled cases."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add section
        self.builder.add_has_section_edge("IPC", "420")
        section_id = "SECTION::IPC::420"
        
        # Add cases
        self.builder.add_case_node("case_valid")
        self.builder.add_case_node("case_overruled")
        self.builder.add_case_node("case_overruler")
        
        # Both cases apply section
        self.builder.graph.add_edge(
            "CASE::CASE_VALID", section_id,
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULED", section_id,
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        # case_overruler overrules case_overruled
        self.builder.graph.add_edge(
            "CASE::CASE_OVERRULER", "CASE::CASE_OVERRULED",
            edge_type=TraversalRelation.OVERRULES_CASE.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_overruled_excluded(self):
        """Test that overruled cases are excluded."""
        result = self.traverser.get_applicable_cases("SECTION::IPC::420")
        
        self.assertEqual(len(result.node_ids), 1)
        self.assertIn("CASE::CASE_VALID", result.node_ids)
        self.assertNotIn("CASE::CASE_OVERRULED", result.node_ids)
    
    def test_overruled_count_reported(self):
        """Test that excluded count is reported."""
        result = self.traverser.get_applicable_cases("SECTION::IPC::420")
        
        self.assertEqual(result.overruled_excluded, 1)
    
    def test_is_overruled_check(self):
        """Test _is_overruled helper."""
        self.assertTrue(self.traverser._is_overruled("CASE::CASE_OVERRULED"))
        self.assertFalse(self.traverser._is_overruled("CASE::CASE_VALID"))


class TestGetRelatedSections(unittest.TestCase):
    """Test get_related_sections method."""
    
    def setUp(self):
        """Set up test graph with related sections."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add sections
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "421")
        self.builder.add_has_section_edge("IPC", "302")
        
        # Add case that references multiple sections
        self.builder.add_case_node("case1")
        
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::421",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::302",
            edge_type=TraversalRelation.INTERPRETS_SECTION.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_related_sections_found(self):
        """Test finding related sections."""
        result = self.traverser.get_related_sections("SECTION::IPC::420")
        
        # Should find 421 and 302 (related via case1)
        self.assertEqual(len(result.node_ids), 2)
        self.assertIn("SECTION::IPC::421", result.node_ids)
        self.assertIn("SECTION::IPC::302", result.node_ids)
    
    def test_related_sections_paths(self):
        """Test that paths are recorded."""
        result = self.traverser.get_related_sections("SECTION::IPC::420")
        
        self.assertGreater(len(result.paths), 0)
        # Each path should go through a case
        for path in result.paths:
            self.assertTrue(any("CASE::" in node for node in path))
    
    def test_related_sections_excludes_self(self):
        """Test that starting section is excluded."""
        result = self.traverser.get_related_sections("SECTION::IPC::420")
        
        self.assertNotIn("SECTION::IPC::420", result.node_ids)


class TestDeterministicOutput(unittest.TestCase):
    """Test that all outputs are deterministic."""
    
    def setUp(self):
        """Set up test graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add data
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "302")
        self.builder.add_case_node("case1")
        self.builder.add_case_node("case2")
        
        self.builder.graph.add_edge(
            "CASE::CASE1", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        self.builder.graph.add_edge(
            "CASE::CASE2", "SECTION::IPC::420",
            edge_type=TraversalRelation.APPLIES_SECTION.value
        )
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_sections_deterministic(self):
        """Test get_sections_for_act is deterministic."""
        result1 = self.traverser.get_sections_for_act("IPC")
        result2 = self.traverser.get_sections_for_act("IPC")
        
        self.assertEqual(result1, result2)
    
    def test_cases_deterministic(self):
        """Test get_cases_for_section is deterministic."""
        result1 = self.traverser.get_cases_for_section("SECTION::IPC::420")
        result2 = self.traverser.get_cases_for_section("SECTION::IPC::420")
        
        self.assertEqual(result1, result2)
    
    def test_applicable_cases_deterministic(self):
        """Test get_applicable_cases is deterministic."""
        result1 = self.traverser.get_applicable_cases("SECTION::IPC::420")
        result2 = self.traverser.get_applicable_cases("SECTION::IPC::420")
        
        self.assertEqual(result1.node_ids, result2.node_ids)


class TestUtilityMethods(unittest.TestCase):
    """Test utility methods."""
    
    def setUp(self):
        """Set up test graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_case_node("case1", metadata={"court": "SC", "year": 2020})
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
    
    def test_node_exists(self):
        """Test node_exists method."""
        self.assertTrue(self.traverser.node_exists("ACT::IPC"))
        self.assertTrue(self.traverser.node_exists("SECTION::IPC::420"))
        self.assertFalse(self.traverser.node_exists("ACT::UNKNOWN"))
    
    def test_get_node_info(self):
        """Test get_node_info method."""
        info = self.traverser.get_node_info("CASE::CASE1")
        
        self.assertIsNotNone(info)
        self.assertEqual(info.get("node_type"), "case")
        self.assertEqual(info.get("court"), "SC")
        self.assertEqual(info.get("year"), 2020)
    
    def test_get_node_info_nonexistent(self):
        """Test get_node_info with non-existent node."""
        info = self.traverser.get_node_info("NONEXISTENT")
        
        self.assertIsNone(info)
    
    def test_get_act_for_section(self):
        """Test get_act_for_section method."""
        act = self.traverser.get_act_for_section("SECTION::IPC::420")
        
        self.assertEqual(act, "ACT::IPC")
    
    def test_get_graph_stats(self):
        """Test get_graph_stats method."""
        stats = self.traverser.get_graph_stats()
        
        self.assertIn("total_nodes", stats)
        self.assertIn("total_edges", stats)
        self.assertIn("nodes_by_type", stats)
        self.assertIn("edges_by_type", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
