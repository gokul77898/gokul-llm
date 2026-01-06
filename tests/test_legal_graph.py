"""
Tests for Legal Graph Builder - Phase 1

Tests the graph building, validation, and persistence logic.
"""

import json
import os
import sys
import tempfile
import unittest
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


class TestNodeIdGeneration(unittest.TestCase):
    """Test deterministic node ID generation."""
    
    def test_act_id_generation(self):
        """Test Act node ID generation."""
        self.assertEqual(
            LegalGraphBuilder.make_act_id("IPC"),
            "ACT::IPC"
        )
        self.assertEqual(
            LegalGraphBuilder.make_act_id("Indian Penal Code"),
            "ACT::IPC"
        )
        self.assertEqual(
            LegalGraphBuilder.make_act_id("ipc"),
            "ACT::IPC"
        )
    
    def test_section_id_generation(self):
        """Test Section node ID generation."""
        self.assertEqual(
            LegalGraphBuilder.make_section_id("IPC", "420"),
            "SECTION::IPC::420"
        )
        self.assertEqual(
            LegalGraphBuilder.make_section_id("IPC", "Section 420"),
            "SECTION::IPC::420"
        )
        self.assertEqual(
            LegalGraphBuilder.make_section_id("IPC", "420(1)"),
            "SECTION::IPC::420_1"
        )
    
    def test_case_id_generation(self):
        """Test Case node ID generation."""
        self.assertEqual(
            LegalGraphBuilder.make_case_id("abc123"),
            "CASE::ABC123"
        )
        self.assertEqual(
            LegalGraphBuilder.make_case_id("State v. Sharma"),
            "CASE::STATE_V._SHARMA"
        )


class TestNormalization(unittest.TestCase):
    """Test normalization functions."""
    
    def test_normalize_act_name(self):
        """Test act name normalization."""
        normalize = LegalGraphBuilder.normalize_act_name
        
        # Known abbreviations
        self.assertEqual(normalize("IPC"), "IPC")
        self.assertEqual(normalize("ipc"), "IPC")
        self.assertEqual(normalize("Indian Penal Code"), "IPC")
        self.assertEqual(normalize("CrPC"), "CRPC")
        self.assertEqual(normalize("Code of Criminal Procedure"), "CRPC")
        
        # Unknown acts
        self.assertEqual(normalize("Some New Act"), "SOME_NEW_ACT")
        self.assertEqual(normalize(""), "UNKNOWN_ACT")
    
    def test_normalize_section(self):
        """Test section normalization."""
        normalize = LegalGraphBuilder.normalize_section
        
        self.assertEqual(normalize("420"), "420")
        self.assertEqual(normalize("Section 420"), "420")
        self.assertEqual(normalize("section 420"), "420")
        self.assertEqual(normalize("420(1)"), "420_1")
        self.assertEqual(normalize("420(1)(a)"), "420_1_A")
        self.assertEqual(normalize("420-A"), "420_A")
        self.assertEqual(normalize(""), "UNKNOWN")


class TestGraphBuilding(unittest.TestCase):
    """Test graph building functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = Path(self.temp_dir) / "documents"
        self.chunks_dir = Path(self.temp_dir) / "chunks"
        self.output_dir = Path(self.temp_dir) / "graph"
        
        self.docs_dir.mkdir(parents=True)
        self.chunks_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        self.builder = LegalGraphBuilder(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            output_dir=str(self.output_dir),
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_act_node(self):
        """Test adding Act nodes."""
        node_id = self.builder.add_act_node("IPC")
        
        self.assertEqual(node_id, "ACT::IPC")
        self.assertIn(node_id, self.builder.graph)
        
        attrs = self.builder.graph.nodes[node_id]
        self.assertEqual(attrs["node_type"], NodeType.ACT.value)
        self.assertEqual(attrs["name"], "IPC")
    
    def test_add_section_node(self):
        """Test adding Section nodes."""
        node_id = self.builder.add_section_node("IPC", "420")
        
        self.assertEqual(node_id, "SECTION::IPC::420")
        self.assertIn(node_id, self.builder.graph)
        
        attrs = self.builder.graph.nodes[node_id]
        self.assertEqual(attrs["node_type"], NodeType.SECTION.value)
        self.assertEqual(attrs["act"], "IPC")
        self.assertEqual(attrs["section"], "420")
    
    def test_add_case_node(self):
        """Test adding Case nodes."""
        node_id = self.builder.add_case_node("case123", metadata={
            "court": "Supreme Court",
            "year": 2020,
        })
        
        self.assertEqual(node_id, "CASE::CASE123")
        self.assertIn(node_id, self.builder.graph)
        
        attrs = self.builder.graph.nodes[node_id]
        self.assertEqual(attrs["node_type"], NodeType.CASE.value)
        self.assertEqual(attrs["court"], "Supreme Court")
        self.assertEqual(attrs["year"], 2020)
    
    def test_add_has_section_edge(self):
        """Test adding HAS_SECTION edges."""
        act_id, section_id = self.builder.add_has_section_edge("IPC", "420")
        
        self.assertTrue(self.builder.graph.has_edge(act_id, section_id))
        
        edge_data = self.builder.graph.get_edge_data(act_id, section_id)
        self.assertEqual(edge_data["edge_type"], EdgeType.HAS_SECTION.value)
    
    def test_add_mentions_section_edge(self):
        """Test adding MENTIONS_SECTION edges."""
        case_id, section_id = self.builder.add_mentions_section_edge(
            "case123", "IPC", "420"
        )
        
        self.assertTrue(self.builder.graph.has_edge(case_id, section_id))
        
        edge_data = self.builder.graph.get_edge_data(case_id, section_id)
        self.assertEqual(edge_data["edge_type"], EdgeType.MENTIONS_SECTION.value)
        
        # Should also create Act->Section edge
        act_id = self.builder.make_act_id("IPC")
        self.assertTrue(self.builder.graph.has_edge(act_id, section_id))
    
    def test_add_belongs_to_act_edge(self):
        """Test adding BELONGS_TO_ACT edges."""
        case_id, act_id = self.builder.add_belongs_to_act_edge("case123", "IPC")
        
        self.assertTrue(self.builder.graph.has_edge(case_id, act_id))
        
        edge_data = self.builder.graph.get_edge_data(case_id, act_id)
        self.assertEqual(edge_data["edge_type"], EdgeType.BELONGS_TO_ACT.value)
    
    def test_deduplication(self):
        """Test that nodes are deduplicated."""
        # Add same act twice
        id1 = self.builder.add_act_node("IPC")
        id2 = self.builder.add_act_node("IPC")
        
        self.assertEqual(id1, id2)
        self.assertEqual(self.builder.graph.number_of_nodes(), 1)
        
        # Add same section twice
        id3 = self.builder.add_section_node("IPC", "420")
        id4 = self.builder.add_section_node("IPC", "420")
        
        self.assertEqual(id3, id4)
        self.assertEqual(self.builder.graph.number_of_nodes(), 2)  # Act + Section


class TestGraphBuildFromData(unittest.TestCase):
    """Test building graph from document/chunk data."""
    
    def setUp(self):
        """Set up test fixtures with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = Path(self.temp_dir) / "documents"
        self.chunks_dir = Path(self.temp_dir) / "chunks"
        self.output_dir = Path(self.temp_dir) / "graph"
        
        self.docs_dir.mkdir(parents=True)
        self.chunks_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create sample documents
        self._create_sample_documents()
        self._create_sample_chunks()
        
        self.builder = LegalGraphBuilder(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            output_dir=str(self.output_dir),
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_documents(self):
        """Create sample document files."""
        # Bare act document
        doc1 = {
            "doc_id": "doc_ipc",
            "title": "Indian Penal Code",
            "doc_type": "bare_act",
            "act": "IPC",
            "year": 1860,
            "raw_text": "Section 420. Cheating...",
            "source": "ipc.pdf",
        }
        with open(self.docs_dir / "doc_ipc.json", "w") as f:
            json.dump(doc1, f)
        
        # Case law document
        doc2 = {
            "doc_id": "case_001",
            "title": "State v. Sharma",
            "doc_type": "case_law",
            "act": "IPC",
            "court": "Supreme Court",
            "year": 2020,
            "citation": "2020 SCC 123",
            "raw_text": "The accused was charged under Section 420 IPC...",
            "source": "case_001.pdf",
        }
        with open(self.docs_dir / "case_001.json", "w") as f:
            json.dump(doc2, f)
    
    def _create_sample_chunks(self):
        """Create sample chunk files and index."""
        chunks = {
            "chunk_001": {
                "chunk_id": "chunk_001",
                "doc_id": "doc_ipc",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Whoever cheats...",
            },
            "chunk_002": {
                "chunk_id": "chunk_002",
                "doc_id": "doc_ipc",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "421",
                "text": "Whoever dishonestly...",
            },
            "chunk_003": {
                "chunk_id": "chunk_003",
                "doc_id": "case_001",
                "doc_type": "case_law",
                "act": "IPC",
                "section": "420",
                "court": "Supreme Court",
                "year": 2020,
                "citation": "2020 SCC 123",
                "text": "The court held that...",
            },
        }
        
        # Create index file
        index_data = {
            "version": 1,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
        with open(self.chunks_dir / "index.json", "w") as f:
            json.dump(index_data, f)
    
    def test_build_from_documents(self):
        """Test building graph from documents."""
        count = self.builder.build_from_documents()
        
        self.assertEqual(count, 2)
        
        # Check Act node exists
        act_id = self.builder.make_act_id("IPC")
        self.assertIn(act_id, self.builder.graph)
        
        # Check Case node exists
        case_id = self.builder.make_case_id("2020 SCC 123")
        self.assertIn(case_id, self.builder.graph)
    
    def test_build_from_chunks(self):
        """Test building graph from chunks."""
        count = self.builder.build_from_chunks()
        
        self.assertEqual(count, 3)
        
        # Check Section nodes exist
        section_420 = self.builder.make_section_id("IPC", "420")
        section_421 = self.builder.make_section_id("IPC", "421")
        
        self.assertIn(section_420, self.builder.graph)
        self.assertIn(section_421, self.builder.graph)
    
    def test_full_build(self):
        """Test full graph build."""
        stats = self.builder.build()
        
        # Should have: 1 Act, 2 Sections, 1 Case
        self.assertEqual(stats.nodes_by_type["act"], 1)
        self.assertEqual(stats.nodes_by_type["section"], 2)
        self.assertEqual(stats.nodes_by_type["case"], 1)
        
        # Should have edges
        self.assertGreater(stats.total_edges, 0)
    
    def test_deterministic_rebuild(self):
        """Test that rebuild produces same graph."""
        # First build
        self.builder.build()
        nodes1 = self.builder.graph.number_of_nodes()
        edges1 = self.builder.graph.number_of_edges()
        
        # Rebuild
        builder2 = LegalGraphBuilder(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            output_dir=str(self.output_dir),
        )
        builder2.build()
        nodes2 = builder2.graph.number_of_nodes()
        edges2 = builder2.graph.number_of_edges()
        
        self.assertEqual(nodes1, nodes2)
        self.assertEqual(edges1, edges2)


class TestValidation(unittest.TestCase):
    """Test graph validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = LegalGraphBuilder(
            documents_dir=str(Path(self.temp_dir) / "docs"),
            chunks_dir=str(Path(self.temp_dir) / "chunks"),
            output_dir=str(Path(self.temp_dir) / "graph"),
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_valid_graph(self):
        """Test validation of valid graph."""
        # Build valid graph
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_mentions_section_edge("case123", "IPC", "420")
        
        result = self.builder.validate()
        
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.orphan_sections), 0)
    
    def test_orphan_section_detection(self):
        """Test detection of orphan sections."""
        # Add section without Act->Section edge
        self.builder.add_section_node("IPC", "420")
        # Don't add the Act->Section edge
        
        result = self.builder.validate()
        
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.orphan_sections), 1)
    
    def test_case_without_act(self):
        """Test detection of cases without act."""
        # Add case without any edges
        self.builder.add_case_node("case123")
        
        result = self.builder.validate()
        
        self.assertEqual(len(result.cases_without_act), 1)


class TestPersistence(unittest.TestCase):
    """Test graph persistence."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "graph"
        self.output_dir.mkdir(parents=True)
        
        self.builder = LegalGraphBuilder(
            documents_dir=str(Path(self.temp_dir) / "docs"),
            chunks_dir=str(Path(self.temp_dir) / "chunks"),
            output_dir=str(self.output_dir),
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load(self):
        """Test saving and loading graph."""
        # Build graph
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "421")
        self.builder.add_mentions_section_edge("case123", "IPC", "420")
        self.builder._stats = self.builder._calculate_stats()
        
        # Save
        pickle_path, json_path = self.builder.save(version="test")
        
        self.assertTrue(pickle_path.exists())
        self.assertTrue(json_path.exists())
        
        # Load
        loaded = LegalGraphBuilder.load(str(pickle_path))
        
        self.assertEqual(
            self.builder.graph.number_of_nodes(),
            loaded.graph.number_of_nodes()
        )
        self.assertEqual(
            self.builder.graph.number_of_edges(),
            loaded.graph.number_of_edges()
        )
    
    def test_json_output(self):
        """Test JSON output format."""
        self.builder.add_has_section_edge("IPC", "420")
        self.builder._stats = self.builder._calculate_stats()
        
        _, json_path = self.builder.save(version="test")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("stats", data["metadata"])


class TestQueryMethods(unittest.TestCase):
    """Test graph query methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = LegalGraphBuilder(
            documents_dir=str(Path(self.temp_dir) / "docs"),
            chunks_dir=str(Path(self.temp_dir) / "chunks"),
            output_dir=str(Path(self.temp_dir) / "graph"),
        )
        
        # Build sample graph
        self.builder.add_has_section_edge("IPC", "420")
        self.builder.add_has_section_edge("IPC", "421")
        self.builder.add_has_section_edge("IPC", "302")
        self.builder.add_mentions_section_edge("case1", "IPC", "420")
        self.builder.add_mentions_section_edge("case2", "IPC", "420")
        self.builder.add_mentions_section_edge("case3", "IPC", "302")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_sections_for_act(self):
        """Test getting sections for an act."""
        sections = self.builder.get_sections_for_act("IPC")
        
        self.assertEqual(len(sections), 3)
        self.assertIn("420", sections)
        self.assertIn("421", sections)
        self.assertIn("302", sections)
    
    def test_get_cases_for_section(self):
        """Test getting cases for a section."""
        cases = self.builder.get_cases_for_section("IPC", "420")
        
        self.assertEqual(len(cases), 2)
        self.assertIn("case1", cases)
        self.assertIn("case2", cases)


if __name__ == "__main__":
    unittest.main(verbosity=2)
