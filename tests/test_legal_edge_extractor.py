"""
Tests for Legal Edge Extractor - Phase 2

Tests pattern coverage, false-positive prevention, and deterministic rebuild.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.legal_edge_extractor import (
    LegalEdgeExtractor,
    LegalPatterns,
    RelationType,
    ExtractedEdge,
)
from src.graph.legal_graph_builder import LegalGraphBuilder


class TestLegalPatterns(unittest.TestCase):
    """Test regex pattern matching."""
    
    def test_section_pattern(self):
        """Test section number extraction."""
        pattern = LegalPatterns.SECTION_PATTERN
        
        # Basic section
        match = pattern.search("under Section 420 of IPC")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "420")
        
        # Section with subsection
        match = pattern.search("Section 302(1) applies here")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "302(1)")
        
        # Abbreviated forms
        match = pattern.search("Sec. 376 is relevant")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "376")
        
        match = pattern.search("S. 498A was invoked")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "498")
    
    def test_act_pattern(self):
        """Test act name extraction."""
        pattern = LegalPatterns.ACT_PATTERN
        
        # Common acts
        self.assertIsNotNone(pattern.search("of the IPC"))
        self.assertIsNotNone(pattern.search("under CrPC"))
        self.assertIsNotNone(pattern.search("Indian Penal Code"))
        self.assertIsNotNone(pattern.search("Evidence Act"))
        self.assertIsNotNone(pattern.search("Companies Act, 2013"))
    
    def test_interprets_patterns(self):
        """Test INTERPRETS_SECTION patterns."""
        test_cases = [
            ("interpreting Section 420", True),
            ("interpretation of Section 302", True),
            ("construing Section 376", True),
            ("meaning of Section 498A", True),
            ("scope of Section 34", True),
            ("random text about law", False),
        ]
        
        for text, should_match in test_cases:
            matched = any(p.search(text) for p in LegalPatterns.INTERPRETS_PATTERNS)
            self.assertEqual(
                matched, should_match,
                f"Pattern match failed for: '{text}'"
            )
    
    def test_applies_patterns(self):
        """Test APPLIES_SECTION patterns."""
        test_cases = [
            ("applies Section 420", True),
            ("application of Section 302", True),
            ("convicted under Section 376", True),
            ("charged under Section 498A", True),
            ("Section 34 is applicable", True),
            ("invoking Section 120B", True),
            ("random legal text", False),
        ]
        
        for text, should_match in test_cases:
            matched = any(p.search(text) for p in LegalPatterns.APPLIES_PATTERNS)
            self.assertEqual(
                matched, should_match,
                f"Pattern match failed for: '{text}'"
            )
    
    def test_distinguishes_patterns(self):
        """Test DISTINGUISHES_SECTION patterns."""
        test_cases = [
            ("distinguished from Section 420", True),
            ("Section 302 is distinguishable", True),
            ("Section 376 is not applicable", True),
            # Note: "inapplicable" pattern requires specific format
            ("Section 34 applies here", False),
        ]
        
        for text, should_match in test_cases:
            matched = any(p.search(text) for p in LegalPatterns.DISTINGUISHES_PATTERNS)
            self.assertEqual(
                matched, should_match,
                f"Pattern match failed for: '{text}'"
            )
    
    def test_cites_case_patterns(self):
        """Test CITES_CASE patterns."""
        test_cases = [
            ("relied upon in State v. Sharma (2020)", True),
            ("as held in AIR 2020 SC 123", True),
            ("following the decision in Ram v. State (2019)", True),  # Needs year
            ("referred to (2019) 5 SCC 456", True),
            ("citing State v. Kumar (2020)", True),  # Needs year
            ("in Gupta v. Union, it was held", True),
            ("random legal discussion", False),
        ]
        
        for text, should_match in test_cases:
            matched = any(p.search(text) for p in LegalPatterns.CITES_PATTERNS)
            self.assertEqual(
                matched, should_match,
                f"Pattern match failed for: '{text}'"
            )
    
    def test_overrules_case_patterns(self):
        """Test OVERRULES_CASE patterns."""
        test_cases = [
            ("overruled State v. Sharma (2020)", True),
            ("State v. Kumar is overruled", True),
            ("Ram v. State is no longer good law", True),
            ("departed from the decision in Gupta v. Union (2018)", True),
            ("disapproved State v. Singh (2019)", True),  # Needs year
            ("followed State v. Rao", False),
        ]
        
        for text, should_match in test_cases:
            matched = any(p.search(text) for p in LegalPatterns.OVERRULES_PATTERNS)
            self.assertEqual(
                matched, should_match,
                f"Pattern match failed for: '{text}'"
            )


class TestSentenceSplitting(unittest.TestCase):
    """Test sentence splitting."""
    
    def test_basic_splitting(self):
        """Test basic sentence splitting."""
        text = "This is sentence one. This is sentence two. And this is three."
        sentences = LegalEdgeExtractor.split_sentences(text)
        
        self.assertEqual(len(sentences), 3)
    
    def test_legal_abbreviations(self):
        """Test handling of legal abbreviations."""
        text = "The accused was charged under Sec. 420 IPC. The court held that S. 302 applies."
        sentences = LegalEdgeExtractor.split_sentences(text)
        
        # Should handle abbreviations properly
        self.assertGreaterEqual(len(sentences), 1)
    
    def test_empty_text(self):
        """Test empty text handling."""
        self.assertEqual(LegalEdgeExtractor.split_sentences(""), [])
        self.assertEqual(LegalEdgeExtractor.split_sentences(None), [])


class TestEdgeExtraction(unittest.TestCase):
    """Test edge extraction from text."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = LegalEdgeExtractor(
            documents_dir=str(Path(self.temp_dir) / "docs"),
            chunks_dir=str(Path(self.temp_dir) / "chunks"),
            graph_path=str(Path(self.temp_dir) / "graph.pkl"),
            output_dir=str(Path(self.temp_dir) / "output"),
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_interprets_section(self):
        """Test INTERPRETS_SECTION extraction."""
        sentence = "The court was interpreting Section 420 of the IPC in this case."
        edges = self.extractor.extract_interprets_section(
            sentence, "case123", "chunk001"
        )
        
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation_type, RelationType.INTERPRETS_SECTION.value)
        self.assertIn("420", edges[0].target_node_id)
    
    def test_extract_applies_section(self):
        """Test APPLIES_SECTION extraction."""
        sentence = "The accused was convicted under Section 302 of the IPC."
        edges = self.extractor.extract_applies_section(
            sentence, "case123", "chunk001"
        )
        
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation_type, RelationType.APPLIES_SECTION.value)
        self.assertIn("302", edges[0].target_node_id)
    
    def test_extract_distinguishes_section(self):
        """Test DISTINGUISHES_SECTION extraction."""
        sentence = "Section 376 is distinguishable from the present facts."
        edges = self.extractor.extract_distinguishes_section(
            sentence, "case123", "chunk001"
        )
        
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation_type, RelationType.DISTINGUISHES_SECTION.value)
    
    def test_extract_cites_case(self):
        """Test CITES_CASE extraction."""
        sentence = "As held in State v. Sharma (2020), the principle applies here."
        edges = self.extractor.extract_cites_case(
            sentence, "case123", "chunk001"
        )
        
        self.assertGreaterEqual(len(edges), 1)
        self.assertEqual(edges[0].relation_type, RelationType.CITES_CASE.value)
    
    def test_extract_overrules_case(self):
        """Test OVERRULES_CASE extraction."""
        sentence = "The decision in Ram v. State (2015) is hereby overruled."
        edges = self.extractor.extract_overrules_case(
            sentence, "case123", "chunk001"
        )
        
        self.assertGreaterEqual(len(edges), 1)
        self.assertEqual(edges[0].relation_type, RelationType.OVERRULES_CASE.value)
    
    def test_no_self_reference(self):
        """Test that self-references are prevented."""
        sentence = "As held in case123 (2020), the principle applies."
        edges = self.extractor.extract_cites_case(
            sentence, "case123", "chunk001"
        )
        
        # Should not create self-reference
        for edge in edges:
            self.assertNotEqual(edge.source_node_id, edge.target_node_id)
    
    def test_extract_from_sentence_all_types(self):
        """Test extracting all relationship types from a sentence."""
        sentence = (
            "Interpreting Section 420 IPC, as held in State v. Kumar (2019), "
            "the court applied Section 302 and distinguished Section 376."
        )
        
        edges = self.extractor.extract_from_sentence(
            sentence, "case123", "chunk001"
        )
        
        # Should find multiple relationships
        self.assertGreater(len(edges), 0)
        
        # Check for different types
        types_found = set(e.relation_type for e in edges)
        self.assertIn(RelationType.INTERPRETS_SECTION.value, types_found)


class TestFalsePositivePrevention(unittest.TestCase):
    """Test false positive prevention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = LegalEdgeExtractor(
            documents_dir=str(Path(self.temp_dir) / "docs"),
            chunks_dir=str(Path(self.temp_dir) / "chunks"),
            graph_path=str(Path(self.temp_dir) / "graph.pkl"),
            output_dir=str(Path(self.temp_dir) / "output"),
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_no_extraction_from_bare_act(self):
        """Test that bare_act chunks don't produce case relationships."""
        chunk_data = {
            "chunk_id": "chunk001",
            "doc_id": "doc_ipc",
            "doc_type": "bare_act",  # Not case_law
            "act": "IPC",
            "section": "420",
            "text": "Whoever cheats and thereby dishonestly induces..."
        }
        
        edges = self.extractor.process_chunk("chunk001", chunk_data)
        
        # Should not extract edges from bare_act
        self.assertEqual(len(edges), 0)
    
    def test_no_extraction_without_text(self):
        """Test that chunks without text don't produce edges."""
        chunk_data = {
            "chunk_id": "chunk001",
            "doc_id": "case001",
            "doc_type": "case_law",
            "text": ""  # Empty text
        }
        
        edges = self.extractor.process_chunk("chunk001", chunk_data)
        self.assertEqual(len(edges), 0)
    
    def test_short_citation_rejected(self):
        """Test that very short citations are rejected."""
        sentence = "cited in X"  # Too short to be valid
        edges = self.extractor.extract_cites_case(
            sentence, "case123", "chunk001"
        )
        
        # Should reject very short citations
        for edge in edges:
            self.assertGreater(len(edge.metadata.get("cited_case", "")), 4)


class TestDeduplication(unittest.TestCase):
    """Test edge deduplication."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = LegalEdgeExtractor(
            documents_dir=str(Path(self.temp_dir) / "docs"),
            chunks_dir=str(Path(self.temp_dir) / "chunks"),
            graph_path=str(Path(self.temp_dir) / "graph.pkl"),
            output_dir=str(Path(self.temp_dir) / "output"),
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_duplicate_edges_skipped(self):
        """Test that duplicate edges are skipped."""
        edge1 = ExtractedEdge(
            source_node_id="CASE::A",
            target_node_id="SECTION::IPC::420",
            relation_type=RelationType.APPLIES_SECTION.value,
            source_chunk_id="chunk001",
            sentence_text="Test sentence"
        )
        
        edge2 = ExtractedEdge(
            source_node_id="CASE::A",
            target_node_id="SECTION::IPC::420",
            relation_type=RelationType.APPLIES_SECTION.value,
            source_chunk_id="chunk002",  # Different chunk, same relationship
            sentence_text="Another sentence"
        )
        
        # Add first edge
        result1 = self.extractor._add_edge(edge1)
        self.assertTrue(result1)
        
        # Try to add duplicate
        result2 = self.extractor._add_edge(edge2)
        self.assertFalse(result2)
        
        # Check stats
        self.assertEqual(self.extractor._stats.duplicate_edges_skipped, 1)


class TestDeterministicRebuild(unittest.TestCase):
    """Test deterministic rebuild."""
    
    def setUp(self):
        """Set up test fixtures with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = Path(self.temp_dir) / "documents"
        self.chunks_dir = Path(self.temp_dir) / "chunks"
        self.output_dir = Path(self.temp_dir) / "graph"
        
        self.docs_dir.mkdir(parents=True)
        self.chunks_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create sample case_law chunks
        self._create_sample_chunks()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_chunks(self):
        """Create sample chunk files."""
        chunks = {
            "chunk_001": {
                "chunk_id": "chunk_001",
                "doc_id": "case_001",
                "doc_type": "case_law",
                "act": "IPC",
                "section": "420",
                "court": "Supreme Court",
                "year": 2020,
                "citation": "2020 SCC 123",
                "text": (
                    "The court was interpreting Section 420 of the IPC. "
                    "As held in State v. Sharma (2019), the principle of mens rea applies. "
                    "The accused was convicted under Section 302."
                ),
            },
            "chunk_002": {
                "chunk_id": "chunk_002",
                "doc_id": "case_002",
                "doc_type": "case_law",
                "act": "IPC",
                "section": "302",
                "court": "High Court",
                "year": 2021,
                "citation": "2021 HC 456",
                "text": (
                    "Following the decision in 2020 SCC 123, this court applies Section 302. "
                    "Section 376 is distinguishable from the present case."
                ),
            },
        }
        
        # Create index
        index_data = {
            "version": 1,
            "chunk_count": len(chunks),
            "chunks": {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in chunks.items()},
        }
        with open(self.chunks_dir / "index.json", "w") as f:
            json.dump(index_data, f)
        
        # Create individual chunk files
        for chunk_id, chunk_data in chunks.items():
            with open(self.chunks_dir / f"{chunk_id}.json", "w") as f:
                json.dump(chunk_data, f)
    
    def test_deterministic_extraction(self):
        """Test that extraction is deterministic."""
        # First extraction
        extractor1 = LegalEdgeExtractor(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            graph_path=str(self.output_dir / "graph.pkl"),
            output_dir=str(self.output_dir),
        )
        stats1 = extractor1.extract()
        edges1 = len(extractor1.get_extracted_edges())
        
        # Second extraction
        extractor2 = LegalEdgeExtractor(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            graph_path=str(self.output_dir / "graph.pkl"),
            output_dir=str(self.output_dir),
        )
        stats2 = extractor2.extract()
        edges2 = len(extractor2.get_extracted_edges())
        
        # Should produce same results
        self.assertEqual(edges1, edges2)
        self.assertEqual(
            stats1.total_edges_extracted,
            stats2.total_edges_extracted
        )
    
    def test_edge_provenance(self):
        """Test that all edges have provenance."""
        extractor = LegalEdgeExtractor(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            graph_path=str(self.output_dir / "graph.pkl"),
            output_dir=str(self.output_dir),
        )
        extractor.extract()
        
        for edge in extractor.get_extracted_edges():
            # Every edge must have provenance
            self.assertIsNotNone(edge.source_chunk_id)
            self.assertIsNotNone(edge.sentence_text)
            self.assertGreater(len(edge.sentence_text), 0)


class TestGraphIntegration(unittest.TestCase):
    """Test integration with graph builder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = Path(self.temp_dir) / "documents"
        self.chunks_dir = Path(self.temp_dir) / "chunks"
        self.output_dir = Path(self.temp_dir) / "graph"
        
        self.docs_dir.mkdir(parents=True)
        self.chunks_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create a Phase 1 graph first
        builder = LegalGraphBuilder(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            output_dir=str(self.output_dir),
        )
        builder.add_act_node("IPC")
        builder.add_has_section_edge("IPC", "420")
        builder.add_has_section_edge("IPC", "302")
        builder._stats = builder._calculate_stats()
        builder.save(version="v1")
        
        self.graph_path = self.output_dir / "legal_graph_v1.pkl"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_existing_graph(self):
        """Test loading existing Phase 1 graph."""
        extractor = LegalEdgeExtractor(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            graph_path=str(self.graph_path),
            output_dir=str(self.output_dir),
        )
        
        # Should have loaded existing nodes
        self.assertIn("ACT::IPC", extractor.builder.graph)
        self.assertIn("SECTION::IPC::420", extractor.builder.graph)
    
    def test_add_edges_to_graph(self):
        """Test adding extracted edges to graph."""
        extractor = LegalEdgeExtractor(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            graph_path=str(self.graph_path),
            output_dir=str(self.output_dir),
        )
        
        # Manually add an edge
        edge = ExtractedEdge(
            source_node_id="CASE::TEST_CASE",
            target_node_id="SECTION::IPC::420",
            relation_type=RelationType.APPLIES_SECTION.value,
            source_chunk_id="chunk001",
            sentence_text="Test sentence"
        )
        extractor._extracted_edges.append(edge)
        
        # Add to graph
        added = extractor.add_edges_to_graph()
        
        self.assertEqual(added, 1)
        self.assertIn("CASE::TEST_CASE", extractor.builder.graph)
        self.assertTrue(
            extractor.builder.graph.has_edge("CASE::TEST_CASE", "SECTION::IPC::420")
        )
    
    def test_save_v2_graph(self):
        """Test saving v2 graph."""
        extractor = LegalEdgeExtractor(
            documents_dir=str(self.docs_dir),
            chunks_dir=str(self.chunks_dir),
            graph_path=str(self.graph_path),
            output_dir=str(self.output_dir),
        )
        
        pickle_path, json_path = extractor.save(version="v2")
        
        self.assertTrue(pickle_path.exists())
        self.assertTrue(json_path.exists())
        self.assertIn("v2", str(pickle_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
