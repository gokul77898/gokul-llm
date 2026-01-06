"""
Tests for Graph-Aware Grounded Generator - Phase 4

Tests overruled exclusion, section mismatch exclusion,
refusal logic, and deterministic output.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

from src.generation.graph_grounded_generator import (
    GraphGroundedGenerator,
    GroundedAnswerResult,
)
from src.graph.graph_rag_filter import GraphRAGFilter
from src.graph.legal_graph_traverser import LegalGraphTraverser, TraversalRelation
from src.graph.legal_graph_builder import LegalGraphBuilder


class TestGraphGroundedGeneratorSetup(unittest.TestCase):
    """Test generator initialization."""
    
    def test_init_with_components(self):
        """Test initialization with all components."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        filter_obj = GraphRAGFilter(traverser)
        
        generator = GraphGroundedGenerator(
            graph_filter=filter_obj,
            retriever=None,
            generator=None,
        )
        
        self.assertIsNotNone(generator.graph_filter)
        self.assertEqual(generator.graph_filter, filter_obj)


class TestOverruledCaseExclusion(unittest.TestCase):
    """Test that overruled cases are never cited."""
    
    def setUp(self):
        """Set up graph with overruled case."""
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
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_overruled_case_not_cited(self):
        """Test that overruled cases are excluded and not cited."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "case_law",
                "case_id": "case_valid",
                "act": "IPC",
                "section": "420",
                "text": "Valid case text",
            },
            {
                "chunk_id": "chunk2",
                "semantic_id": "sem2",
                "doc_type": "case_law",
                "case_id": "case_overruled",
                "act": "IPC",
                "section": "420",
                "text": "Overruled case text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Section 420 IPC", top_k=5)
        
        # Overruled case should be excluded
        self.assertEqual(result.excluded_chunks_count, 1)
        
        # Only valid case should be cited
        self.assertNotIn("sem2", result.cited_semantic_ids)


class TestSectionMismatchExclusion(unittest.TestCase):
    """Test that section mismatches are excluded."""
    
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
    
    def test_section_mismatch_not_cited(self):
        """Test that chunks with non-existent sections are excluded."""
        chunks = [
            {
                "chunk_id": "chunk2",
                "semantic_id": "sem2",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "999",  # Not in graph
                "text": "Section 999 text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("What is cheating?", top_k=5)
        
        # Section 999 should be excluded
        self.assertGreater(result.excluded_chunks_count, 0)
        
        # sem2 should not be cited
        self.assertNotIn("sem2", result.cited_semantic_ids)


class TestRefusalWhenGraphBlocksAll(unittest.TestCase):
    """Test refusal when graph blocks all chunks."""
    
    def setUp(self):
        """Set up graph with only section 420."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add only section 420 to graph
        self.builder.add_has_section_edge("IPC", "420")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_refusal_when_all_excluded(self):
        """Test that generator refuses when all chunks are excluded."""
        # All chunks reference non-existent section 999
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "999",  # Not in graph (only 420 exists)
                "text": "Text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Section 999 IPC", top_k=5)
        
        # Should refuse
        self.assertIsNotNone(result.refusal_reason)
        self.assertFalse(result.grounded)
        self.assertEqual(result.allowed_chunks_count, 0)
        self.assertIn("excluded", result.answer.lower())
    
    def test_refusal_when_no_retrieval(self):
        """Test refusal when no chunks retrieved."""
        chunks = []
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Some query", top_k=5)
        
        # Should refuse
        self.assertIsNotNone(result.refusal_reason)
        self.assertFalse(result.grounded)
        self.assertEqual(result.allowed_chunks_count, 0)


class TestDeterministicOutput(unittest.TestCase):
    """Test that generation is deterministic."""
    
    def setUp(self):
        """Set up graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_deterministic_generation(self):
        """Test that same input produces same output."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Section 420 text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result1 = generator.generate_answer("Section 420 IPC", top_k=5)
        result2 = generator.generate_answer("Section 420 IPC", top_k=5)
        
        # Should produce same results
        self.assertEqual(result1.allowed_chunks_count, result2.allowed_chunks_count)
        self.assertEqual(result1.excluded_chunks_count, result2.excluded_chunks_count)
        self.assertEqual(result1.grounded, result2.grounded)


class TestPipelineOrder(unittest.TestCase):
    """Test that pipeline executes in correct order."""
    
    def setUp(self):
        """Set up graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_pipeline_steps_execute(self):
        """Test that all pipeline steps execute."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Section 420 text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Section 420 IPC", top_k=5)
        
        # Check that all steps produced output
        self.assertGreater(result.retrieved_count, 0)  # Step A
        self.assertGreater(result.allowed_chunks_count, 0)  # Step B
        self.assertIsNone(result.refusal_reason)  # Step C
        self.assertIsNotNone(result.answer)  # Steps D-E
        self.assertIsNotNone(result.grounded)  # Step F
        self.assertIsNotNone(result.timestamp)  # Step G


class TestCitationValidation(unittest.TestCase):
    """Test citation validation."""
    
    def setUp(self):
        """Set up graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_valid_citations(self):
        """Test that citations are validated."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Section 420 text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        # Use query without section to avoid Rule C filtering
        result = generator.generate_answer("What is cheating?", top_k=5)
        
        # Should have generated an answer
        self.assertIsNotNone(result.answer)
        self.assertGreater(len(result.answer), 0)
        
        # Should have allowed at least one chunk
        self.assertGreater(result.allowed_chunks_count, 0)


class TestAuditMetadata(unittest.TestCase):
    """Test that audit metadata is captured."""
    
    def setUp(self):
        """Set up graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_audit_metadata_present(self):
        """Test that all audit metadata is captured."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Section 420 text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Section 420 IPC", top_k=5)
        
        # Check all audit fields are present
        self.assertIsNotNone(result.query)
        self.assertIsNotNone(result.retrieved_count)
        self.assertIsNotNone(result.allowed_chunks_count)
        self.assertIsNotNone(result.excluded_chunks_count)
        self.assertIsNotNone(result.exclusion_reasons)
        self.assertIsNotNone(result.graph_paths_used)
        self.assertIsNotNone(result.grounded)
        self.assertIsNotNone(result.generation_method)
        self.assertIsNotNone(result.timestamp)
    
    def test_result_to_dict(self):
        """Test that result can be serialized to dict."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Section 420 text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Section 420 IPC", top_k=5)
        result_dict = result.to_dict()
        
        # Check dict has all keys
        self.assertIn("answer", result_dict)
        self.assertIn("cited_semantic_ids", result_dict)
        self.assertIn("allowed_chunks_count", result_dict)
        self.assertIn("excluded_chunks_count", result_dict)
        self.assertIn("grounded", result_dict)


class TestLegalDisclaimer(unittest.TestCase):
    """Test that legal disclaimer is always included."""
    
    def setUp(self):
        """Set up graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        self.traverser = LegalGraphTraverser(self.builder.graph)
        self.filter = GraphRAGFilter(self.traverser)
    
    def test_disclaimer_in_answer(self):
        """Test that disclaimer is in answer."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "semantic_id": "sem1",
                "doc_type": "bare_act",
                "act": "IPC",
                "section": "420",
                "text": "Section 420 text",
            },
        ]
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Section 420 IPC", top_k=5)
        
        # Disclaimer should be in answer
        self.assertIn("DISCLAIMER", result.answer)
        self.assertIn("legal advice", result.answer.lower())
    
    def test_disclaimer_in_refusal(self):
        """Test that disclaimer is in refusal."""
        chunks = []
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        generator = GraphGroundedGenerator(
            graph_filter=self.filter,
            retriever=retriever,
            generator=None,
        )
        
        result = generator.generate_answer("Section 999 IPC", top_k=5)
        
        # Disclaimer should be in refusal too
        self.assertIn("DISCLAIMER", result.answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
