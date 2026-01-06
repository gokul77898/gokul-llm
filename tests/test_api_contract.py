"""
Tests for Legal Reasoning API - Phase 6

Tests API contract, response structure, determinism,
and constraint compliance.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

from src.api.legal_reasoning_api import LegalReasoningAPI
from src.api.schemas import LegalAnswerResponse, RefusalReason
from src.graph.legal_graph_traverser import LegalGraphTraverser
from src.graph.legal_graph_builder import LegalGraphBuilder




class TestAPISetup(unittest.TestCase):
    """Test API initialization."""
    
    def test_init_with_components(self):
        """Test initialization with all components."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        
        # Real retriever required - will be imported in test setup
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
        
        self.assertIsNotNone(api.graph_filter)
        self.assertIsNotNone(api.grounded_generator)
        self.assertIsNotNone(api.precedent_extractor)
        self.assertIsNotNone(api.precedent_labeler)
        self.assertIsNotNone(api.explanation_assembler)


class TestAnsweredResponseStructure(unittest.TestCase):
    """Test answered response structure."""
    
    def setUp(self):
        """Set up API with graph."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add section to graph
        self.builder.add_has_section_edge("IPC", "420")
        
        traverser = LegalGraphTraverser(self.builder.graph)
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
    
    def test_answered_response_has_all_fields(self):
        """Test that answered response has all required fields."""
        response = self.api.answer_query("What is cheating?", top_k=5)
        
        # Core fields
        self.assertIsNotNone(response.query)
        self.assertIsInstance(response.answered, bool)
        
        # Answer fields
        self.assertIsNotNone(response.answer)
        self.assertIsInstance(response.statutory_basis, list)
        self.assertIsInstance(response.judicial_interpretations, list)
        self.assertIsInstance(response.applied_precedents, list)
        self.assertIsInstance(response.supporting_precedents, list)
        self.assertIsInstance(response.excluded_precedents, list)
        self.assertIsInstance(response.explanation_text, str)
        
        # Audit metadata
        self.assertIsInstance(response.retrieved_count, int)
        self.assertIsInstance(response.allowed_chunks_count, int)
        self.assertIsInstance(response.excluded_chunks_count, int)
        self.assertIsInstance(response.cited_count, int)
        self.assertIsInstance(response.grounded, bool)
        
        # Timestamp
        self.assertIsNotNone(response.timestamp)
    
    def test_answered_response_answered_true(self):
        """Test that successful response has answered=True."""
        response = self.api.answer_query("What is cheating?", top_k=5)
        
        self.assertTrue(response.answered)
        self.assertIsNone(response.refusal_reason)


class TestRefusalResponseStructure(unittest.TestCase):
    """Test refusal response structure."""
    
    def setUp(self):
        """Set up API with graph that will cause refusal."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        # Add only section 420
        self.builder.add_has_section_edge("IPC", "420")
        
        traverser = LegalGraphTraverser(self.builder.graph)
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
    
    def test_refusal_response_has_all_fields(self):
        """Test that refusal response has all required fields."""
        response = self.api.answer_query("What is Section 999?", top_k=5)
        
        # Core fields
        self.assertIsNotNone(response.query)
        self.assertIsInstance(response.answered, bool)
        
        # Refusal fields
        self.assertIsNotNone(response.refusal_reason)
        
        # Audit metadata
        self.assertIsInstance(response.retrieved_count, int)
        self.assertIsInstance(response.allowed_chunks_count, int)
        self.assertIsInstance(response.excluded_chunks_count, int)
        
        # Timestamp
        self.assertIsNotNone(response.timestamp)
    
    def test_refusal_response_answered_false(self):
        """Test that refusal response has answered=False."""
        response = self.api.answer_query("What is Section 999?", top_k=5)
        
        self.assertFalse(response.answered)
        self.assertIsNotNone(response.refusal_reason)
        self.assertFalse(response.grounded)


class TestNoMissingFields(unittest.TestCase):
    """Test that no fields are missing in responses."""
    
    def setUp(self):
        """Set up API."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
    
    def test_to_dict_has_all_keys(self):
        """Test that to_dict() includes all expected keys."""
        response = self.api.answer_query("Test query", top_k=5)
        
        result_dict = response.to_dict()
        
        expected_keys = [
            "query",
            "answered",
            "answer",
            "statutory_basis",
            "judicial_interpretations",
            "applied_precedents",
            "supporting_precedents",
            "excluded_precedents",
            "explanation_text",
            "refusal_reason",
            "retrieved_count",
            "allowed_chunks_count",
            "excluded_chunks_count",
            "cited_count",
            "grounded",
            "timestamp",
        ]
        
        for key in expected_keys:
            self.assertIn(key, result_dict)


class TestDeterministicOutput(unittest.TestCase):
    """Test that API output is deterministic."""
    
    def setUp(self):
        """Set up API."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        traverser = LegalGraphTraverser(self.builder.graph)
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
    
    def test_deterministic_response(self):
        """Test that same query produces same response structure."""
        query = "What is cheating?"
        
        response1 = self.api.answer_query(query, top_k=5)
        response2 = self.api.answer_query(query, top_k=5)
        
        # Should have same answered status
        self.assertEqual(response1.answered, response2.answered)
        
        # Should have same counts
        self.assertEqual(response1.retrieved_count, response2.retrieved_count)
        self.assertEqual(response1.allowed_chunks_count, response2.allowed_chunks_count)
        self.assertEqual(response1.excluded_chunks_count, response2.excluded_chunks_count)
        
        # Should have same precedent lists
        self.assertEqual(response1.statutory_basis, response2.statutory_basis)
        self.assertEqual(response1.judicial_interpretations, response2.judicial_interpretations)
        self.assertEqual(response1.applied_precedents, response2.applied_precedents)


class TestNoGraphAccessInAPI(unittest.TestCase):
    """Test that API doesn't directly access graph."""
    
    def test_api_uses_components_not_graph(self):
        """Test that API uses components, not direct graph access."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
        
        # API should not have direct graph reference
        self.assertFalse(hasattr(api, 'graph'))
        
        # API should have component references
        self.assertTrue(hasattr(api, 'graph_filter'))
        self.assertTrue(hasattr(api, 'grounded_generator'))
        self.assertTrue(hasattr(api, 'precedent_extractor'))
        self.assertTrue(hasattr(api, 'precedent_labeler'))
        self.assertTrue(hasattr(api, 'explanation_assembler'))


class TestNoLLMAccessInAPI(unittest.TestCase):
    """Test that API doesn't directly call LLMs."""
    
    def setUp(self):
        """Set up API."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
        
        # API should not have LLM client
        self.assertFalse(hasattr(api, 'llm'))
        self.assertFalse(hasattr(api, 'model'))
        self.assertFalse(hasattr(api, 'client'))


class TestRefusalReasonEnum(unittest.TestCase):
    """Test RefusalReason enum."""
    
    def test_refusal_reason_values(self):
        """Test that RefusalReason has expected values."""
        expected_values = [
            "all_chunks_excluded",
            "no_retrieval",
            "citation_validation_failed",
            "graph_filter_blocked",
            "unknown",
        ]
        
        for value in expected_values:
            # Should be able to get enum by value
            reason = RefusalReason(value)
            self.assertEqual(reason.value, value)


class TestAPIStats(unittest.TestCase):
    """Test API statistics."""
    
    def test_get_api_stats(self):
        """Test that API stats are available."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        api = LegalReasoningAPI(
            traverser=traverser,
            retriever=retriever,
            generator=None,
        )
        
        stats = api.get_api_stats()
        
        self.assertIn("components", stats)
        self.assertIn("pipeline", stats)
        
        # Check components
        self.assertIn("graph_filter", stats["components"])
        self.assertIn("grounded_generator", stats["components"])
        self.assertIn("precedent_extractor", stats["components"])
        self.assertIn("precedent_labeler", stats["components"])
        self.assertIn("explanation_assembler", stats["components"])
        
        # Check pipeline
        self.assertEqual(len(stats["pipeline"]), 4)


class TestResponseSerialization(unittest.TestCase):
    """Test response serialization."""
    
    def test_response_to_dict_serializable(self):
        """Test that response can be serialized to dict."""
        response = LegalAnswerResponse(
            query="Test query",
            answered=True,
            answer="Test answer",
            statutory_basis=["SECTION::IPC::420"],
            timestamp="2024-01-01T00:00:00",
        )
        
        result = response.to_dict()
        
        # Should be a dict
        self.assertIsInstance(result, dict)
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
