"""
Tests for API Intent Integration - Phase 7

Tests that intent classification is properly integrated into API
and that blocked intents don't trigger downstream pipeline execution.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

from src.api.legal_reasoning_api import LegalReasoningAPI
from src.graph.legal_graph_traverser import LegalGraphTraverser
from src.graph.legal_graph_builder import LegalGraphBuilder


                "section": "420",
                "text": "Section 420 text",
            },
        ]


class TestIntentIntegration(unittest.TestCase):
    """Test intent classifier integration into API."""
    
    def setUp(self):
        """Set up API with mock components."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        traverser = LegalGraphTraverser(self.builder.graph)
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        self.retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=self.retriever,
            generator=None,
        )
    
    def test_api_has_intent_classifier(self):
        """Test that API has intent classifier."""
        self.assertTrue(hasattr(self.api, 'intent_classifier'))
        self.assertIsNotNone(self.api.intent_classifier)


class TestAllowedIntentProceedsToRetrieval(unittest.TestCase):
    """Test that allowed intents proceed to retrieval."""
    
    def setUp(self):
        """Set up API with mock components."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        traverser = LegalGraphTraverser(self.builder.graph)
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        self.retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=self.retriever,
            generator=None,
        )
    
    def test_factual_query_proceeds_to_retrieval(self):
        """Test that factual queries proceed to retrieval."""
        response = self.api.answer_query("What is Section 420 IPC?", top_k=5)
        
        # Retriever should have been called
        self.assertTrue(self.retriever.retrieve_called)
        self.assertGreater(self.retriever.retrieve_count, 0)
    
    def test_definitional_query_proceeds_to_retrieval(self):
        """Test that definitional queries proceed to retrieval."""
        self.retriever.retrieve_called = False
        self.retriever.retrieve_count = 0
        
        response = self.api.answer_query("Define cheating", top_k=5)
        
        # Retriever should have been called
        self.assertTrue(self.retriever.retrieve_called)
        self.assertGreater(self.retriever.retrieve_count, 0)
    
    def test_procedural_query_proceeds_to_retrieval(self):
        """Test that procedural queries proceed to retrieval."""
        self.retriever.retrieve_called = False
        self.retriever.retrieve_count = 0
        
        response = self.api.answer_query("How to file a complaint?", top_k=5)
        
        # Retriever should have been called
        self.assertTrue(self.retriever.retrieve_called)
        self.assertGreater(self.retriever.retrieve_count, 0)


class TestBlockedIntentDoesNotProceedToRetrieval(unittest.TestCase):
    """Test that blocked intents don't proceed to retrieval."""
    
    def setUp(self):
        """Set up API with mock components."""
        self.builder = LegalGraphBuilder(
            documents_dir="/tmp/docs",
            chunks_dir="/tmp/chunks",
            output_dir="/tmp/graph",
        )
        
        self.builder.add_has_section_edge("IPC", "420")
        
        traverser = LegalGraphTraverser(self.builder.graph)
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        self.retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=self.retriever,
            generator=None,
        )
    
    def test_advisory_query_does_not_proceed_to_retrieval(self):
        """Test that advisory queries are blocked before retrieval."""
        response = self.api.answer_query("Should I plead guilty?", top_k=5)
        
        # Retriever should NOT have been called
        self.assertFalse(self.retriever.retrieve_called)
        self.assertEqual(self.retriever.retrieve_count, 0)
        
        # Response should be refusal
        self.assertFalse(response.answered)
        self.assertIsNotNone(response.refusal_reason)
    
    def test_strategic_query_does_not_proceed_to_retrieval(self):
        """Test that strategic queries are blocked before retrieval."""
        response = self.api.answer_query("How to avoid conviction?", top_k=5)
        
        # Retriever should NOT have been called
        self.assertFalse(self.retriever.retrieve_called)
        self.assertEqual(self.retriever.retrieve_count, 0)
        
        # Response should be refusal
        self.assertFalse(response.answered)
        self.assertIsNotNone(response.refusal_reason)
    
    def test_speculative_query_does_not_proceed_to_retrieval(self):
        """Test that speculative queries are blocked before retrieval."""
        response = self.api.answer_query("Will I win this case?", top_k=5)
        
        # Retriever should NOT have been called
        self.assertFalse(self.retriever.retrieve_called)
        self.assertEqual(self.retriever.retrieve_count, 0)
        
        # Response should be refusal
        self.assertFalse(response.answered)
        self.assertIsNotNone(response.refusal_reason)


class TestIntentRefusalResponse(unittest.TestCase):
    """Test intent refusal response structure."""
    
    def setUp(self):
        """Set up API with mock components."""
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
    
    def test_advisory_refusal_has_correct_reason(self):
        """Test that advisory refusal has correct reason."""
        response = self.api.answer_query("Should I plead guilty?", top_k=5)
        
        self.assertFalse(response.answered)
        self.assertEqual(response.refusal_reason, "legal_advice_not_provided")
        self.assertIn("legal advice", response.answer.lower())
    
    def test_strategic_refusal_has_correct_reason(self):
        """Test that strategic refusal has correct reason."""
        response = self.api.answer_query("How to avoid conviction?", top_k=5)
        
        self.assertFalse(response.answered)
        self.assertEqual(response.refusal_reason, "strategic_assistance_blocked")
        self.assertIn("strategic", response.answer.lower())
    
    def test_speculative_refusal_has_correct_reason(self):
        """Test that speculative refusal has correct reason."""
        response = self.api.answer_query("Will I win?", top_k=5)
        
        self.assertFalse(response.answered)
        self.assertEqual(response.refusal_reason, "speculative_query_blocked")
        self.assertIn("predict", response.answer.lower())


class TestIntentRefusalMetadata(unittest.TestCase):
    """Test that intent refusal responses have correct metadata."""
    
    def setUp(self):
        """Set up API with mock components."""
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
    
    def test_intent_refusal_has_zero_counts(self):
        """Test that intent refusal has zero retrieval counts."""
        response = self.api.answer_query("Should I plead guilty?", top_k=5)
        
        # Should have zero counts since retrieval was never called
        self.assertEqual(response.retrieved_count, 0)
        self.assertEqual(response.allowed_chunks_count, 0)
        self.assertEqual(response.excluded_chunks_count, 0)
        self.assertEqual(response.cited_count, 0)
        self.assertFalse(response.grounded)
    
    def test_intent_refusal_has_timestamp(self):
        """Test that intent refusal has timestamp."""
        response = self.api.answer_query("Should I plead guilty?", top_k=5)
        
        self.assertIsNotNone(response.timestamp)
        self.assertGreater(len(response.timestamp), 0)


class TestNoDownstreamPipelineExecution(unittest.TestCase):
    """Test that blocked intents don't execute downstream pipeline."""
    
    def setUp(self):
        """Set up API with mock components."""
        graph = nx.DiGraph()
        traverser = LegalGraphTraverser(graph)
        # Real retriever required
        from src.rag.retrieval.retriever import LegalRetriever
        self.retriever = LegalRetriever(
            chunks_dir="data/rag/chunks",
            chromadb_dir="data/rag/chromadb"
        )
        
        self.api = LegalReasoningAPI(
            traverser=traverser,
            retriever=self.retriever,
            generator=None,
        )
    
    def test_no_precedent_extraction_for_blocked_intent(self):
        """Test that precedent extraction is not called for blocked intents."""
        # Advisory query should be blocked
        response = self.api.answer_query("Should I plead guilty?", top_k=5)
        
        # Should be blocked
        self.assertFalse(response.answered)
        
        # Should have no precedent data
        self.assertEqual(len(response.statutory_basis), 0)
        self.assertEqual(len(response.judicial_interpretations), 0)
        self.assertEqual(len(response.applied_precedents), 0)
        self.assertEqual(len(response.supporting_precedents), 0)
        self.assertEqual(len(response.excluded_precedents), 0)
        self.assertEqual(response.explanation_text, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
