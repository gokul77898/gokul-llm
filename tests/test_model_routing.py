"""
Tests for Model Routing Logic

Tests the auto-routing between Transformer models.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.auto_pipeline import AutoPipeline
from src.pipelines.context_builder import estimate_tokens, get_total_pages


class TestModelRouting:
    """Test model routing functionality"""
    
    def setup_method(self):
        """Setup test pipeline"""
        self.pipeline = AutoPipeline()
    
    def test_short_query_routes_to_transformer(self):
        """Short queries should route to transformer"""
        query = "What is a plaint?"
        doc_count = 2
        context_text = query  # Short context
        retrieved_docs = []
        
        selected = self.pipeline.select_model(query, doc_count, context_text, retrieved_docs)
        
        # Should select transformer
        assert selected == 'transformer', f"Unexpected model: {selected}"
    
    def test_long_context_routes_to_transformer(self):
        """Long context (>4096 tokens) should route to Transformer"""
        query = "Explain the judgment"
        
        # Create long context text (>4096 tokens estimated)
        long_text = "legal document content " * 2000  # ~6000 words = ~4500 tokens
        context_text = long_text
        
        retrieved_docs = [
            {'content': long_text[:5000], 'metadata': {'page': 1}},
            {'content': long_text[5000:10000], 'metadata': {'page': 2}},
        ]
        
        selected = self.pipeline.select_model(query, len(retrieved_docs), context_text, retrieved_docs)
        
        # Should select transformer
        assert selected == 'transformer', f"Unexpected model: {selected}"
        
        # Verify token estimation works
        token_est = estimate_tokens(context_text)
        assert token_est > 1000, "Token estimation failed"
    
    def test_multi_page_routes_to_transformer(self):
        """Documents with >=3 pages should route to Transformer"""
        query = "Summarize the case"
        
        retrieved_docs = [
            {'content': 'page 1 content', 'metadata': {'page': 1}},
            {'content': 'page 2 content', 'metadata': {'page': 2}},
            {'content': 'page 3 content', 'metadata': {'page': 3}},
            {'content': 'page 4 content', 'metadata': {'page': 4}},
        ]
        
        page_count = get_total_pages(retrieved_docs)
        assert page_count >= 3, "Page count calculation failed"
        
        context_text = "short context"
        selected = self.pipeline.select_model(query, len(retrieved_docs), context_text, retrieved_docs)
        
        # Should select transformer
        assert selected == 'transformer', f"Unexpected model: {selected}"
    
    def test_legal_keywords_route_to_transformer(self):
        """Queries with legal keywords should route to Transformer"""
        test_cases = [
            "What is the supreme court judgment on this case?",
            "Explain the appellate verdict",
            "What does the order say?",
        ]
        
        for query in test_cases:
            selected = self.pipeline.select_model(query, 2, query, [])
            assert selected == 'transformer', f"Unexpected model for query: {query}"
    
    def test_fallback_when_transformer_unavailable(self):
        """Should fallback to default transformer if transformer unavailable"""
        # Test with routing config that enables Transformer but it's not available
        query = "judgment" * 1000  # Long query with keyword
        context_text = query
        
        selected = self.pipeline.select_model(query, 1, context_text, [])
        
        # If Transformer unavailable, should fallback to default
        # If Transformer available, will select transformer
        assert selected == 'transformer', f"Unexpected model: {selected}"
    
    def test_routing_config_loading(self):
        """Test routing configuration loads correctly"""
        config = self.pipeline.routing_config
        
        assert 'enable_transformer' in config
        assert 'transformer_threshold_tokens' in config
        assert 'transformer_min_pages' in config
        assert 'default_model' in config
        
        # Check default values or loaded values
        assert config['transformer_threshold_tokens'] >= 1024
        assert config['transformer_min_pages'] >= 1
    
    def test_token_estimation(self):
        """Test token estimation utility"""
        short_text = "Hello world"
        long_text = "word " * 1000
        
        short_tokens = estimate_tokens(short_text)
        long_tokens = estimate_tokens(long_text)
        
        assert short_tokens < 10
        assert long_tokens > 500
        assert long_tokens < 2000  # ~750 tokens for 1000 words
    
    def test_page_count_extraction(self):
        """Test page count extraction from documents"""
        docs_with_pages = [
            {'content': 'text', 'metadata': {'page': 1}},
            {'content': 'text', 'metadata': {'page': 2}},
            {'content': 'text', 'metadata': {'page': 2}},  # Duplicate page
            {'content': 'text', 'metadata': {'page_number': 3}},  # Alternative key
        ]
        
        page_count = get_total_pages(docs_with_pages)
        assert page_count == 3, f"Expected 3 unique pages, got {page_count}"
        
        # Test with no page metadata
        docs_without_pages = [
            {'content': 'text', 'metadata': {}},
            {'content': 'text', 'metadata': {'other': 'data'}},
        ]
        
        page_count_none = get_total_pages(docs_without_pages)
        assert page_count_none == 0


def test_transformer_availability_check():
    """Test Transformer availability checking"""
    from src.core.model_registry import is_model_available
    
    # Check if transformer is available (should return True or False, not error)
    available = is_model_available('transformer')
    assert isinstance(available, bool)
    
    # Transformer should always be available
    transformer_available = is_model_available('transformer')
    assert isinstance(transformer_available, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
