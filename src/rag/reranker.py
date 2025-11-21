"""Cross-Encoder Reranker for Retrieved Documents"""

import logging
from typing import List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieved documents using cross-encoder or embedding fallback"""
    
    def __init__(self):
        """Initialize reranker with cross-encoder or embedding fallback"""
        self.model = None
        self.use_cross_encoder = False
        
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.use_cross_encoder = True
            logger.info("Cross-encoder reranker initialized")
        except Exception as e:
            logger.warning(f"Cross-encoder not available, using embedding fallback: {e}")
            self.use_cross_encoder = False
    
    def rerank(self, query: str, documents: List[Any], top_k: int = 3) -> List[Tuple[Any, float]]:
        """
        Rerank documents by relevance to query
        
        Args:
            query: User query
            documents: List of document objects with 'content' attribute
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        try:
            if self.use_cross_encoder and self.model:
                return self._rerank_with_cross_encoder(query, documents, top_k)
            else:
                return self._rerank_with_embedding_fallback(query, documents, top_k)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order with default scores
            return [(doc, 0.5) for doc in documents[:top_k]]
    
    def _rerank_with_cross_encoder(self, query: str, documents: List[Any], top_k: int) -> List[Tuple[Any, float]]:
        """Rerank using cross-encoder model"""
        pairs = []
        for doc in documents:
            content = getattr(doc, 'content', str(doc))
            pairs.append([query, content[:512]])  # Truncate for efficiency
        
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]
    
    def _rerank_with_embedding_fallback(self, query: str, documents: List[Any], top_k: int) -> List[Tuple[Any, float]]:
        """Fallback reranking using simple word overlap"""
        query_words = set(query.lower().split())
        
        doc_scores = []
        for doc in documents:
            content = getattr(doc, 'content', str(doc)).lower()
            doc_words = set(content.split())
            
            # Jaccard similarity
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            score = intersection / union if union > 0 else 0.0
            
            doc_scores.append((doc, score))
        
        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]
    
    def extract_key_excerpts(self, query: str, documents: List[Tuple[Any, float]], max_excerpts: int = 3) -> List[dict]:
        """
        Extract most relevant sentence-level excerpts from top documents
        
        Returns:
            List of excerpt dicts with doc_id, content, score, metadata
        """
        excerpts = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc, doc_score in documents[:max_excerpts]:
            content = getattr(doc, 'content', str(doc))
            metadata = getattr(doc, 'metadata', {})
            
            # Split into sentences
            sentences = self._split_sentences(content)
            
            # Find best sentence(s) matching query
            best_sentence = ""
            best_score = 0.0
            
            for sentence in sentences:
                if len(sentence.split()) < 5:  # Skip too short
                    continue
                
                sentence_lower = sentence.lower()
                sentence_words = set(sentence_lower.split())
                
                # Word overlap score
                overlap = len(query_words & sentence_words)
                sent_score = overlap / len(query_words) if query_words else 0.0
                
                if sent_score > best_score:
                    best_score = sent_score
                    best_sentence = sentence
            
            if best_sentence:
                excerpts.append({
                    'doc_id': metadata.get('doc_id', f'doc_{len(excerpts)}'),
                    'content': best_sentence.strip(),
                    'score': float(doc_score),
                    'page': metadata.get('page', 'N/A'),
                    'metadata': metadata
                })
        
        return excerpts
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter"""
        import re
        # Split on period, question mark, exclamation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
