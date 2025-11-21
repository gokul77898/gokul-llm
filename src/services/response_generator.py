"""
Zero-Hallucination Response Generator
Only uses retrieved document content
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GeneratedResponse:
    """Clean response format"""
    answer: str
    model_used: str
    confidence: float
    sources: List[Dict[str, Any]]
    contains_answer: bool

class ResponseGenerator:
    """
    Zero-hallucination response generator
    
    Rules:
    - Only use extracted text from retrieved chunks
    - Never generate text not in source
    - If no answer found → return "Not available in uploaded documents."
    """
    
    def generate_from_documents(
        self,
        query: str,
        documents: List[Any],
        model_name: str
    ) -> GeneratedResponse:
        """
        Generate response strictly from document content
        
        Returns:
            GeneratedResponse with answer or "not available" message
        """
        if not documents or len(documents) == 0:
            return self._no_documents_response(model_name)
        
        # Extract relevant content from documents
        relevant_content = self._extract_relevant_content(query, documents)
        
        if not relevant_content:
            return self._no_answer_response(model_name)
        
        # Build answer from document content only
        answer = self._build_answer_from_content(query, relevant_content)
        
        # Verify answer quality
        if len(answer.strip()) < 20 or self._is_nonsense(answer):
            return self._no_answer_response(model_name)
        
        # Prepare sources
        sources = self._prepare_sources(documents[:3])
        
        return GeneratedResponse(
            answer=answer,
            model_used=model_name,
            confidence=self._calculate_confidence(relevant_content, documents),
            sources=sources,
            contains_answer=True
        )
    
    def _extract_relevant_content(self, query: str, documents: List[Any]) -> List[str]:
        """Extract only relevant sentences from documents"""
        relevant_sentences = []
        query_keywords = set(query.lower().split())
        
        for doc in documents[:5]:  # Top 5 documents
            content = getattr(doc, 'content', str(doc))
            
            # Split into sentences
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # Check relevance
                sentence_words = set(sentence.lower().split())
                overlap = len(query_keywords & sentence_words)
                
                if overlap >= 2 or self._contains_definition(sentence):
                    relevant_sentences.append(sentence)
        
        return relevant_sentences[:10]  # Top 10 relevant sentences
    
    def _build_answer_from_content(self, query: str, content_list: List[str]) -> str:
        """Build answer using ONLY document content"""
        
        # If asking for definition
        if 'definition' in query.lower() or 'define' in query.lower() or 'what is' in query.lower():
            for content in content_list:
                if 'means' in content.lower() or 'defined as' in content.lower():
                    return f"According to the Act: {content.strip()}."
        
        # If asking about specific term
        query_terms = self._extract_key_terms(query)
        for term in query_terms:
            for content in content_list:
                if term.lower() in content.lower():
                    # Found relevant content
                    return f"According to the uploaded document: {content.strip()}."
        
        # General answer from top relevant content
        if content_list:
            top_content = content_list[0]
            return f"Based on the Act: {top_content.strip()}."
        
        return "Not available in uploaded documents."
    
    def _contains_definition(self, sentence: str) -> bool:
        """Check if sentence contains a definition"""
        definition_markers = ['means', 'defined as', 'refers to', 'includes', 'shall mean']
        return any(marker in sentence.lower() for marker in definition_markers)
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Remove common words
        stop_words = {'what', 'is', 'the', 'of', 'in', 'a', 'an', 'how', 'does', 'can'}
        words = query.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Also check for phrases
        if 'appropriate government' in query.lower():
            key_terms.append('appropriate government')
        if 'scheduled employment' in query.lower():
            key_terms.append('scheduled employment')
        
        return key_terms
    
    def _is_nonsense(self, answer: str) -> bool:
        """Check if answer is nonsensical"""
        # Check for incomplete sentences
        if not answer.strip().endswith('.'):
            return True
        
        # Check for too many special characters
        special_char_count = sum(1 for c in answer if not c.isalnum() and c not in [' ', '.', ',', '-'])
        if special_char_count > len(answer) * 0.1:
            return True
        
        return False
    
    def _calculate_confidence(self, relevant_content: List[str], documents: List[Any]) -> float:
        """Calculate confidence based on content quality"""
        if not relevant_content:
            return 0.1
        
        # Base confidence
        confidence = 0.7
        
        # Bonus for multiple relevant sentences
        if len(relevant_content) >= 3:
            confidence += 0.1
        
        # Bonus for document scores
        if documents and hasattr(documents[0], 'score'):
            avg_score = sum(getattr(d, 'score', 0) for d in documents[:3]) / 3
            confidence += avg_score * 0.2
        
        return min(0.95, confidence)
    
    def _prepare_sources(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Prepare source citations"""
        sources = []
        for i, doc in enumerate(documents[:3], 1):
            sources.append({
                'doc_number': i,
                'content': getattr(doc, 'content', str(doc))[:200] + '...',
                'score': getattr(doc, 'score', 0.0),
                'metadata': getattr(doc, 'metadata', {})
            })
        return sources
    
    def _no_documents_response(self, model_name: str) -> GeneratedResponse:
        """Response when no documents retrieved"""
        return GeneratedResponse(
            answer="No relevant documents found in the database. Please ensure documents are uploaded and indexed.",
            model_used=model_name,
            confidence=0.0,
            sources=[],
            contains_answer=False
        )
    
    def _no_answer_response(self, model_name: str) -> GeneratedResponse:
        """Response when documents don't contain answer"""
        return GeneratedResponse(
            answer="The uploaded documents do not contain an answer to this question.",
            model_used=model_name,
            confidence=0.0,
            sources=[],
            contains_answer=False
        )
