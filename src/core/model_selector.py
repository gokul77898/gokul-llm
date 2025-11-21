"""
Automatic Model Selection System

Analyzes query complexity and selects optimal model.
"""

import logging
import re
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types"""
    SMALL_LOCAL = "small_local"
    LEGAL_MODEL = "legal_model"
    LARGE_REASONING = "large_reasoning"
    RL_TRAINED = "rl_trained"
    MAMBA = "mamba"


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    LEGAL = "legal"


class ModelSelector:
    """Intelligent model selection based on query analysis"""
    
    # Legal keywords for legal model selection
    LEGAL_KEYWORDS = [
        'act', 'section', 'law', 'penalty', 'provision', 'statute',
        'employer', 'employee', 'wages', 'minimum', 'government',
        'scheduled', 'employment', 'inspector', 'authority', 'amendment',
        'clause', 'regulation', 'compliance', 'jurisdiction', 'tribunal'
    ]
    
    # Reasoning keywords for complex model selection
    REASONING_KEYWORDS = [
        'why', 'how', 'explain', 'compare', 'analyze', 'evaluate',
        'discuss', 'elaborate', 'justify', 'reasoning', 'implications'
    ]
    
    def __init__(self):
        """Initialize model selector"""
        logger.info("ModelSelector initialized")
        self.selection_log = []
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query characteristics.
        
        Args:
            query: User query string
            
        Returns:
            dict: Query analysis with word_count, complexity, has_legal_terms, etc.
        """
        query_lower = query.lower()
        words = query.split()
        word_count = len(words)
        
        # Check for legal terms
        legal_term_count = sum(1 for keyword in self.LEGAL_KEYWORDS if keyword in query_lower)
        has_legal_terms = legal_term_count > 0
        
        # Check for reasoning requirements
        reasoning_term_count = sum(1 for keyword in self.REASONING_KEYWORDS if keyword in query_lower)
        requires_reasoning = reasoning_term_count > 0
        
        # Determine complexity
        if word_count <= 5 and not requires_reasoning:
            complexity = QueryComplexity.SIMPLE
        elif word_count <= 12 and not requires_reasoning:
            complexity = QueryComplexity.MODERATE
        elif has_legal_terms:
            complexity = QueryComplexity.LEGAL
        else:
            complexity = QueryComplexity.COMPLEX
        
        # Check for questions
        is_question = query.strip().endswith('?')
        
        return {
            'word_count': word_count,
            'complexity': complexity,
            'has_legal_terms': has_legal_terms,
            'legal_term_count': legal_term_count,
            'requires_reasoning': requires_reasoning,
            'is_question': is_question,
            'query_length': len(query)
        }
    
    def pick(self, query: str, doc_count: int = 0) -> str:
        """
        Select optimal model for query.
        
        Selection logic:
        - Simple queries (≤5 words) → RL trained model (fast)
        - Legal queries → RL trained model (legal-specific)
        - Complex reasoning → Mamba (hierarchical attention)
        - Long queries (>12 words) → Mamba
        - Default → RL trained
        
        Args:
            query: User query
            doc_count: Number of retrieved documents
            
        Returns:
            str: Model name to use
        """
        analysis = self.analyze_query(query)
        
        word_count = analysis['word_count']
        complexity = analysis['complexity']
        has_legal_terms = analysis['has_legal_terms']
        requires_reasoning = analysis['requires_reasoning']
        
        # Rule 1: Simple queries → RL (fast lookup)
        if complexity == QueryComplexity.SIMPLE and word_count <= 5:
            selected_model = "rl_trained"
            reason = f"Simple query ({word_count} words) - fast RL lookup"
        
        # Rule 2: Legal queries → RL (trained on legal corpus)
        elif has_legal_terms or complexity == QueryComplexity.LEGAL:
            selected_model = "rl_trained"
            reason = f"Legal query detected ({analysis['legal_term_count']} legal terms)"
        
        # Rule 3: Complex reasoning or long queries → Mamba
        elif requires_reasoning or word_count > 12 or doc_count > 5:
            selected_model = "mamba"
            reason = f"Complex query (reasoning={requires_reasoning}, words={word_count}, docs={doc_count})"
        
        # Default: RL trained
        else:
            selected_model = "rl_trained"
            reason = "Default selection"
        
        # Log selection
        selection_record = {
            'query': query[:100],
            'model': selected_model,
            'reason': reason,
            'analysis': analysis
        }
        self.selection_log.append(selection_record)
        
        logger.info(f"Model selected: {selected_model} | Reason: {reason}")
        
        return selected_model
    
    def get_selection_log(self, limit: int = 50) -> list:
        """Get recent model selection history"""
        return self.selection_log[-limit:]
    
    def clear_log(self):
        """Clear selection log"""
        self.selection_log = []
        logger.info("Selection log cleared")


# Global singleton instance
_selector_instance: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """Get global model selector instance"""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = ModelSelector()
    return _selector_instance
