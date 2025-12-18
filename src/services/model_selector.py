"""
Clean Auto Model Selector
No hallucinations, clear rules
"""

import re
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelSelection:
    """Model selection result"""
    selected_model: str
    reason: str
    confidence: float

class AutoModelSelector:
    """
    Clean auto-selector with explicit rules:
    - Query length < 10 words → TLH (fast)
    - Legal keywords (Act, section, law, punishment) → RL model
    - Multiple document aggregation → Transformer-based model
    """
    
    LEGAL_KEYWORDS = [
        'act', 'section', 'law', 'punishment', 'penalty', 'clause',
        'provision', 'statute', 'regulation', 'amendment', 'ordinance',
        'definition', 'employer', 'employee', 'wages', 'minimum wages',
        'appropriate government', 'scheduled employment', 'inspector',
        'authority', 'notification', 'prescribed', 'tribunal'
    ]
    
    AGGREGATION_KEYWORDS = [
        'all', 'list', 'summarize', 'overview', 'compare', 'difference',
        'multiple', 'various', 'different', 'explain all', 'what are the'
    ]
    
    def select_model(self, query: str, retrieved_docs: int = 0) -> ModelSelection:
        """
        Select best model based on query characteristics
        
        Rules:
        1. Short queries (< 10 words) → TLH (fast lookup)
        2. Legal queries → RL model (trained on legal text)
        3. Aggregation queries → Transformer-based model
        4. Default → RL model
        """
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())
        
        # Rule 1: Short queries → TLH
        if word_count < 10 and retrieved_docs <= 3:
            return ModelSelection(
                selected_model="rl_trained",  # Fast RL for simple queries
                reason="Short query - using fast RL lookup",
                confidence=0.95
            )
        
        # Rule 2: Legal queries → RL model
        if self._is_legal_query(query_lower):
            return ModelSelection(
                selected_model="rl_trained",
                reason="Legal query detected - using RL model trained on legal text",
                confidence=0.92
            )
        
        # Rule 3: Aggregation queries → Transformer-based model
        if self._requires_aggregation(query_lower):
            return ModelSelection(
                selected_model="transformer",
                reason="Multi-document aggregation required - using Transformer-based model",
                confidence=0.88
            )
        
        # Rule 4: Complex queries with many docs → Transformer-based model
        if retrieved_docs > 5:
            return ModelSelection(
                selected_model="transformer",
                reason="Multiple documents retrieved - using Transformer-based model for synthesis",
                confidence=0.85
            )
        
        # Default: RL model
        return ModelSelection(
            selected_model="rl_trained",
            reason="Default selection - RL model for general legal queries",
            confidence=0.80
        )
    
    def _is_legal_query(self, query: str) -> bool:
        """Check if query contains legal keywords"""
        return any(keyword in query for keyword in self.LEGAL_KEYWORDS)
    
    def _requires_aggregation(self, query: str) -> bool:
        """Check if query requires multi-document aggregation"""
        return any(keyword in query for keyword in self.AGGREGATION_KEYWORDS)
    
    def _detect_sections(self, query: str) -> List[str]:
        """Detect section references like 12(3), 7A, 11B"""
        patterns = [
            r'\d+\([a-z0-9]+\)',  # 12(3)
            r'\d+[A-Z]',           # 7A
            r'section\s+\d+',      # section 12
        ]
        
        sections = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            sections.extend(matches)
        
        return sections
