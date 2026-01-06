"""
API Schemas - Phase 6

Defines response schemas for the Legal Reasoning API.

STRICT CONTRACTS.
NO OPTIONAL FIELDS WITHOUT DEFAULTS.
FULLY SERIALIZABLE.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class RefusalReason(Enum):
    """
    Reasons for refusing to answer a query.
    """
    ALL_CHUNKS_EXCLUDED = "all_chunks_excluded"
    NO_RETRIEVAL = "no_retrieval"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"
    GRAPH_FILTER_BLOCKED = "graph_filter_blocked"
    UNKNOWN = "unknown"


@dataclass
class LegalAnswerResponse:
    """
    Complete legal answer response.
    
    Either contains a full answer with explanation,
    or a refusal with reason.
    """
    # Core response
    query: str
    answered: bool
    
    # Answer fields (populated if answered=True)
    answer: str = ""
    statutory_basis: List[str] = field(default_factory=list)
    judicial_interpretations: List[str] = field(default_factory=list)
    applied_precedents: List[str] = field(default_factory=list)
    supporting_precedents: List[str] = field(default_factory=list)
    excluded_precedents: List[str] = field(default_factory=list)
    explanation_text: str = ""
    
    # Refusal fields (populated if answered=False)
    refusal_reason: Optional[str] = None
    
    # Audit metadata (always populated)
    retrieved_count: int = 0
    allowed_chunks_count: int = 0
    excluded_chunks_count: int = 0
    cited_count: int = 0
    grounded: bool = False
    
    # Timestamps
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "answered": self.answered,
            "answer": self.answer,
            "statutory_basis": self.statutory_basis,
            "judicial_interpretations": self.judicial_interpretations,
            "applied_precedents": self.applied_precedents,
            "supporting_precedents": self.supporting_precedents,
            "excluded_precedents": self.excluded_precedents,
            "explanation_text": self.explanation_text,
            "refusal_reason": self.refusal_reason,
            "retrieved_count": self.retrieved_count,
            "allowed_chunks_count": self.allowed_chunks_count,
            "excluded_chunks_count": self.excluded_chunks_count,
            "cited_count": self.cited_count,
            "grounded": self.grounded,
            "timestamp": self.timestamp,
        }
