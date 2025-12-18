"""
RAG Validation - Phase R3: Retrieval Validation & Filtering

Prevents incorrect, weak, or unsafe evidence from reaching generation.

Implements:
- Rule-based validation
- Evidence thresholds
- Hard refusal triggers

Phase R3 does NOT:
- Generate answers
- Call decoder/LLM
- Modify retrieval logic

Failure = refusal, never guess.
"""

from .threshold import EvidenceThreshold, ThresholdConfig
from .statute_validator import StatuteValidator, StatuteMatch
from .evidence_filter import EvidenceFilter, FilterResult
from .validator import RetrievalValidator, ValidationResult, RefusalReason

__all__ = [
    "EvidenceThreshold",
    "ThresholdConfig",
    "StatuteValidator",
    "StatuteMatch",
    "EvidenceFilter",
    "FilterResult",
    "RetrievalValidator",
    "ValidationResult",
    "RefusalReason",
]
