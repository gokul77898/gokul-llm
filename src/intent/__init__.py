"""
Intent Classification Module - Phase 7

Rule-based legal query intent classification.

NO ML.
NO LLM.
NO EMBEDDINGS.
DETERMINISTIC ONLY.
"""

from .legal_intent_classifier import (
    LegalIntentClassifier,
    IntentResult,
    IntentClass,
    IntentRefusalReason,
)

__all__ = [
    "LegalIntentClassifier",
    "IntentResult",
    "IntentClass",
    "IntentRefusalReason",
]
