"""
Explanation Module - Phase 5: Precedent Extraction, Labeling & Assembly

Produces STRUCTURED EXPLANATIONS ONLY.
NO NATURAL LANGUAGE GENERATION (except fixed templates in 5C).
NO LLM USAGE.
"""

from .precedent_extractor import (
    PrecedentPathExtractor,
    PrecedentExplanation,
)

from .precedent_labeler import (
    PrecedentLabeler,
    LabeledPrecedent,
)

from .explanation_assembler import (
    ExplanationAssembler,
    LegalExplanation,
)

__all__ = [
    # Phase 5A
    "PrecedentPathExtractor",
    "PrecedentExplanation",
    # Phase 5B
    "PrecedentLabeler",
    "LabeledPrecedent",
    # Phase 5C
    "ExplanationAssembler",
    "LegalExplanation",
]
