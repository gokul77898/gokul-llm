"""RAG Ingestion - Document loading, normalization, and validation"""

from .loaders import load_document, load_from_text
from .normalizer import normalize_text, normalize_section_refs, normalize_act_names
from .validator import validate_document, ValidationError

__all__ = [
    "load_document",
    "load_from_text",
    "normalize_text",
    "normalize_section_refs",
    "normalize_act_names",
    "validate_document",
    "ValidationError",
]
