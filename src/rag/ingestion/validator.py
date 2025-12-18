"""
Document Validator

Phase R0: RAG Foundations

Strict validation for legal documents.
Rejects documents that don't meet quality standards.
"""

import re
from datetime import datetime
from typing import List, Optional

from ..schemas.document import DocumentType, LegalDocument


class ValidationError(Exception):
    """Raised when document validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(f"Validation failed{f' for {field}' if field else ''}: {message}")


def validate_document(doc: LegalDocument) -> LegalDocument:
    """
    Validate a legal document against strict rules.
    
    Raises ValidationError if document is invalid.
    
    Args:
        doc: LegalDocument to validate
        
    Returns:
        The validated document (unchanged)
        
    Raises:
        ValidationError: If validation fails
    """
    errors: List[str] = []
    
    # Rule 1: raw_text must not be empty
    if not doc.raw_text or not doc.raw_text.strip():
        raise ValidationError("raw_text is empty or whitespace only", "raw_text")
    
    # Rule 2: raw_text must have meaningful content (at least 50 chars)
    if len(doc.raw_text.strip()) < 50:
        raise ValidationError(
            f"raw_text too short ({len(doc.raw_text.strip())} chars, minimum 50)",
            "raw_text"
        )
    
    # Rule 3: title must not be empty
    if not doc.title or not doc.title.strip():
        raise ValidationError("title is empty", "title")
    
    # Rule 4: doc_type must be valid
    if doc.doc_type not in DocumentType:
        raise ValidationError(
            f"Unknown doc_type: {doc.doc_type}. Valid types: {[t.value for t in DocumentType]}",
            "doc_type"
        )
    
    # Rule 5: year must not be in the future
    if doc.year is not None:
        current_year = datetime.now().year
        if doc.year > current_year:
            raise ValidationError(
                f"Year {doc.year} is in the future (current: {current_year})",
                "year"
            )
        if doc.year < 1800:
            raise ValidationError(
                f"Year {doc.year} is too old (minimum: 1800)",
                "year"
            )
    
    # Rule 6: source must not be empty
    if not doc.source or not doc.source.strip():
        raise ValidationError("source is empty", "source")
    
    # Rule 7: doc_id must be valid format (16 hex chars)
    if not doc.doc_id or len(doc.doc_id) != 16:
        raise ValidationError(
            f"doc_id must be 16 characters, got {len(doc.doc_id) if doc.doc_id else 0}",
            "doc_id"
        )
    if not all(c in '0123456789abcdef' for c in doc.doc_id.lower()):
        raise ValidationError("doc_id must be hexadecimal", "doc_id")
    
    # Rule 8: Validate section numbers if present (for bare_act)
    if doc.doc_type == DocumentType.BARE_ACT:
        _validate_section_numbers(doc.raw_text)
    
    # Rule 9: case_law must have court
    if doc.doc_type == DocumentType.CASE_LAW and not doc.court:
        raise ValidationError(
            "case_law documents must specify court",
            "court"
        )
    
    # Rule 10: bare_act and amendment must have act name
    if doc.doc_type in (DocumentType.BARE_ACT, DocumentType.AMENDMENT) and not doc.act:
        raise ValidationError(
            f"{doc.doc_type.value} documents must specify act name",
            "act"
        )
    
    return doc


def _validate_section_numbers(text: str) -> None:
    """
    Validate section number format in bare act text.
    
    Raises ValidationError if malformed section numbers found.
    
    Args:
        text: Document text to validate
        
    Raises:
        ValidationError: If malformed section numbers found
    """
    # Find all section references
    section_pattern = r'Section\s+(\S+)'
    matches = re.findall(section_pattern, text)
    
    for match in matches:
        # Section number should be numeric or numeric with subsection
        # Valid: "420", "420(1)", "420A", "420(1)(a)"
        # Invalid: "XYZ", "###", empty
        
        if not match:
            raise ValidationError(
                "Empty section number found",
                "raw_text"
            )
        
        # Check if it starts with a digit or is a valid format
        valid_pattern = r'^(\d+[A-Z]?(?:\([0-9a-z]+\))*)$'
        if not re.match(valid_pattern, match, re.IGNORECASE):
            # Allow some flexibility - just ensure it's not completely malformed
            if not any(c.isdigit() for c in match):
                raise ValidationError(
                    f"Malformed section number: '{match}' (must contain digits)",
                    "raw_text"
                )


def validate_text_quality(text: str) -> List[str]:
    """
    Check text quality and return warnings (non-fatal).
    
    Args:
        text: Document text
        
    Returns:
        List of warning messages
    """
    warnings: List[str] = []
    
    # Check for excessive special characters
    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if special_ratio > 0.3:
        warnings.append(f"High special character ratio: {special_ratio:.2%}")
    
    # Check for very short paragraphs (might indicate OCR issues)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    short_paragraphs = sum(1 for p in paragraphs if len(p) < 20)
    if paragraphs and short_paragraphs / len(paragraphs) > 0.5:
        warnings.append(f"Many short paragraphs: {short_paragraphs}/{len(paragraphs)}")
    
    # Check for repeated characters (OCR artifacts)
    if re.search(r'(.)\1{10,}', text):
        warnings.append("Repeated character sequences detected (possible OCR artifact)")
    
    return warnings
