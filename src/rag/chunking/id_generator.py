"""
Deterministic Chunk ID Generator

Phase R1: Legal Chunking & Indexing

Generates deterministic chunk IDs based on content.
Re-ingestion produces the SAME IDs.
"""

import hashlib


def generate_chunk_id(
    doc_id: str,
    section: str | None,
    subsection: str | None,
    start_offset: int,
) -> str:
    """
    Generate a deterministic chunk ID.
    
    The chunk ID is a SHA256 hash of:
    - doc_id
    - section (or "none")
    - subsection (or "none")
    - start_offset
    
    This ensures:
    - Same content always produces same ID
    - Re-ingestion is idempotent
    - IDs are unique per chunk
    
    Args:
        doc_id: Parent document ID
        section: Section number (e.g., "420")
        subsection: Subsection identifier (e.g., "(1)")
        start_offset: Start character offset in document
        
    Returns:
        16-character hex string chunk ID
    """
    # Normalize None values
    section_str = section if section else "none"
    subsection_str = subsection if subsection else "none"
    
    # Create deterministic content string
    content = f"{doc_id}|{section_str}|{subsection_str}|{start_offset}"
    
    # Generate hash
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_chunk_id_from_text(
    doc_id: str,
    text: str,
    start_offset: int,
) -> str:
    """
    Generate chunk ID from text content (for case law paragraphs).
    
    Used when section/subsection are not available.
    
    Args:
        doc_id: Parent document ID
        text: Chunk text content
        start_offset: Start character offset
        
    Returns:
        16-character hex string chunk ID
    """
    # Use first 100 chars of text for uniqueness
    text_sample = text[:100] if len(text) > 100 else text
    
    content = f"{doc_id}|{text_sample}|{start_offset}"
    
    return hashlib.sha256(content.encode()).hexdigest()[:16]
