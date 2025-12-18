"""
RAG Chunking - Phase R1: Legal Chunking & Indexing

Legal-structure-aware chunking for legal documents.

Phase R1 implements:
- Section-level chunking for bare acts
- Paragraph-level chunking for case law
- Deterministic chunk IDs
- Chunk storage and indexing

Phase R1 does NOT implement:
- Embeddings
- Vector databases
- Retrieval logic
"""

from .chunker import LegalChunker, chunk_document
from .section_parser import SectionParser, ParsedSection
from .id_generator import generate_chunk_id

__all__ = [
    "LegalChunker",
    "chunk_document",
    "SectionParser",
    "ParsedSection",
    "generate_chunk_id",
]
