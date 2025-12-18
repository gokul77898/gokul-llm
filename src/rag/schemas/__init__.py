"""RAG Schemas - Canonical data models for legal documents and chunks"""

from .document import LegalDocument, DocumentType
from .chunk import LegalChunk, ChunkIndexEntry

__all__ = ["LegalDocument", "DocumentType", "LegalChunk", "ChunkIndexEntry"]
