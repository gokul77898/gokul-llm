"""
Chunk Schema - Phase R1: Legal Chunking & Indexing

Canonical schema for legal document chunks.
Each chunk represents ONE legal unit (section, subsection, or paragraph).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .document import DocumentType


class LegalChunk(BaseModel):
    """
    Canonical schema for legal document chunks.
    
    Each chunk represents ONE legal unit:
    - Section (for bare acts)
    - Subsection (for detailed acts)
    - Paragraph/holding (for case law)
    
    Chunk IDs are deterministic based on content.
    """
    
    chunk_id: str = Field(
        ...,
        description="Deterministic chunk ID (hash of doc_id + section + subsection + offset)"
    )
    doc_id: str = Field(
        ...,
        description="Parent document ID"
    )
    act: Optional[str] = Field(
        default=None,
        description="Act name (required for bare_act, amendment)"
    )
    section: Optional[str] = Field(
        default=None,
        description="Section number (e.g., '420', '420(1)')"
    )
    subsection: Optional[str] = Field(
        default=None,
        description="Subsection identifier (e.g., '(1)', '(a)')"
    )
    doc_type: DocumentType = Field(
        ...,
        description="Type of legal document"
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Chunk text content"
    )
    citation: Optional[str] = Field(
        default=None,
        description="Legal citation reference"
    )
    court: Optional[str] = Field(
        default=None,
        description="Court name (for case_law)"
    )
    year: Optional[int] = Field(
        default=None,
        ge=1800,
        le=2100,
        description="Year of the document"
    )
    start_offset: int = Field(
        ...,
        ge=0,
        description="Start character offset in original document"
    )
    end_offset: int = Field(
        ...,
        ge=0,
        description="End character offset in original document"
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Position of chunk in document (0-indexed)"
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Chunk version"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of chunk creation"
    )
    
    @field_validator("end_offset")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        """Validate end_offset is after start_offset"""
        start = info.data.get("start_offset", 0)
        if v < start:
            raise ValueError(f"end_offset ({v}) must be >= start_offset ({start})")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "a1b2c3d4e5f6g7h8",
                "doc_id": "parent_doc_id_123",
                "act": "IPC",
                "section": "420",
                "subsection": None,
                "doc_type": "bare_act",
                "text": "Whoever cheats and thereby dishonestly induces...",
                "citation": "Section 420, IPC",
                "court": None,
                "year": 1860,
                "start_offset": 1000,
                "end_offset": 1500,
                "chunk_index": 5,
                "version": 1,
                "created_at": "2024-01-01T00:00:00"
            }
        }


class ChunkIndexEntry(BaseModel):
    """
    Index entry for chunk manifest.
    
    Lightweight representation for the index.json file.
    Does NOT include full text content.
    """
    
    chunk_id: str
    doc_id: str
    act: Optional[str] = None
    section: Optional[str] = None
    doc_type: str
    year: Optional[int] = None
    
    @classmethod
    def from_chunk(cls, chunk: LegalChunk) -> "ChunkIndexEntry":
        """Create index entry from a chunk."""
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            act=chunk.act,
            section=chunk.section,
            doc_type=chunk.doc_type.value,
            year=chunk.year,
        )
