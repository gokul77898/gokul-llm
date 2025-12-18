"""
Canonical Legal Document Schema

Phase R0: RAG Foundations

This module defines the canonical Pydantic model for legal documents.
All ingested documents must conform to this schema.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Legal document types supported by the system"""
    BARE_ACT = "bare_act"
    CASE_LAW = "case_law"
    AMENDMENT = "amendment"
    NOTIFICATION = "notification"


class LegalDocument(BaseModel):
    """
    Canonical schema for legal documents.
    
    All ingested documents must conform to this schema.
    The doc_id is deterministic based on content hash.
    """
    
    doc_id: str = Field(
        ...,
        description="Deterministic document ID (SHA256 hash of raw_text + source)"
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Document title"
    )
    doc_type: DocumentType = Field(
        ...,
        description="Type of legal document"
    )
    act: Optional[str] = Field(
        default=None,
        description="Name of the act (for bare_act, amendment)"
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
    citation: Optional[str] = Field(
        default=None,
        description="Legal citation reference"
    )
    raw_text: str = Field(
        ...,
        min_length=1,
        description="Full text content of the document"
    )
    source: str = Field(
        ...,
        description="Source filename or URL"
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Document version (for versioned updates)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of document creation"
    )
    
    @field_validator("year")
    @classmethod
    def year_not_future(cls, v: Optional[int]) -> Optional[int]:
        """Validate year is not in the future"""
        if v is not None:
            current_year = datetime.now().year
            if v > current_year:
                raise ValueError(f"Year {v} is in the future (current: {current_year})")
        return v
    
    @staticmethod
    def generate_doc_id(raw_text: str, source: str) -> str:
        """
        Generate deterministic document ID.
        
        Args:
            raw_text: Document text content
            source: Source filename or URL
            
        Returns:
            SHA256 hash as hex string (first 16 chars)
        """
        content = f"{raw_text}|{source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "a1b2c3d4e5f6g7h8",
                "title": "Indian Penal Code, 1860",
                "doc_type": "bare_act",
                "act": "IPC",
                "court": None,
                "year": 1860,
                "citation": "Act No. 45 of 1860",
                "raw_text": "Section 1. Title and extent of operation...",
                "source": "ipc_1860.pdf",
                "version": 1,
                "created_at": "2024-01-01T00:00:00"
            }
        }
