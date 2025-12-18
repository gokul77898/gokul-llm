"""
RAG Module - Phase R4: Context Assembler

This module provides legal document ingestion, chunking, retrieval, validation, and context assembly.

Phase R0 implements:
- Canonical document schema
- Text normalization for legal documents
- Strict validation
- Filesystem storage

Phase R1 implements:
- Legal-structure-aware chunking
- Section parsing
- Deterministic chunk IDs
- Chunk storage and indexing

Phase R2 implements:
- BM25 sparse retrieval
- ChromaDB dense retrieval
- Deterministic fusion logic
- Explainable retrieval scores

Phase R3 implements:
- Rule-based validation
- Evidence thresholds
- Statute/section consistency checks
- Hard refusal triggers

Phase R4 implements:
- Token budget management
- Evidence ordering
- Citation formatting
- Context assembly

Phase R4 does NOT implement:
- LLM integration
- Answer generation
- MoE integration

No LLM is ever called with unvalidated evidence.
The decoder never sees raw documents — only assembled evidence blocks.
"""

from .schemas.document import LegalDocument, DocumentType
from .schemas.chunk import LegalChunk, ChunkIndexEntry
from .ingestion.loaders import load_document, load_from_text
from .ingestion.normalizer import normalize_text
from .ingestion.validator import validate_document
from .storage.filesystem import FilesystemStorage
from .storage.chunk_storage import ChunkStorage
from .chunking.chunker import LegalChunker, chunk_document
from .chunking.section_parser import SectionParser, ParsedSection
from .chunking.id_generator import generate_chunk_id
from .retrieval.retriever import LegalRetriever, RetrievedChunk
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever
from .retrieval.fusion import FusionRetriever, fuse_results
from .validation.validator import RetrievalValidator, ValidationResult, RefusalReason
from .validation.threshold import EvidenceThreshold, ThresholdConfig
from .validation.statute_validator import StatuteValidator
from .validation.evidence_filter import EvidenceFilter
from .context.assembler import ContextAssembler, ContextResult, ContextRefusalReason
from .context.token_budget import TokenBudget, BudgetConfig
from .context.citation import CitationFormatter, Citation
from .context.formatter import EvidenceFormatter

__all__ = [
    # Document schemas
    "LegalDocument",
    "DocumentType",
    # Chunk schemas
    "LegalChunk",
    "ChunkIndexEntry",
    # Document ingestion
    "load_document",
    "load_from_text",
    "normalize_text",
    "validate_document",
    # Storage
    "FilesystemStorage",
    "ChunkStorage",
    # Chunking
    "LegalChunker",
    "chunk_document",
    "SectionParser",
    "ParsedSection",
    "generate_chunk_id",
    # Retrieval (Phase R2)
    "LegalRetriever",
    "RetrievedChunk",
    "BM25Retriever",
    "DenseRetriever",
    "FusionRetriever",
    "fuse_results",
    # Validation (Phase R3)
    "RetrievalValidator",
    "ValidationResult",
    "RefusalReason",
    "EvidenceThreshold",
    "ThresholdConfig",
    "StatuteValidator",
    "EvidenceFilter",
    # Context Assembly (Phase R4)
    "ContextAssembler",
    "ContextResult",
    "ContextRefusalReason",
    "TokenBudget",
    "BudgetConfig",
    "CitationFormatter",
    "Citation",
    "EvidenceFormatter",
]
