"""
ChromaDB Database Layer for Legal AI System

This package provides a complete vector database solution with:
- Persistent ChromaDB storage
- Universal document ingestion (PDF, TXT, DOCX, HTML)
- Automatic chunking and embedding
- Efficient vector retrieval
"""

from .client import ChromaDBClient
from .retriever import VectorRetriever
from .ingestion import ingest_file, ingest_directory
from .embeddings import EmbeddingModel

__all__ = [
    'ChromaDBClient',
    'VectorRetriever',
    'ingest_file',
    'ingest_directory',
    'EmbeddingModel'
]

__version__ = '1.0.0'
