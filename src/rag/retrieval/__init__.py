"""
RAG Retrieval - Phase R2: Dual Retrieval Engine

Implements sparse (BM25) and dense (ChromaDB) retrieval
with deterministic fusion logic.

Phase R2 does NOT:
- Call the decoder
- Integrate MoE
- Generate answers
- Use LLMs

Retrieval is standalone and explainable.
"""

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import FusionRetriever, fuse_results
from .retriever import LegalRetriever, RetrievedChunk

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "FusionRetriever",
    "fuse_results",
    "LegalRetriever",
    "RetrievedChunk",
]
