"""
Legal Retriever API

Phase R2: Dual Retrieval Engine

Unified retriever interface combining BM25 and dense retrieval.
Returns structured chunks with explainable scores.

NO LLMs used - retrieval only, no answer generation.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal
from enum import Enum

from .bm25 import BM25Retriever, BM25Result
from .dense import DenseRetriever, DenseResult
from .fusion import FusionRetriever, FusedResult, fuse_results


class RetrievalSource(str, Enum):
    """Source of retrieval result."""
    BM25 = "bm25"
    DENSE = "dense"
    FUSED = "fused"


@dataclass
class RetrievedChunk:
    """
    Retrieved chunk with metadata and score.
    
    This is the canonical output format for all retrieval operations.
    """
    chunk_id: str
    text: str
    act: Optional[str]
    section: Optional[str]
    doc_type: str
    citation: Optional[str]
    court: Optional[str]
    year: Optional[int]
    score: float
    source: str  # "bm25", "dense", or "fused"
    
    # Additional score breakdown (for explainability)
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class LegalRetriever:
    """
    Unified legal document retriever.
    
    Supports:
    - BM25 (sparse) retrieval
    - Dense (ChromaDB) retrieval
    - Fused retrieval (BM25 + Dense)
    
    Returns structured RetrievedChunk objects with
    explainable scores and metadata.
    
    NO LLMs are used - this is retrieval only.
    """
    
    def __init__(
        self,
        chunks_dir: str = "data/rag/chunks",
        chromadb_dir: str = "data/rag/chromadb",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        """
        Initialize legal retriever.
        
        Args:
            chunks_dir: Directory containing chunk JSON files
            chromadb_dir: Directory for ChromaDB persistence
            bm25_weight: Weight for BM25 in fusion
            dense_weight: Weight for dense in fusion
        """
        self.chunks_dir = chunks_dir
        self.chromadb_dir = chromadb_dir
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # Initialize retrievers lazily
        self._bm25: Optional[BM25Retriever] = None
        self._dense: Optional[DenseRetriever] = None
        self._initialized = False
    
    @property
    def bm25(self) -> BM25Retriever:
        """Get BM25 retriever (lazy init)."""
        if self._bm25 is None:
            self._bm25 = BM25Retriever(chunks_dir=self.chunks_dir)
        return self._bm25
    
    @property
    def dense(self) -> DenseRetriever:
        """Get dense retriever (lazy init)."""
        if self._dense is None:
            self._dense = DenseRetriever(persist_dir=self.chromadb_dir)
        return self._dense
    
    def initialize(self, index_dense: bool = True) -> Dict:
        """
        Initialize retrievers and build indices.
        
        Args:
            index_dense: Whether to index chunks into ChromaDB
            
        Returns:
            Initialization statistics
        """
        # Load BM25 index
        bm25_count = self.bm25.load()
        
        # Index into ChromaDB if requested
        dense_indexed = 0
        if index_dense:
            dense_indexed = self.dense.index_chunks(self.chunks_dir)
        
        dense_count = self.dense.get_chunk_count()
        
        self._initialized = True
        
        return {
            "bm25_chunks": bm25_count,
            "dense_chunks": dense_count,
            "dense_indexed": dense_indexed,
            "initialized": True,
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: Literal["bm25", "dense", "fused"] = "fused",
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            method: Retrieval method ("bm25", "dense", or "fused")
            
        Returns:
            List of RetrievedChunk sorted by relevance
        """
        if not self._initialized:
            self.initialize()
        
        if method == "bm25":
            return self._retrieve_bm25(query, top_k)
        elif method == "dense":
            return self._retrieve_dense(query, top_k)
        else:  # fused
            return self._retrieve_fused(query, top_k)
    
    def _retrieve_bm25(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve using BM25 only."""
        results = self.bm25.query(query, top_k=top_k)
        
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                act=r.act,
                section=r.section,
                doc_type=r.doc_type,
                citation=r.citation,
                court=r.court,
                year=r.year,
                score=r.score,
                source=RetrievalSource.BM25.value,
                bm25_score=r.score,
                dense_score=None,
            )
            for r in results
        ]
    
    def _retrieve_dense(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve using dense retrieval only."""
        results = self.dense.query(query, top_k=top_k)
        
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                act=r.act,
                section=r.section,
                doc_type=r.doc_type,
                citation=r.citation,
                court=r.court,
                year=r.year,
                score=r.score,
                source=RetrievalSource.DENSE.value,
                bm25_score=None,
                dense_score=r.score,
            )
            for r in results
        ]
    
    def _retrieve_fused(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve using fused BM25 + dense."""
        # Get results from both sources
        fetch_k = min(top_k * 2, 50)
        
        bm25_results = self.bm25.query(query, top_k=fetch_k)
        dense_results = self.dense.query(query, top_k=fetch_k)
        
        # Fuse results
        fused = fuse_results(
            bm25_results=bm25_results,
            dense_results=dense_results,
            query_text=query,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )
        
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                act=r.act,
                section=r.section,
                doc_type=r.doc_type,
                citation=r.citation,
                court=r.court,
                year=r.year,
                score=r.score,
                source=r.source,
                bm25_score=r.bm25_score,
                dense_score=r.dense_score,
            )
            for r in fused[:top_k]
        ]
    
    def retrieve_by_section(
        self,
        act: str,
        section: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks by act and section.
        
        Args:
            act: Act name (e.g., "IPC")
            section: Section number (e.g., "420")
            top_k: Number of results
            
        Returns:
            Matching chunks
        """
        query = f"Section {section} {act}"
        return self.retrieve(query, top_k=top_k, method="bm25")
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        stats = {
            "initialized": self._initialized,
            "chunks_dir": self.chunks_dir,
            "chromadb_dir": self.chromadb_dir,
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
        }
        
        if self._initialized:
            stats["bm25"] = self.bm25.get_stats()
            stats["dense"] = self.dense.get_stats()
        
        return stats
    
    def explain_result(self, result: RetrievedChunk) -> Dict:
        """
        Explain why a result was retrieved.
        
        Args:
            result: Retrieved chunk to explain
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            "chunk_id": result.chunk_id,
            "final_score": result.score,
            "source": result.source,
            "score_breakdown": {},
            "metadata_match": {},
        }
        
        # Score breakdown
        if result.bm25_score is not None:
            explanation["score_breakdown"]["bm25"] = {
                "raw_score": result.bm25_score,
                "weight": self.bm25_weight,
                "contribution": result.bm25_score * self.bm25_weight,
            }
        
        if result.dense_score is not None:
            explanation["score_breakdown"]["dense"] = {
                "raw_score": result.dense_score,
                "weight": self.dense_weight,
                "contribution": result.dense_score * self.dense_weight,
            }
        
        # Metadata
        explanation["metadata_match"] = {
            "act": result.act,
            "section": result.section,
            "doc_type": result.doc_type,
            "year": result.year,
        }
        
        return explanation
