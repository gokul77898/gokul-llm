"""
Data Schema Module

Defines data structures for document chunks and metadata.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib


@dataclass
class DocumentChunk:
    """
    Represents a chunk of document text with metadata.
    
    Attributes:
        id: Unique identifier for the chunk
        text: The actual text content
        metadata: Dictionary containing document metadata
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        text: str,
        filename: str,
        chunk_index: int,
        page: Optional[int] = None,
        total_chunks: Optional[int] = None,
        **extra_metadata
    ) -> 'DocumentChunk':
        """
        Factory method to create a DocumentChunk with standard metadata.
        
        Args:
            text: Chunk text content
            filename: Source filename
            chunk_index: Index of this chunk in the document
            page: Page number (for PDFs)
            total_chunks: Total number of chunks in document
            **extra_metadata: Additional metadata fields
            
        Returns:
            DocumentChunk: New chunk instance
        """
        # Generate unique ID
        chunk_id = cls._generate_id(filename, chunk_index)
        
        # Build metadata
        metadata = {
            "source": filename,
            "chunk_index": chunk_index,
            "timestamp": datetime.now().isoformat(),
            "char_count": len(text),
            "word_count": len(text.split())
        }
        
        if page is not None:
            metadata["page"] = page
        
        if total_chunks is not None:
            metadata["total_chunks"] = total_chunks
        
        # Add extra metadata
        metadata.update(extra_metadata)
        
        return cls(id=chunk_id, text=text, metadata=metadata)
    
    @staticmethod
    def _generate_id(filename: str, chunk_index: int) -> str:
        """Generate deterministic ID for chunk."""
        content = f"{filename}_{chunk_index}_{datetime.now().date()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_chroma_format(self) -> tuple:
        """
        Convert to ChromaDB format.
        
        Returns:
            tuple: (id, document_text, metadata)
        """
        return (self.id, self.text, self.metadata)


@dataclass
class RetrievalResult:
    """
    Represents a retrieval result from vector search.
    
    Attributes:
        id: Document chunk ID
        text: Retrieved text
        score: Similarity score
        metadata: Associated metadata
        distance: Distance metric (lower is better)
    """
    
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    distance: Optional[float] = None
    
    @classmethod
    def from_chroma_result(
        cls,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
        distances: List[float] = None
    ) -> List['RetrievalResult']:
        """
        Convert ChromaDB results to RetrievalResult objects.
        
        Args:
            ids: List of document IDs
            documents: List of document texts
            metadatas: List of metadata dicts
            distances: Optional list of distances
            
        Returns:
            List[RetrievalResult]: List of retrieval results
        """
        results = []
        
        for i, (doc_id, text, metadata) in enumerate(zip(ids, documents, metadatas)):
            # Convert distance to similarity score (inverse)
            distance = distances[i] if distances else None
            score = 1.0 / (1.0 + distance) if distance is not None else 1.0
            
            results.append(cls(
                id=doc_id,
                text=text,
                score=score,
                metadata=metadata or {},
                distance=distance
            ))
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_source(self) -> str:
        """Get source filename."""
        return self.metadata.get("source", "unknown")
    
    def get_page(self) -> Optional[int]:
        """Get page number if available."""
        return self.metadata.get("page")


@dataclass
class IngestionStats:
    """Statistics from document ingestion."""
    
    filename: str
    chunks_created: int
    total_chars: int
    total_words: int
    pages: Optional[int] = None
    ingestion_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        parts = [
            f"File: {self.filename}",
            f"Chunks: {self.chunks_created}",
            f"Characters: {self.total_chars:,}",
            f"Words: {self.total_words:,}"
        ]
        
        if self.pages:
            parts.append(f"Pages: {self.pages}")
        
        parts.append(f"Time: {self.ingestion_time:.2f}s")
        
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        
        return " | ".join(parts)
