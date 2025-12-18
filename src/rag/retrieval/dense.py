"""
Dense Retrieval with ChromaDB

Phase R2: Dual Retrieval Engine

Implements dense vector retrieval using ChromaDB and sentence-transformers.
Stores chunk embeddings with metadata for semantic search.

NO LLMs used in this module - only embedding models.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings


@dataclass
class DenseResult:
    """Result from dense retrieval."""
    chunk_id: str
    text: str
    score: float  # Similarity score (higher = more similar)
    act: Optional[str]
    section: Optional[str]
    doc_type: str
    year: Optional[int]
    citation: Optional[str]
    court: Optional[str]


class DenseRetriever:
    """
    Dense vector retrieval using ChromaDB.
    
    Features:
    - Uses sentence-transformers for embeddings
    - Stores chunks in ChromaDB collection
    - Supports semantic similarity search
    - Persists data to disk
    
    NO LLMs are used - only embedding models.
    """
    
    # Default embedding model - production-ready
    # BAAI/bge-large-en-v1.5 is CPU compatible, deterministic, no API calls
    DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
    
    # Legacy model (dev) - kept for reference
    LEGACY_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        persist_dir: str = "data/rag/chromadb",
        collection_name: str = "legal_chunks",
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize dense retriever.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence-transformer model name
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_model = embedding_model or self.DEFAULT_MODEL
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection with embedding function
        self._embedding_fn = None
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        try:
            from chromadb.utils import embedding_functions
            
            # Use sentence-transformers embedding function
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        except Exception as e:
            print(f"Warning: Failed to create embedding function: {e}")
            # Fallback to default
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def index_chunks(self, chunks_dir: str = "data/rag/chunks") -> int:
        """
        Index all chunks from the filesystem into ChromaDB.
        
        Args:
            chunks_dir: Directory containing chunk JSON files
            
        Returns:
            Number of chunks indexed
        """
        chunks_path = Path(chunks_dir)
        indexed = 0
        
        # Collect chunks to add
        ids = []
        documents = []
        metadatas = []
        
        for chunk_path in chunks_path.glob("*.json"):
            if chunk_path.name == "index.json":
                continue
            
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                
                chunk_id = chunk.get('chunk_id', '')
                text = chunk.get('text', '')
                
                if not chunk_id or not text:
                    continue
                
                # Check if already indexed
                existing = self.collection.get(ids=[chunk_id])
                if existing and existing['ids']:
                    continue
                
                ids.append(chunk_id)
                documents.append(text)
                metadatas.append({
                    "act": chunk.get('act') or "",
                    "section": chunk.get('section') or "",
                    "doc_type": chunk.get('doc_type', 'unknown'),
                    "year": chunk.get('year') or 0,
                    "citation": chunk.get('citation') or "",
                    "court": chunk.get('court') or "",
                    "doc_id": chunk.get('doc_id') or "",
                })
                
            except Exception as e:
                print(f"Warning: Failed to load {chunk_path}: {e}")
        
        # Batch add to collection
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            indexed = len(ids)
        
        return indexed
    
    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict,
    ) -> bool:
        """
        Add a single chunk to the index.
        
        Args:
            chunk_id: Unique chunk identifier
            text: Chunk text content
            metadata: Chunk metadata
            
        Returns:
            True if added successfully
        """
        try:
            # Check if already exists
            existing = self.collection.get(ids=[chunk_id])
            if existing and existing['ids']:
                return False
            
            self.collection.add(
                ids=[chunk_id],
                documents=[text],
                metadatas=[{
                    "act": metadata.get('act') or "",
                    "section": metadata.get('section') or "",
                    "doc_type": metadata.get('doc_type', 'unknown'),
                    "year": metadata.get('year') or 0,
                    "citation": metadata.get('citation') or "",
                    "court": metadata.get('court') or "",
                    "doc_id": metadata.get('doc_id') or "",
                }],
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to add chunk {chunk_id}: {e}")
            return False
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filter_act: Optional[str] = None,
        filter_doc_type: Optional[str] = None,
    ) -> List[DenseResult]:
        """
        Query the dense index.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            filter_act: Optional filter by act name
            filter_doc_type: Optional filter by document type
            
        Returns:
            List of DenseResult sorted by similarity
        """
        # Build where clause for filtering
        where = None
        if filter_act or filter_doc_type:
            conditions = []
            if filter_act:
                conditions.append({"act": {"$eq": filter_act}})
            if filter_doc_type:
                conditions.append({"doc_type": {"$eq": filter_doc_type}})
            
            if len(conditions) == 1:
                where = conditions[0]
            else:
                where = {"$and": conditions}
        
        # Query collection
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"Warning: Query failed: {e}")
            return []
        
        # Build result objects
        dense_results = []
        
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            for i, chunk_id in enumerate(ids):
                # Convert distance to similarity score
                # ChromaDB returns L2 distance for cosine, so 1 - distance/2 for similarity
                distance = distances[i] if i < len(distances) else 0
                similarity = 1 - (distance / 2)  # Convert to 0-1 similarity
                
                metadata = metadatas[i] if i < len(metadatas) else {}
                text = documents[i] if i < len(documents) else ""
                
                dense_results.append(DenseResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(similarity),
                    act=metadata.get('act') or None,
                    section=metadata.get('section') or None,
                    doc_type=metadata.get('doc_type', 'unknown'),
                    year=metadata.get('year') or None,
                    citation=metadata.get('citation') or None,
                    court=metadata.get('court') or None,
                ))
        
        return dense_results
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from the index."""
        try:
            self.collection.delete(ids=[chunk_id])
            return True
        except Exception:
            return False
    
    def clear(self) -> bool:
        """Clear all chunks from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            return True
        except Exception:
            return False
    
    def get_chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return self.collection.count()
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        count = self.collection.count()
        
        return {
            "chunk_count": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "persist_dir": str(self.persist_dir),
        }


def index_all_chunks(
    chunk_dir: str = "data/rag/chunks",
    persist_dir: str = "data/rag/chromadb",
) -> Dict:
    """
    Index all chunks from chunk storage into ChromaDB.
    
    Args:
        chunk_dir: Path to chunk storage directory
        persist_dir: Path to ChromaDB persistence directory
        
    Returns:
        Indexing results
    """
    from ..storage.chunk_storage import ChunkStorage
    
    chunk_storage = ChunkStorage(base_path=chunk_dir)
    retriever = DenseRetriever(persist_dir=persist_dir)
    
    chunk_ids = chunk_storage.list_all()
    
    results = {
        "processed": 0,
        "indexed": 0,
        "failed": 0,
        "errors": [],
    }
    
    for chunk_id in chunk_ids:
        results["processed"] += 1
        
        try:
            chunk = chunk_storage.load(chunk_id)
            
            success = retriever.add_chunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "act": chunk.act,
                    "doc_type": chunk.doc_type.value if hasattr(chunk.doc_type, 'value') else str(chunk.doc_type),
                    "year": chunk.year,
                    "citation": chunk.citation,
                    "court": chunk.court,
                },
            )
            
            if success:
                results["indexed"] += 1
            else:
                results["failed"] += 1
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "chunk_id": chunk_id[:8],
                "error": str(e),
            })
    
    return results


if __name__ == "__main__":
    """CLI entry point for dense indexing."""
    import sys
    
    print("=" * 50)
    print("RAG Dense Indexing (ChromaDB)")
    print("=" * 50)
    
    try:
        results = index_all_chunks()
        
        print(f"\nProcessed: {results['processed']}")
        print(f"Indexed: {results['indexed']}")
        print(f"Failed: {results['failed']}")
        
        if results['errors']:
            print("\nErrors:")
            for err in results['errors']:
                print(f"  - {err['chunk_id']}: {err['error']}")
        
        # Show stats
        retriever = DenseRetriever()
        stats = retriever.get_stats()
        print(f"\nChromaDB Stats:")
        print(f"  - Collection: {stats['collection_name']}")
        print(f"  - Chunks indexed: {stats['chunk_count']}")
        print(f"  - Model: {stats['embedding_model']}")
        
        sys.exit(0 if results['failed'] == 0 else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
