"""Vector Retrieval Module"""
import logging
from typing import List, Optional, Dict, Any
from .client import ChromaDBClient
from .embeddings import EmbeddingModel
from .schema import RetrievalResult
from .config import DEFAULT_COLLECTION, DEFAULT_TOP_K, MAX_TOP_K

logger = logging.getLogger(__name__)

class VectorRetriever:
    """Vector search retriever for ChromaDB."""
    
    def __init__(self, collection_name: str = None):
        """Initialize retriever with collection."""
        self.client = ChromaDBClient()
        self.embedder = EmbeddingModel()
        self.collection_name = collection_name or DEFAULT_COLLECTION
        self.collection = self.client.get_or_create_collection(self.collection_name)
        logger.info(f"Retriever ready (collection: {self.collection_name}, docs: {self.collection.count()})")
    
    def query(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar documents.
        
        Args:
            text: Query text
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List[RetrievalResult]: Search results
        """
        if top_k > MAX_TOP_K:
            logger.warning(f"top_k={top_k} exceeds max={MAX_TOP_K}, capping")
            top_k = MAX_TOP_K
        
        # Embed query
        query_embedding = self.embedder.embed_single(text)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Parse results
        if not results['ids'] or not results['ids'][0]:
            logger.info("No results found")
            return []
        
        retrieval_results = RetrievalResult.from_chroma_result(
            ids=results['ids'][0],
            documents=results['documents'][0],
            metadatas=results['metadatas'][0],
            distances=results.get('distances', [[]])[0]
        )
        
        logger.info(f"Retrieved {len(retrieval_results)} results for query")
        return retrieval_results
    
    def get_by_id(self, doc_id: str) -> Optional[RetrievalResult]:
        """Get document by ID."""
        result = self.collection.get(ids=[doc_id])
        
        if not result['ids']:
            return None
        
        return RetrievalResult(
            id=result['ids'][0],
            text=result['documents'][0],
            score=1.0,
            metadata=result['metadatas'][0]
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "name": self.collection_name,
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata
        }
