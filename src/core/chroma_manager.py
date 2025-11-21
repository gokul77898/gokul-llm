"""
ChromaDB Manager for API Integration

Provides centralized ChromaDB initialization and management.
"""

import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import ChromaDB components
try:
    # First check if chromadb package is available
    import chromadb
    from db.chroma import ChromaDBClient, VectorRetriever
    CHROMA_AVAILABLE = True
    logger.info("ChromaDB components loaded successfully")
except ImportError as e:
    logger.warning(f"ChromaDB not available: {e}")
    CHROMA_AVAILABLE = False
    ChromaDBClient = None
    VectorRetriever = None


class ChromaManager:
    """Centralized ChromaDB management"""
    
    _instance: Optional['ChromaManager'] = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ChromaDB manager"""
        if not hasattr(self, 'initialized'):
            self.client = None
            self.retriever = None
            self.collection_name = "legal_docs"
            self.initialized = False
            self._collection = None
            
            if not CHROMA_AVAILABLE:
                print("ChromaDB not available")
                return
                
            self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create client
            self.client = ChromaDBClient()
            
            # Create/get collection (will be empty initially)
            self._collection = self.client.get_or_create_collection(self.collection_name)
            
            # Create retriever
            self.retriever = VectorRetriever(self.collection_name)
            
            self.initialized = True
            logger.info(f"ChromaDB initialized (collection: {self.collection_name}, count: {self._collection.count()})")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            self.initialized = False
    
    def get_retriever(self) -> Optional[VectorRetriever]:
        """Get ChromaDB retriever"""
        return self.retriever if self.initialized else None
    
    def get_client(self) -> Optional[ChromaDBClient]:
        """Get ChromaDB client"""
        return self.client if self.initialized else None
    
    def get_collection_stats(self) -> dict:
        """Get collection statistics"""
        if not self.initialized or not self.retriever:
            return {
                "status": "not_initialized",
                "collection": self.collection_name,
                "document_count": 0
            }
        
        try:
            stats = self.retriever.get_collection_stats()
            stats['status'] = 'ok'
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "collection": self.collection_name
            }
    
    def is_ready(self) -> bool:
        """Check if ChromaDB is ready"""
        return self.initialized and self.client is not None
    
    @property
    def collection(self):
        """Get the collection (for backward compatibility)"""
        if not self.initialized:
            raise AttributeError("ChromaDB not initialized - collection not available")
        return self._collection
    
    def get_collection(self):
        """Get the collection"""
        if not self.initialized:
            raise AttributeError("ChromaDB not initialized - collection not available")
        return self._collection
    
    def stats(self) -> dict:
        """Get ChromaDB statistics (for backward compatibility)"""
        return self.get_collection_stats()
    
    def list_documents(self) -> list:
        """List all documents in the collection"""
        if not self.initialized or not self._collection:
            return []
        
        try:
            results = self._collection.get(include=['metadatas'])
            sources = list(set([m.get('source', 'unknown') for m in results['metadatas']]))
            return sources
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def count_documents(self) -> int:
        """Count total documents"""
        if not self.initialized or not self._collection:
            return 0
        
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0


# Global instance
def get_chroma_manager() -> ChromaManager:
    """Get global ChromaDB manager instance"""
    return ChromaManager()
