"""
ChromaDB Client Module

Manages ChromaDB client connection and collection management.
"""

import logging
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection

from .config import get_chroma_settings, get_collection_name

logger = logging.getLogger(__name__)


class ChromaDBClient:
    """
    ChromaDB Client Manager
    
    Provides singleton access to ChromaDB client and collection management.
    Ensures persistent storage and proper resource handling.
    """
    
    _instance: Optional['ChromaDBClient'] = None
    _client: Optional[chromadb.Client] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single client instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ChromaDB client with persistent settings."""
        if self._client is None:
            settings = get_chroma_settings()
            self._client = chromadb.Client(settings)
            logger.info(f"ChromaDB client initialized with persistent storage: {settings.persist_directory}")
    
    def get_client(self) -> chromadb.Client:
        """
        Get ChromaDB client instance.
        
        Returns:
            chromadb.Client: Active ChromaDB client
        """
        return self._client
    
    def get_or_create_collection(
        self,
        name: str = None,
        metadata: dict = None,
        embedding_function: callable = None
    ) -> Collection:
        """
        Get existing collection or create new one.
        
        Args:
            name: Collection name (default: from config)
            metadata: Optional collection metadata
            embedding_function: Optional custom embedding function
            
        Returns:
            Collection: ChromaDB collection instance
        """
        collection_name = get_collection_name(name)
        
        try:
            collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata=metadata or {"description": "Legal documents collection"},
                embedding_function=embedding_function
            )
            logger.info(f"Collection '{collection_name}' ready (count: {collection.count()})")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get/create collection '{collection_name}': {e}")
            raise
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Returns:
            bool: True if successful
        """
        try:
            self._client.delete_collection(name=name)
            logger.info(f"Collection '{name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False
    
    def list_collections(self) -> list:
        """
        List all collections.
        
        Returns:
            list: List of collection names
        """
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_collection_info(self, name: str) -> dict:
        """
        Get collection information.
        
        Args:
            name: Collection name
            
        Returns:
            dict: Collection metadata and stats
        """
        try:
            collection = self._client.get_collection(name=name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{name}': {e}")
            return {}
    
    def reset(self):
        """Reset ChromaDB (delete all collections). Use with caution!"""
        try:
            self._client.reset()
            logger.warning("ChromaDB reset - all collections deleted")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
            raise
