"""
ChromaDB Configuration Module

Centralized configuration for ChromaDB persistence and settings.
"""

import os
from pathlib import Path
from chromadb.config import Settings

# Database paths
CHROMA_DB_PATH = "db_store/chroma"
CHROMA_DB_PATH_ABS = Path(__file__).parent.parent.parent / CHROMA_DB_PATH

# Collection names
DEFAULT_COLLECTION = "legal_docs"

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Retrieval parameters
DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def get_chroma_settings() -> Settings:
    """
    Get ChromaDB settings with persistent storage configuration.
    
    Returns:
        Settings: ChromaDB settings object configured for persistence
    """
    # Ensure directory exists
    CHROMA_DB_PATH_ABS.mkdir(parents=True, exist_ok=True)
    
    return Settings(
        persist_directory=str(CHROMA_DB_PATH_ABS),
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )


def get_db_path() -> Path:
    """Get absolute path to ChromaDB storage directory."""
    return CHROMA_DB_PATH_ABS


def get_collection_name(name: str = None) -> str:
    """Get collection name with validation."""
    return name if name else DEFAULT_COLLECTION


# Logging configuration
ENABLE_LOGGING = True
LOG_LEVEL = "INFO"
