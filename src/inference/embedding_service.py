"""
Remote Embedding Service - Generic wrapper for HF Space
"""

import os
import requests
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Remote embedding service URL
EMBEDDING_SERVICE_URL = "https://omilosaisolutions-indian-legal-encoder-8b.hf.space/encode"

def get_embedding(text: str) -> List[float]:
    """
    Get embedding from remote service.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of embedding floats
        
    Raises:
        RuntimeError: If embedding service fails
    """
    try:
        response = requests.post(
            EMBEDDING_SERVICE_URL,
            json={"query": text},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        embedding = data.get("embedding", [])
        
        if not embedding:
            raise RuntimeError("Empty embedding returned")
            
        return embedding
        
    except Exception as e:
        logger.error(f"Embedding service failed: {e}")
        raise RuntimeError(f"Failed to get embedding: {str(e)}")

def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Batch get embeddings.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding lists
    """
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings
