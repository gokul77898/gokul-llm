"""Embedding Module - Sentence Transformers"""
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Sentence transformer embedding model (singleton)."""
    
    _instance: Optional['EmbeddingModel'] = None
    _model = None
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    def __new__(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = None):
        if self._model is None:
            self.model_name = model_name or self.DEFAULT_MODEL
            self._load_model()
    
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded (dim: {self.EMBEDDING_DIMENSION})")
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise
    
    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed texts. Returns list of embeddings."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 100)
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """Embed single text."""
        return self.embed([text])[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.EMBEDDING_DIMENSION
