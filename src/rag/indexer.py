"""FAISS indexer for RAG system"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not available, using random embeddings")


class FAISSIndexer:
    """FAISS-based document indexer"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        normalize: bool = True,
        device: str = "cpu"
    ):
        """
        Args:
            embedding_model: Model name for sentence embeddings
            embedding_dim: Dimension of embeddings
            normalize: Whether to normalize embeddings
            device: Device to use for embedding
        """
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.device = device
        
        # Load embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(embedding_model)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, using random embeddings")
                self.model = None
        else:
            self.model = None
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        else:
            # Fallback: random embeddings for testing
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype('float32')
        
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings.astype('float32')
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build FAISS index from documents
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Extract texts
        texts = [doc['text'] for doc in documents]
        self.documents = documents
        self.metadata = [doc.get('metadata', {}) for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embed(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if self.normalize:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for normalized vectors
        
        self.index.add(embeddings)
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk"""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim,
            'normalize': self.normalize,
        }
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.embedding_dim = data['embedding_dim']
        self.normalize = data['normalize']
        
        logger.info(f"Loaded index from {index_path} with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of results with documents and scores
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        # Embed query
        query_embedding = self.embed([query])
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'index': int(idx),
                })
        
        return results


def index_documents(documents_file: str, index_path: str, metadata_path: str, config):
    """Build and save FAISS index from documents file"""
    # Load documents
    documents = []
    with open(documents_file, 'r') as f:
        for line in f:
            documents.append(json.loads(line))
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create indexer
    indexer = FAISSIndexer(
        embedding_model=config.model.embedding_model,
        embedding_dim=config.model.embedding_dim,
        normalize=config.model.normalize_embeddings,
        device=config.system.device
    )
    
    # Build and save index
    indexer.build_index(documents)
    indexer.save(index_path, metadata_path)
    
    return indexer


def load_index(index_path: str, metadata_path: str, config):
    """Load FAISS index from disk"""
    indexer = FAISSIndexer(
        embedding_model=config.model.embedding_model,
        embedding_dim=config.model.embedding_dim,
        normalize=config.model.normalize_embeddings,
        device=config.system.device
    )
    indexer.load(index_path, metadata_path)
    return indexer
