"""Document Store implementations for RAG system"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path


class Document:
    """Document container"""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or str(hash(content))
        self.embedding = None
    
    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, content_length={len(self.content)})"


class DocumentStore(ABC):
    """Abstract base class for document stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save document store to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load document store from disk"""
        pass


class FAISSStore(DocumentStore):
    """
    FAISS-based document store for efficient similarity search.
    
    Features:
    - Fast vector similarity search
    - Support for large document collections
    - Configurable embedding models
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: Optional[int] = None,
        index_type: str = "Flat"
    ):
        """
        Args:
            embedding_model: Name of sentence transformer model
            dimension: Embedding dimension (auto-detected if None)
            index_type: FAISS index type ("Flat", "IVFFlat", "HNSW")
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = dimension or self.embedding_model.get_sentence_embedding_dimension()
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        print(f"Adding {len(documents)} documents to FAISS store...")
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store documents and embeddings
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self.documents.append(doc)
            self.embeddings.append(emb)
        
        # Add to FAISS index
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            print("Training FAISS index...")
            self.index.train(embeddings_array)
        
        self.index.add(embeddings_array)
        
        print(f"Total documents in store: {len(self.documents)}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve documents
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply metadata filter if provided
                if filter_metadata:
                    if all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                        results.append((doc, float(dist)))
                else:
                    results.append((doc, float(dist)))
        
        return results
    
    def save(self, path: str):
        """Save document store to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save documents
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save config
        config = {
            'embedding_model': self.embedding_model._modules['0'].auto_model.config.name_or_path,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        with open(path / "config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Document store saved to {path}")
    
    def load(self, path: str):
        """Load document store from disk"""
        path = Path(path)
        
        # Load config
        with open(path / "config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load documents
        with open(path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        # Extract embeddings from documents
        self.embeddings = [doc.embedding for doc in self.documents]
        
        print(f"Document store loaded from {path}")
        print(f"Total documents: {len(self.documents)}")


class ChromaStore(DocumentStore):
    """
    ChromaDB-based document store for persistent storage.
    
    Features:
    - Persistent storage
    - Metadata filtering
    - Easy integration with LangChain
    """
    
    def __init__(
        self,
        collection_name: str = "legal_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_model: Name of sentence transformer model
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.documents = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        print(f"Adding {len(documents)} documents to ChromaDB store...")
        
        # Prepare data for ChromaDB
        ids = [doc.doc_id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        self.documents.extend(documents)
        
        print(f"Total documents in store: {len(self.documents)}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Convert to Document objects
        documents_with_scores = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                doc_id=results['ids'][0][i]
            )
            distance = results['distances'][0][i]
            documents_with_scores.append((doc, distance))
        
        return documents_with_scores
    
    def save(self, path: str):
        """Save is automatic with ChromaDB persistence"""
        self.client.persist()
        print(f"ChromaDB persisted to {self.persist_directory}")
    
    def load(self, path: str):
        """Load is automatic with ChromaDB"""
        # Collection is loaded on initialization
        count = self.collection.count()
        print(f"ChromaDB loaded from {self.persist_directory}")
        print(f"Total documents: {count}")


class HybridStore(DocumentStore):
    """
    Hybrid document store combining multiple backends.
    
    Uses FAISS for fast retrieval and ChromaDB for persistence.
    """
    
    def __init__(
        self,
        faiss_store: FAISSStore,
        chroma_store: ChromaStore
    ):
        self.faiss_store = faiss_store
        self.chroma_store = chroma_store
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both stores"""
        self.faiss_store.add_documents(documents)
        self.chroma_store.add_documents(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_faiss: bool = True
    ) -> List[Tuple[Document, float]]:
        """Search using either FAISS or ChromaDB"""
        if use_faiss:
            return self.faiss_store.search(query, top_k, filter_metadata)
        else:
            return self.chroma_store.search(query, top_k, filter_metadata)
    
    def save(self, path: str):
        """Save both stores"""
        self.faiss_store.save(f"{path}/faiss")
        self.chroma_store.save(f"{path}/chroma")
    
    def load(self, path: str):
        """Load both stores"""
        self.faiss_store.load(f"{path}/faiss")
        self.chroma_store.load(f"{path}/chroma")
