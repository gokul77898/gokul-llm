"""Retriever component for RAG system"""

import torch
from typing import List, Dict, Optional, Tuple
from .document_store import DocumentStore, Document
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    documents: List[Document]
    scores: List[float]
    query: str
    metadata: Optional[Dict] = None


class LegalRetriever:
    """
    Advanced retriever for legal documents.
    
    Features:
    - Dense retrieval using embeddings
    - Re-ranking capabilities
    - Query expansion
    - Context-aware retrieval
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        top_k: int = 5,
        use_reranking: bool = True,
        reranker_model: Optional[str] = None
    ):
        """
        Args:
            document_store: Document store to retrieve from
            top_k: Number of documents to retrieve
            use_reranking: Whether to use re-ranking
            reranker_model: Optional re-ranker model name
        """
        self.document_store = document_store
        self.top_k = top_k
        self.use_reranking = use_reranking
        
        # Initialize re-ranker if requested
        self.reranker = None
        if use_reranking and reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model)
            except ImportError:
                print("Warning: CrossEncoder not available. Install sentence-transformers.")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        expand_query: bool = False
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)
            filter_metadata: Metadata filters
            expand_query: Whether to expand the query
            
        Returns:
            RetrievalResult with documents and scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Expand query if requested
        if expand_query:
            query = self._expand_query(query)
        
        # Initial retrieval
        initial_k = top_k * 2 if self.use_reranking else top_k
        results = self.document_store.search(
            query,
            top_k=initial_k,
            filter_metadata=filter_metadata
        )
        
        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]
        
        # Re-ranking
        if self.use_reranking and self.reranker and len(documents) > 0:
            documents, scores = self._rerank(query, documents, scores, top_k)
        else:
            documents = documents[:top_k]
            scores = scores[:top_k]
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
            metadata={'expand_query': expand_query, 'reranked': self.use_reranking}
        )
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with legal synonyms and related terms.
        
        Simple implementation - can be enhanced with legal thesaurus.
        """
        # Legal term expansions
        expansions = {
            'contract': 'agreement',
            'plaintiff': 'claimant',
            'defendant': 'respondent',
            'lawsuit': 'litigation',
            'judge': 'court',
        }
        
        expanded_terms = [query]
        words = query.lower().split()
        
        for word in words:
            if word in expansions:
                expanded_terms.append(expansions[word])
        
        return ' '.join(expanded_terms)
    
    def _rerank(
        self,
        query: str,
        documents: List[Document],
        scores: List[float],
        top_k: int
    ) -> Tuple[List[Document], List[float]]:
        """
        Re-rank retrieved documents using cross-encoder.
        
        Args:
            query: Original query
            documents: Retrieved documents
            scores: Initial retrieval scores
            top_k: Number of documents to return
            
        Returns:
            Re-ranked documents and scores
        """
        if self.reranker is None:
            return documents[:top_k], scores[:top_k]
        
        # Prepare pairs for re-ranking
        pairs = [[query, doc.content] for doc in documents]
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Sort by re-ranking scores
        sorted_indices = sorted(
            range(len(rerank_scores)),
            key=lambda i: rerank_scores[i],
            reverse=True
        )
        
        # Return top-k re-ranked results
        reranked_docs = [documents[i] for i in sorted_indices[:top_k]]
        reranked_scores = [float(rerank_scores[i]) for i in sorted_indices[:top_k]]
        
        return reranked_docs, reranked_scores
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Retrieve documents for multiple queries"""
        return [
            self.retrieve(query, top_k, filter_metadata)
            for query in queries
        ]


class ContextualRetriever(LegalRetriever):
    """
    Retriever that considers conversation context.
    
    Useful for multi-turn legal Q&A.
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        top_k: int = 5,
        use_reranking: bool = True,
        reranker_model: Optional[str] = None,
        context_window: int = 3
    ):
        super().__init__(document_store, top_k, use_reranking, reranker_model)
        self.context_window = context_window
        self.conversation_history = []
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> RetrievalResult:
        """
        Retrieve considering conversation history.
        
        Args:
            query: Current query
            top_k: Number of documents to retrieve
            filter_metadata: Metadata filters
            
        Returns:
            RetrievalResult with documents
        """
        # Add current query to history
        self.conversation_history.append(query)
        
        # Keep only recent context
        if len(self.conversation_history) > self.context_window:
            self.conversation_history = self.conversation_history[-self.context_window:]
        
        # Combine query with context
        contextualized_query = self._contextualize_query(query)
        
        # Retrieve with contextualized query
        result = self.retrieve(
            contextualized_query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Update metadata
        result.metadata = result.metadata or {}
        result.metadata['original_query'] = query
        result.metadata['contextualized'] = True
        
        return result
    
    def _contextualize_query(self, current_query: str) -> str:
        """
        Combine current query with conversation context.
        
        Args:
            current_query: Current query
            
        Returns:
            Contextualized query string
        """
        if len(self.conversation_history) <= 1:
            return current_query
        
        # Simple concatenation of recent queries
        # Can be enhanced with query reformulation models
        context = " ".join(self.conversation_history[:-1])
        return f"{context} {current_query}"
    
    def reset_context(self):
        """Reset conversation history"""
        self.conversation_history = []


class MultiModalRetriever(LegalRetriever):
    """
    Retriever supporting multiple retrieval strategies.
    
    Can combine:
    - Dense retrieval (embeddings)
    - Sparse retrieval (BM25)
    - Hybrid approaches
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        super().__init__(document_store, top_k)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize BM25 for sparse retrieval
        self.bm25_index = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index for sparse retrieval"""
        try:
            from rank_bm25 import BM25Okapi
            
            # Get all documents from store
            if hasattr(self.document_store, 'documents'):
                documents = self.document_store.documents
                tokenized_docs = [doc.content.lower().split() for doc in documents]
                self.bm25_index = BM25Okapi(tokenized_docs)
        except ImportError:
            print("Warning: rank_bm25 not installed. Sparse retrieval unavailable.")
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> RetrievalResult:
        """
        Hybrid retrieval combining dense and sparse methods.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_metadata: Metadata filters
            
        Returns:
            RetrievalResult with documents
        """
        if top_k is None:
            top_k = self.top_k
        
        # Dense retrieval
        dense_results = self.retrieve(query, top_k=top_k * 2, filter_metadata=filter_metadata)
        
        # Sparse retrieval (BM25)
        sparse_results = []
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top documents
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k * 2]
            
            sparse_results = [
                (self.document_store.documents[i], bm25_scores[i])
                for i in top_indices
            ]
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for doc, score in zip(dense_results.documents, dense_results.scores):
            combined_scores[doc.doc_id] = self.dense_weight * (1.0 / (1.0 + score))
        
        # Add sparse scores
        for doc, score in sparse_results:
            if doc.doc_id in combined_scores:
                combined_scores[doc.doc_id] += self.sparse_weight * score
            else:
                combined_scores[doc.doc_id] = self.sparse_weight * score
        
        # Sort by combined score
        sorted_doc_ids = sorted(
            combined_scores.keys(),
            key=lambda doc_id: combined_scores[doc_id],
            reverse=True
        )[:top_k]
        
        # Get documents
        doc_id_to_doc = {doc.doc_id: doc for doc in dense_results.documents}
        for doc, _ in sparse_results:
            if doc.doc_id not in doc_id_to_doc:
                doc_id_to_doc[doc.doc_id] = doc
        
        final_docs = [doc_id_to_doc[doc_id] for doc_id in sorted_doc_ids]
        final_scores = [combined_scores[doc_id] for doc_id in sorted_doc_ids]
        
        return RetrievalResult(
            documents=final_docs,
            scores=final_scores,
            query=query,
            metadata={'retrieval_type': 'hybrid'}
        )
