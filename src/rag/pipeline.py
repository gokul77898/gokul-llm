"""RAG Pipeline - End-to-end Retrieval-Augmented Generation"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
from .document_store import DocumentStore, Document
from .retriever import LegalRetriever, ContextualRetriever
from .generator import RAGGenerator, ChainOfThoughtGenerator, CitationGenerator
from pathlib import Path
import json


class RAGPipeline:
    """
    End-to-end RAG pipeline for legal document processing.
    
    Combines:
    - Document storage and indexing
    - Intelligent retrieval
    - Context-aware generation
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        retriever: LegalRetriever,
        generator: RAGGenerator,
        cache_results: bool = True
    ):
        """
        Args:
            document_store: Document store for corpus
            retriever: Retriever component
            generator: Generator component
            cache_results: Whether to cache query results
        """
        self.document_store = document_store
        self.retriever = retriever
        self.generator = generator
        self.cache_results = cache_results
        self.query_cache = {}
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        num_docs_to_use: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict[str, any]:
        """
        End-to-end query processing.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            num_docs_to_use: Number of documents to use for generation
            filter_metadata: Metadata filters for retrieval
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with answer and supporting information
        """
        # Check cache
        cache_key = f"{query}_{top_k}_{filter_metadata}"
        if use_cache and self.cache_results and cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Step 1: Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Step 2: Generate answer
        generation_result = self.generator.generate(
            query=query,
            retrieval_result=retrieval_result,
            num_docs_to_use=num_docs_to_use
        )
        
        # Combine results
        result = {
            'query': query,
            'answer': generation_result['answer'],
            'retrieved_documents': [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'relevance_score': score
                }
                for doc, score in zip(
                    retrieval_result.documents,
                    retrieval_result.scores
                )
            ],
            'generation_metadata': generation_result.get('metadata', {}),
            'num_docs_retrieved': len(retrieval_result.documents),
            'num_docs_used': generation_result['num_docs_used']
        }
        
        # Cache result
        if self.cache_results:
            self.query_cache[cache_key] = result
        
        return result
    
    def batch_query(
        self,
        queries: List[str],
        top_k: int = 5,
        **kwargs
    ) -> List[Dict]:
        """Process multiple queries"""
        return [self.query(q, top_k, **kwargs) for q in queries]
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the pipeline"""
        self.document_store.add_documents(documents)
        # Clear cache when documents are added
        self.query_cache = {}
    
    def save(self, path: str):
        """Save pipeline components"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save document store
        self.document_store.save(str(path / "document_store"))
        
        # Save pipeline config
        config = {
            'cache_results': self.cache_results,
        }
        with open(path / "pipeline_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"RAG Pipeline saved to {path}")
    
    def load(self, path: str):
        """Load pipeline components"""
        path = Path(path)
        
        # Load document store
        self.document_store.load(str(path / "document_store"))
        
        # Load config
        with open(path / "pipeline_config.json", 'r') as f:
            config = json.load(f)
            self.cache_results = config['cache_results']
        
        # Clear cache
        self.query_cache = {}
        
        print(f"RAG Pipeline loaded from {path}")


class ConversationalRAGPipeline(RAGPipeline):
    """
    RAG pipeline with conversational capabilities.
    
    Maintains conversation history and context across turns.
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        generator: RAGGenerator,
        context_window: int = 5
    ):
        # Use contextual retriever
        retriever = ContextualRetriever(
            document_store=document_store,
            context_window=context_window
        )
        
        super().__init__(document_store, retriever, generator)
        self.conversation_history = []
    
    def chat(
        self,
        message: str,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, any]:
        """
        Process a conversational message.
        
        Args:
            message: User message
            top_k: Number of documents to retrieve
            
        Returns:
            Response with answer and context
        """
        # Retrieve with conversation context
        retrieval_result = self.retriever.retrieve_with_context(
            query=message,
            top_k=top_k
        )
        
        # Generate response
        generation_result = self.generator.generate(
            query=message,
            retrieval_result=retrieval_result
        )
        
        # Add to conversation history
        self.conversation_history.append({
            'user': message,
            'assistant': generation_result['answer']
        })
        
        result = {
            'message': message,
            'response': generation_result['answer'],
            'retrieved_documents': retrieval_result.documents,
            'conversation_turn': len(self.conversation_history)
        }
        
        return result
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        if isinstance(self.retriever, ContextualRetriever):
            self.retriever.reset_context()


class RAGTrainer:
    """
    Trainer for fine-tuning RAG pipeline components.
    
    Supports:
    - Retriever training with hard negatives
    - Generator fine-tuning on legal QA
    - End-to-end pipeline optimization
    """
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.pipeline = pipeline
        self.device = device
    
    def train_retriever(
        self,
        queries: List[str],
        relevant_doc_ids: List[List[str]],
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """
        Train retriever with query-document pairs.
        
        Args:
            queries: Training queries
            relevant_doc_ids: Lists of relevant document IDs for each query
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print("Training retriever...")
        print(f"Number of queries: {len(queries)}")
        
        # This is a placeholder for retriever training
        # In practice, you would:
        # 1. Sample hard negatives
        # 2. Compute contrastive loss
        # 3. Update retriever model
        
        # For now, we'll just validate the setup
        for query, doc_ids in zip(queries[:5], relevant_doc_ids[:5]):
            results = self.pipeline.retriever.retrieve(query, top_k=10)
            print(f"\nQuery: {query}")
            print(f"Expected relevant: {doc_ids}")
            print(f"Retrieved: {[doc.doc_id for doc in results.documents[:5]]}")
    
    def train_generator(
        self,
        queries: List[str],
        reference_answers: List[str],
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-5
    ):
        """
        Fine-tune generator on query-answer pairs.
        
        Args:
            queries: Training queries
            reference_answers: Reference answers
            num_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print("Training generator...")
        print(f"Number of query-answer pairs: {len(queries)}")
        
        # This is a placeholder for generator training
        # In practice, you would:
        # 1. Retrieve documents for each query
        # 2. Compute generation loss against reference
        # 3. Update generator model
        
        for query, answer in zip(queries[:3], reference_answers[:3]):
            print(f"\nQuery: {query}")
            print(f"Reference: {answer}")
    
    def evaluate(
        self,
        eval_queries: List[str],
        eval_answers: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate pipeline performance.
        
        Args:
            eval_queries: Evaluation queries
            eval_answers: Reference answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating RAG pipeline...")
        
        results = []
        for query in eval_queries:
            result = self.pipeline.query(query)
            results.append(result)
        
        # Compute metrics (placeholder)
        metrics = {
            'num_evaluated': len(eval_queries),
            'avg_num_docs_retrieved': sum(
                r['num_docs_retrieved'] for r in results
            ) / len(results),
            'avg_num_docs_used': sum(
                r['num_docs_used'] for r in results
            ) / len(results)
        }
        
        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        return metrics


def create_legal_rag_pipeline(
    documents: List[Document],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    generation_model: str = "gpt2",
    use_citations: bool = True,
    use_chain_of_thought: bool = False
) -> RAGPipeline:
    """
    Factory function to create a complete RAG pipeline for legal documents.
    
    Args:
        documents: List of legal documents
        embedding_model: Model for document embeddings
        generation_model: Model for text generation
        use_citations: Whether to use citation generator
        use_chain_of_thought: Whether to use CoT reasoning
        
    Returns:
        Configured RAGPipeline
    """
    from .document_store import FAISSStore
    
    # Create document store
    print("Creating document store...")
    document_store = FAISSStore(embedding_model=embedding_model)
    document_store.add_documents(documents)
    
    # Create retriever
    print("Creating retriever...")
    retriever = LegalRetriever(
        document_store=document_store,
        top_k=5,
        use_reranking=True
    )
    
    # Create generator
    print("Creating generator...")
    if use_citations:
        generator = CitationGenerator(model_name=generation_model)
    elif use_chain_of_thought:
        generator = ChainOfThoughtGenerator(model_name=generation_model)
    else:
        generator = RAGGenerator(model_name=generation_model)
    
    # Create pipeline
    print("Creating RAG pipeline...")
    pipeline = RAGPipeline(
        document_store=document_store,
        retriever=retriever,
        generator=generator
    )
    
    print("RAG pipeline created successfully!")
    return pipeline
