"""Unit tests for RAG System"""

import pytest
import torch
import numpy as np
from src.rag.document_store import Document, FAISSStore
from src.rag.retriever import LegalRetriever
from src.rag.generator import RAGGenerator
from src.rag.pipeline import RAGPipeline


class TestDocument:
    """Test suite for Document class"""
    
    def test_document_creation(self):
        """Test document creation"""
        doc = Document(
            content="This is a legal document.",
            metadata={'source': 'test', 'date': '2024-01-01'}
        )
        
        assert doc.content == "This is a legal document."
        assert doc.metadata['source'] == 'test'
        assert doc.doc_id is not None
    
    def test_document_repr(self):
        """Test document representation"""
        doc = Document(content="Test content")
        repr_str = repr(doc)
        
        assert 'Document' in repr_str
        assert 'content_length' in repr_str


class TestFAISSStore:
    """Test suite for FAISS Document Store"""
    
    def test_store_initialization(self):
        """Test store initialization"""
        store = FAISSStore(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert store.dimension > 0
        assert store.index is not None
    
    def test_add_documents(self):
        """Test adding documents"""
        store = FAISSStore()
        
        documents = [
            Document("First legal document about contracts."),
            Document("Second legal document about torts."),
            Document("Third legal document about criminal law.")
        ]
        
        store.add_documents(documents)
        
        assert len(store.documents) == 3
        assert len(store.embeddings) == 3
    
    def test_search(self):
        """Test document search"""
        store = FAISSStore()
        
        documents = [
            Document("Contract law document."),
            Document("Criminal law document."),
            Document("Tort law document.")
        ]
        
        store.add_documents(documents)
        
        # Search for contract-related documents
        results = store.search("contract law", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
    
    def test_metadata_filtering(self):
        """Test search with metadata filtering"""
        store = FAISSStore()
        
        documents = [
            Document("Doc 1", metadata={'category': 'A'}),
            Document("Doc 2", metadata={'category': 'B'}),
            Document("Doc 3", metadata={'category': 'A'})
        ]
        
        store.add_documents(documents)
        
        # Search with filter
        results = store.search(
            "document",
            top_k=5,
            filter_metadata={'category': 'A'}
        )
        
        # Should only return category A documents
        assert all(doc.metadata['category'] == 'A' for doc, _ in results)


class TestLegalRetriever:
    """Test suite for Legal Retriever"""
    
    def test_retriever_initialization(self):
        """Test retriever initialization"""
        store = FAISSStore()
        retriever = LegalRetriever(document_store=store, top_k=5)
        
        assert retriever.top_k == 5
        assert retriever.document_store is store
    
    def test_basic_retrieval(self):
        """Test basic retrieval"""
        store = FAISSStore()
        documents = [
            Document("Contract law basics."),
            Document("Criminal law overview."),
            Document("Tort law principles.")
        ]
        store.add_documents(documents)
        
        retriever = LegalRetriever(document_store=store, top_k=2)
        result = retriever.retrieve("contract law")
        
        assert len(result.documents) <= 2
        assert len(result.scores) <= 2
        assert result.query == "contract law"
    
    def test_query_expansion(self):
        """Test query expansion"""
        store = FAISSStore()
        documents = [Document("Legal document about plaintiff and defendant.")]
        store.add_documents(documents)
        
        retriever = LegalRetriever(document_store=store)
        result = retriever.retrieve("plaintiff", expand_query=True)
        
        assert result.metadata['expand_query'] is True


class TestRAGGenerator:
    """Test suite for RAG Generator"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = RAGGenerator(model_name="gpt2")
        
        assert generator.model is not None
        assert generator.tokenizer is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_generation(self):
        """Test text generation"""
        from src.rag.retriever import RetrievalResult
        
        generator = RAGGenerator(model_name="gpt2")
        
        # Create mock retrieval result
        documents = [Document("Legal document about contracts.")]
        retrieval_result = RetrievalResult(
            documents=documents,
            scores=[0.9],
            query="What is a contract?"
        )
        
        # Generate
        result = generator.generate(
            query="What is a contract?",
            retrieval_result=retrieval_result
        )
        
        assert 'answer' in result
        assert 'query' in result
        assert 'context' in result


class TestRAGPipeline:
    """Test suite for RAG Pipeline"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        store = FAISSStore()
        retriever = LegalRetriever(document_store=store)
        generator = RAGGenerator(model_name="gpt2")
        
        pipeline = RAGPipeline(
            document_store=store,
            retriever=retriever,
            generator=generator
        )
        
        assert pipeline.document_store is store
        assert pipeline.retriever is retriever
        assert pipeline.generator is generator
    
    def test_add_documents(self):
        """Test adding documents to pipeline"""
        store = FAISSStore()
        retriever = LegalRetriever(document_store=store)
        generator = RAGGenerator(model_name="gpt2")
        
        pipeline = RAGPipeline(store, retriever, generator)
        
        documents = [
            Document("Legal document 1."),
            Document("Legal document 2.")
        ]
        
        pipeline.add_documents(documents)
        
        assert len(pipeline.document_store.documents) == 2
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_end_to_end_query(self):
        """Test end-to-end query processing"""
        store = FAISSStore()
        documents = [
            Document("Contracts are agreements between parties."),
            Document("Torts involve civil wrongs.")
        ]
        store.add_documents(documents)
        
        retriever = LegalRetriever(document_store=store, top_k=2)
        generator = RAGGenerator(model_name="gpt2")
        
        pipeline = RAGPipeline(store, retriever, generator)
        
        result = pipeline.query(
            query="What is a contract?",
            top_k=2
        )
        
        assert 'query' in result
        assert 'answer' in result
        assert 'retrieved_documents' in result
        assert len(result['retrieved_documents']) <= 2


@pytest.fixture
def sample_documents():
    """Fixture for sample documents"""
    return [
        Document(
            "A contract is a legally binding agreement between two or more parties.",
            metadata={'category': 'contract_law', 'date': '2024-01-01'}
        ),
        Document(
            "Tort law deals with civil wrongs and remedies.",
            metadata={'category': 'tort_law', 'date': '2024-01-02'}
        ),
        Document(
            "Criminal law defines crimes and their punishments.",
            metadata={'category': 'criminal_law', 'date': '2024-01-03'}
        )
    ]


@pytest.fixture
def sample_document_store(sample_documents):
    """Fixture for document store with sample documents"""
    store = FAISSStore()
    store.add_documents(sample_documents)
    return store


def test_full_rag_workflow(sample_document_store):
    """Test complete RAG workflow"""
    retriever = LegalRetriever(document_store=sample_document_store, top_k=2)
    
    # Retrieve documents
    result = retriever.retrieve("contract law")
    
    assert len(result.documents) > 0
    assert result.scores[0] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
