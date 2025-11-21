"""Tests for RAG indexer"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.common import load_config
from src.data import create_sample_data
from src.rag.indexer import FAISSIndexer, index_documents
from src.rag.eval import recall_at_k, mean_reciprocal_rank, evaluate_rag_pipeline


@pytest.fixture(scope="module")
def temp_dir():
    """Create temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def sample_data(temp_dir):
    """Create sample data"""
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(exist_ok=True)
    create_sample_data(str(data_dir))
    return data_dir


def test_faiss_indexer_creation():
    """Test FAISS indexer creation"""
    indexer = FAISSIndexer(embedding_dim=128)
    # Note: actual dimension may differ if sentence-transformers model is loaded
    assert indexer.index is None


def test_faiss_indexer_build(sample_data):
    """Test building FAISS index"""
    import json
    
    # Load documents
    documents = []
    docs_file = sample_data / "documents.jsonl"
    with open(docs_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Only use 10 documents for testing
                break
            documents.append(json.loads(line))
    
    # Build index
    indexer = FAISSIndexer(embedding_dim=128)
    indexer.build_index(documents)
    
    assert indexer.index is not None
    assert indexer.index.ntotal == len(documents)


def test_faiss_indexer_search(sample_data):
    """Test FAISS index search"""
    import json
    
    # Load documents
    documents = []
    docs_file = sample_data / "documents.jsonl"
    with open(docs_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            documents.append(json.loads(line))
    
    # Build and search
    indexer = FAISSIndexer(embedding_dim=128)
    indexer.build_index(documents)
    
    results = indexer.search("contract law", k=3)
    
    assert len(results) <= 3
    assert len(results) > 0
    assert 'document' in results[0]
    assert 'score' in results[0]


def test_recall_metrics():
    """Test recall calculation"""
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc1', 'doc3', 'doc6']
    
    recall_1 = recall_at_k(retrieved, relevant, 1)
    recall_3 = recall_at_k(retrieved, relevant, 3)
    recall_5 = recall_at_k(retrieved, relevant, 5)
    
    assert 0 <= recall_1 <= 1
    assert 0 <= recall_3 <= 1
    assert 0 <= recall_5 <= 1
    assert recall_3 >= recall_1  # More results should have higher recall


def test_mrr_calculation():
    """Test MRR calculation"""
    retrieved = ['doc1', 'doc2', 'doc3']
    relevant = ['doc2']
    
    mrr = mean_reciprocal_rank(retrieved, relevant)
    
    assert mrr == 0.5  # doc2 is at position 2, so MRR = 1/2


def test_index_save_load(sample_data, temp_dir):
    """Test saving and loading index"""
    import json
    
    # Load documents
    documents = []
    docs_file = sample_data / "documents.jsonl"
    with open(docs_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            documents.append(json.loads(line))
    
    # Build and save
    indexer = FAISSIndexer(embedding_dim=128)
    indexer.build_index(documents)
    
    index_path = Path(temp_dir) / "test.index"
    metadata_path = Path(temp_dir) / "test_meta.json"
    
    indexer.save(str(index_path), str(metadata_path))
    
    assert index_path.exists()
    assert metadata_path.exists()
    
    # Load and verify
    new_indexer = FAISSIndexer(embedding_dim=128)
    new_indexer.load(str(index_path), str(metadata_path))
    
    assert new_indexer.index.ntotal == len(documents)
    assert len(new_indexer.documents) == len(documents)


def test_evaluate_rag_pipeline(sample_data, temp_dir):
    """Test RAG pipeline evaluation"""
    import json
    
    # Create indexer
    documents = []
    docs_file = sample_data / "documents.jsonl"
    with open(docs_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            documents.append(json.loads(line))
    
    indexer = FAISSIndexer(embedding_dim=128)
    indexer.build_index(documents)
    
    # Evaluate
    eval_file = sample_data / "rag_eval.jsonl"
    if eval_file.exists():
        metrics = evaluate_rag_pipeline(indexer, str(eval_file), top_k_values=[1, 3])
        
        assert 'recall@1' in metrics
        assert 'recall@3' in metrics
        assert 'mrr' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
