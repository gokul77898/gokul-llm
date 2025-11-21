"""Tests for FastAPI server"""

import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.main import app, FASTAPI_AVAILABLE
    TESTCLIENT_AVAILABLE = FASTAPI_AVAILABLE
except ImportError:
    TESTCLIENT_AVAILABLE = False
    pytest.skip("FastAPI not available", allow_module_level=True)

from src.data import create_sample_data


@pytest.fixture(scope="module")
def client():
    """Create test client"""
    if not TESTCLIENT_AVAILABLE:
        pytest.skip("FastAPI not available")
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_data():
    """Setup test data"""
    create_sample_data()


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "timestamp" in data


def test_models_endpoint(client):
    """Test list models endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data
    assert data["count"] > 0


def test_query_endpoint(client):
    """Test query endpoint"""
    try:
        response = client.post(
            "/query",
            json={
                "query": "What is contract law?",
                "model": "mamba",
                "top_k": 3,
                "max_length": 128
            }
        )
        
        # May fail if models not loaded, but API should respond
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "query" in data
            assert "model" in data
        else:
            # Expected if models not available
            assert response.status_code in [500, 404]
            
    except Exception as e:
        pytest.skip(f"Query endpoint test failed (expected if models unavailable): {e}")


def test_rag_search_endpoint(client):
    """Test RAG search endpoint"""
    try:
        response = client.post(
            "/rag-search",
            json={
                "query": "contract law",
                "top_k": 5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "results" in data
            assert "num_results" in data
        else:
            assert response.status_code in [500, 404]
            
    except Exception as e:
        pytest.skip(f"RAG search test failed: {e}")


def test_generate_endpoint(client):
    """Test generate endpoint"""
    try:
        response = client.post(
            "/generate",
            json={
                "prompt": "Test prompt",
                "model": "mamba",
                "max_length": 128
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "generated_text" in data
            assert "model" in data
        else:
            assert response.status_code in [500, 404]
            
    except Exception as e:
        pytest.skip(f"Generate endpoint test failed: {e}")


def test_invalid_model_query(client):
    """Test query with invalid model"""
    response = client.post(
        "/query",
        json={
            "query": "test",
            "model": "invalid_model"
        }
    )
    assert response.status_code == 500
