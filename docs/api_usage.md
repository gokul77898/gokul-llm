## API Usage Guide

## Starting the Server

### Basic Start
```bash
python -m src.api.main --host 0.0.0.0 --port 8000
```

### Development Mode (with auto-reload)
```bash
python -m src.api.main --host localhost --port 8000 --reload
```

### Using uvicorn directly
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check if the API is running and which models are loaded.

**Example**:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": ["mamba", "transformer", "rag_encoder", "rl_trained"],
  "timestamp": "2025-01-01T12:00:00"
}
```

### 2. Query (RAG + Generation)

**Endpoint**: `POST /query`

**Description**: Process a query using RAG retrieval and generation.

**Request Body**:
```json
{
  "query": "What is contract law?",
  "model": "mamba",
  "top_k": 5,
  "max_length": 256
}
```

**Parameters**:
- `query` (required): User question
- `model` (optional): "mamba" or "transformer" (default: "mamba")
- `top_k` (optional): Number of documents to retrieve (default: 5)
- `max_length` (optional): Maximum generation length (default: 256)

**Example**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is contract law?",
    "model": "mamba",
    "top_k": 5
  }'
```

**Response**:
```json
{
  "answer": "Contract law governs agreements between parties...",
  "query": "What is contract law?",
  "model": "mamba",
  "retrieved_docs": 5,
  "confidence": 0.85,
  "timestamp": "2025-01-01T12:00:00"
}
```

### 3. RAG Search

**Endpoint**: `POST /rag-search`

**Description**: Search for documents without generation.

**Request Body**:
```json
{
  "query": "contract law",
  "top_k": 10
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/rag-search \
  -H "Content-Type: application/json" \
  -d '{"query": "contract law", "top_k": 10}'
```

**Response**:
```json
{
  "query": "contract law",
  "results": [
    {
      "text": "Contract law governs agreements...",
      "score": 0.92,
      "metadata": {"category": "contract"},
      "index": 0
    }
  ],
  "num_results": 10,
  "timestamp": "2025-01-01T12:00:00"
}
```

### 4. Generate

**Endpoint**: `POST /generate`

**Description**: Generate text using a specified model (without retrieval).

**Request Body**:
```json
{
  "prompt": "Summarize the following contract...",
  "model": "mamba",
  "max_length": 200,
  "temperature": 0.7
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain contract law",
    "model": "mamba",
    "max_length": 200
  }'
```

**Response**:
```json
{
  "generated_text": "Contract law is a body of law...",
  "model": "mamba",
  "prompt_length": 20,
  "timestamp": "2025-01-01T12:00:00"
}
```

### 5. List Models

**Endpoint**: `GET /models`

**Description**: List all available models.

**Example**:
```bash
curl http://localhost:8000/models
```

**Response**:
```json
{
  "models": {
    "mamba": {
      "architecture": "mamba",
      "description": "Mamba hierarchical attention model",
      "config_path": "configs/mamba_train.yaml",
      "checkpoint_path": "checkpoints/mamba/best_model.pt"
    },
    "transformer": {
      "architecture": "transformer",
      "description": "BERT-based transformer",
      "config_path": "configs/transfer_train.yaml",
      "checkpoint_path": "checkpoints/transfer/best_model.pt"
    }
  },
  "count": 4
}
```

## Python Client Example

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Query with RAG
response = requests.post(
    f"{BASE_URL}/query",
    json={
        "query": "What is contract law?",
        "model": "mamba",
        "top_k": 5
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")

# RAG search
response = requests.post(
    f"{BASE_URL}/rag-search",
    json={
        "query": "contract law",
        "top_k": 3
    }
)
results = response.json()
print(f"Found {results['num_results']} documents")

# Generate text
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "Explain legal principles",
        "model": "mamba",
        "max_length": 200
    }
)
generation = response.json()
print(f"Generated: {generation['generated_text']}")
```

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (endpoint doesn't exist)
- `500`: Internal Server Error (processing failed)

### Example Error
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "model": "invalid_model"}'
```

Response:
```json
{
  "detail": "Model 'invalid_model' not found in registry"
}
```

## Rate Limiting

(To be implemented)

## Authentication

(To be implemented)

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Browse all endpoints
- See request/response schemas
- Test API calls directly in the browser
- Download OpenAPI specification

## Best Practices

1. **Always check health** before sending requests
2. **Handle errors gracefully** in your client code
3. **Use appropriate timeout** for long-running requests
4. **Batch requests** when possible for efficiency
5. **Cache responses** when appropriate
6. **Monitor performance** and adjust parameters

## Performance Tips

- Use `top_k=3-5` for faster retrieval
- Set `max_length` appropriately (shorter = faster)
- Consider using "transformer" for classification tasks (faster than generation)
- Reuse connections with a session object
- Implement client-side caching for repeated queries

## Troubleshooting

### Connection Refused
- Check if server is running
- Verify host and port
- Check firewall settings

### Slow Responses
- Reduce `top_k` value
- Reduce `max_length`
- Ensure models are properly loaded
- Check system resources (CPU/GPU/RAM)

### Out of Memory
- Use CPU instead of GPU for inference
- Reduce batch size
- Restart server to clear cache

## Production Deployment

For production deployment, consider:

1. **Use a production ASGI server** (Gunicorn + Uvicorn)
2. **Add reverse proxy** (Nginx)
3. **Enable HTTPS**
4. **Implement authentication**
5. **Add rate limiting**
6. **Monitor with tools** (Prometheus, Grafana)
7. **Use load balancing** for high traffic
8. **Implement health checks** in orchestration (Kubernetes)

Example production command:
```bash
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```
