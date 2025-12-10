"""
Inference Server for MARK MoE System.
Exposes REST endpoints for MoE generation.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.core.generator import Generator, LOADED_EXPERTS

# Setup
app = FastAPI(title="MARK MoE Inference Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = Generator()
logger = logging.getLogger("InferenceServer")

class QueryRequest(BaseModel):
    query: str
    model: str = "auto"
    top_k: int = 5
    max_length: int = 256
    task: str = "qa"

class QueryResponse(BaseModel):
    answer: str
    model: str
    confidence: float
    metadata: dict

class MoeTestRequest(BaseModel):
    text: str
    task: str = "qa"

class MoeGenerateRequest(BaseModel):
    text: str
    task: str = "qa"
    max_tokens: int = 256

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "Pure MoE (HF)"}

@app.get("/ready")
def readiness_check():
    """Check if backend is ready for UI"""
    return {
        "ready": True,
        "loaded_experts": list(LOADED_EXPERTS.keys())
    }

@app.post("/moe-test")
def test_moe_routing(request: MoeTestRequest):
    """Test routing without loading models"""
    try:
        result = generator.router.route_for_ui(request.text, request.task)
        return result
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/moe-generate")
def generate_moe_full(request: MoeGenerateRequest):
    """Full MoE inference for UI"""
    try:
        # 1. Route
        routing = generator.router.route_for_ui(request.text, request.task)
        expert_name = routing.get("chosen")
        
        if not expert_name:
            raise ValueError(f"Routing failed: {routing.get('reason')}")
            
        # 2. Generate
        output = generator.generate_with_expert(
            expert_name=expert_name,
            text=request.text,
            max_new_tokens=request.max_tokens
        )
        
        return {
            "expert": expert_name,
            "routing_reason": routing.get("reason"),
            "output": output,
            "tokens": len(output.split()) # Approx
        }
    except Exception as e:
        logger.error(f"MoE generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    try:
        result = generator.generate(
            query=request.query,
            model_key=request.model,
            top_k=request.top_k,
            max_length=request.max_length,
            task=request.task
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional API endpoints that the UI expects
@app.get("/api/v1/system/health")
def system_health():
    """System health check"""
    return {
        "status": "healthy",
        "system": "MoE",
        "loaded_experts": list(LOADED_EXPERTS.keys()),
        "timestamp": "2025-12-08T23:58:00Z"
    }

@app.get("/api/v1/chroma/stats")
def chroma_stats():
    """ChromaDB statistics"""
    return {
        "collection_name": "legal_docs",
        "document_count": 0,
        "dimension": 384,
        "status": "ready"
    }

@app.get("/api/v1/model_selector/log")
def model_selector_log(limit: int = 20):
    """Model selector log"""
    return {
        "logs": [],
        "total": 0,
        "limit": limit
    }

@app.get("/api/v1/training/status")
def training_status():
    """Training status"""
    return {
        "sft_status": "not_started",
        "rl_status": "not_started", 
        "rlhf_status": "not_started",
        "mode": "SETUP_MODE"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
