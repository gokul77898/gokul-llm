"""API v1 Endpoints - ChromaDB integrated, FAISS removed"""
import logging
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

from src.core.chroma_manager import get_chroma_manager
from src.core.model_selector import get_model_selector
from src.training.training_manager import TrainingManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])

# Request/Response Models
class HealthCheckResponse(BaseModel):
    chroma: str
    retriever: str
    model_selector: str
    pipeline: str
    training: str
    data_loaded: bool
    timestamp: str
    version: str

@router.get("/system/health", response_model=HealthCheckResponse)
async def system_health():
    """System health check endpoint"""
    chroma_manager = get_chroma_manager()
    model_selector = get_model_selector()
    training_mgr = TrainingManager()
    
    # Check ChromaDB
    chroma_status = "ok" if chroma_manager.is_ready() else "not_initialized"
    
    # Check retriever
    retriever_status = "ok" if chroma_manager.get_retriever() else "unavailable"
    
    # Check model selector
    selector_status = "ok" if model_selector else "unavailable"
    
    # Check training
    training_status = training_mgr.get_status()
    training_ready = "initialized" if training_status["overall_status"] == "SETUP_MODE" else "error"
    
    # Check data loaded
    stats = chroma_manager.get_collection_stats()
    data_loaded = stats.get("document_count", 0) > 0
    
    return HealthCheckResponse(
        chroma=chroma_status,
        retriever=retriever_status,
        model_selector=selector_status,
        pipeline="ready",
        training=training_ready,
        data_loaded=data_loaded,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@router.get("/chroma/stats")
async def get_chroma_stats():
    """Get ChromaDB collection statistics"""
    chroma_manager = get_chroma_manager()
    
    if not chroma_manager.is_ready():
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    stats = chroma_manager.get_collection_stats()
    
    return {
        "collection": stats.get("name", "legal_docs"),
        "document_count": stats.get("document_count", 0),
        "status": stats.get("status", "unknown"),
        "embedding_dimension": 384,
        "storage_path": "db_store/chroma",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/model_selector/log")
async def get_model_selection_log(limit: int = 50):
    """Get recent model selection history"""
    selector = get_model_selector()
    log = selector.get_selection_log(limit=limit)
    
    return {
        "count": len(log),
        "selections": log
    }

@router.get("/training/status")
async def get_training_status():
    """Get training pipeline status"""
    training_mgr = TrainingManager()
    status = training_mgr.get_status()
    
    return status

@router.post("/training/start")
async def start_training(training_type: str):
    """Attempt to start training (will be blocked)"""
    logger.error(f"❌ Attempt to start {training_type} training - BLOCKED")
    
    raise HTTPException(
        status_code=403,
        detail=f"Training is disabled in SETUP MODE. {training_type.upper()} training cannot be started."
    )


@router.get("/model/routing_log")
async def get_routing_log(limit: int = Query(100, ge=1, le=1000)):
    """
    Get model routing decision log
    
    Returns recent routing decisions showing which model was selected
    for each query and why.
    
    Args:
        limit: Number of recent entries to return (1-1000, default 100)
    
    Returns:
        List of routing log entries with timestamps and decision reasons
    """
    routing_log_path = Path("logs/model_routing.json")
    
    if not routing_log_path.exists():
        return {
            "count": 0,
            "entries": [],
            "message": "No routing log available yet"
        }
    
    try:
        with open(routing_log_path, 'r') as f:
            log_entries = json.load(f)
        
        # Get last N entries
        recent_entries = log_entries[-limit:] if len(log_entries) > limit else log_entries
        
        # Reverse to show most recent first
        recent_entries.reverse()
        
        return {
            "count": len(recent_entries),
            "total_logged": len(log_entries),
            "entries": recent_entries,
            "log_file": str(routing_log_path)
        }
    
    except Exception as e:
        logger.error(f"Failed to read routing log: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read routing log: {str(e)}"
        )
