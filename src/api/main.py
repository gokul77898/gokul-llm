"""FastAPI Production Server for MARK System"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

from src.core import load_model, get_registry
from src.pipelines.fusion_pipeline import FusionPipeline
from src.pipelines.auto_pipeline import AutoPipeline
from src.feedback.worker import FeedbackWorker
from src.rag.reranker import CrossEncoderReranker
from src.rag.grounded_generator import GroundedAnswerGenerator
from src.core.chroma_manager import get_chroma_manager
from src.core.response_formatter import format_chatgpt_response
from src.common.config import get_config
from db.chroma import VectorRetriever
from src.common import init_logger
import os

logger = init_logger("api_server")

# Initialize FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="MARK Legal AI System",
        description="Production API for MARK (Modular Architecture for Reinforced Knowledge)",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request"""
    query: str = Field(..., description="User query")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    max_length: int = Field(default=256, description="Maximum generation length")


class QueryResponse(BaseModel):
    """Query response"""
    answer: str
    query: str
    model: str
    auto_model_used: str
    retrieved_docs: int
    confidence: float
    latency: float
    ensemble: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str


class RAGSearchRequest(BaseModel):
    """RAG search request"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results")


class RAGSearchResponse(BaseModel):
    """RAG search response"""
    query: str
    results: List[Dict[str, Any]]
    num_results: int
    timestamp: str


class GenerateRequest(BaseModel):
    """Generation request"""
    prompt: str = Field(..., description="Input prompt")
    model: str = Field(default="mamba", description="Model to use")
    max_length: int = Field(default=256, description="Maximum length")
    temperature: float = Field(default=0.7, description="Sampling temperature")


class GenerateResponse(BaseModel):
    """Generation response"""
    generated_text: str
    model: str
    prompt_length: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: List[str]
    timestamp: str


# Global state
class APIState:
    """Global API state"""
    retriever: Optional[VectorRetriever] = None
    fusion_pipeline: Optional[FusionPipeline] = None
    fusion_pipelines: Dict[str, FusionPipeline] = {}
    auto_pipeline: Optional[AutoPipeline] = None
    chroma_manager: Optional[Any] = None
    models_cache: Dict[str, Any] = {}
    feedback_worker: Optional[FeedbackWorker] = None
    reranker: Optional[CrossEncoderReranker] = None
    grounded_generator: Optional[GroundedAnswerGenerator] = None

# Initialize state
state = APIState()

# ChromaDB is initialized automatically via chroma_manager
def initialize_chromadb():
    """Initialize ChromaDB on startup"""
    try:
        logger.info("Initializing ChromaDB...")
        chroma_manager = get_chroma_manager()
        if chroma_manager.is_ready():
            state.chroma_manager = chroma_manager
            state.retriever = chroma_manager.get_retriever()
            stats = chroma_manager.get_collection_stats()
            doc_count = stats.get('document_count', 0)
            logger.info(f"ChromaDB initialized (collection: legal_docs, documents: {doc_count})")
            return True
        else:
            logger.warning("ChromaDB not ready")
            return False
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
        return False

def get_fusion_pipeline(model: str = "mamba") -> FusionPipeline:
    """Get or create fusion pipeline with ChromaDB retriever"""
    if state.fusion_pipeline is None or state.fusion_pipeline.generator_model_name != model:
        logger.info(f"Initializing fusion pipeline with model: {model}")
        
        # Ensure ChromaDB is initialized
        if state.retriever is None:
            initialize_chromadb()
        
        if state.retriever is not None:
            # Create pipeline with custom retriever
            try:
                generator, tokenizer, device = load_model(model)
                state.fusion_pipeline = FusionPipeline(
                    generator_model=model,
                    retriever_model="rag_encoder",
                    device="cpu",
                    top_k=5
                )
                # Override retriever with our ChromaDB-based one
                state.fusion_pipeline.retriever = state.retriever
                logger.info("Fusion pipeline initialized with ChromaDB retriever")
            except Exception as e:
                logger.error(f"Failed to create fusion pipeline: {e}")
                raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
        else:
            logger.warning("No ChromaDB retriever available, creating pipeline without retriever")
            state.fusion_pipeline = FusionPipeline(
                generator_model=model,
                retriever_model="rag_encoder",
                device="cpu",
                top_k=5
            )
    
    return state.fusion_pipeline


# Startup event to initialize ChromaDB
if FASTAPI_AVAILABLE:
    @app.on_event("startup")
    async def startup_event():
        """Initialize ChromaDB on application startup"""
        logger.info("Running startup initialization...")
        initialize_chromadb()
        logger.info("Startup complete")


# API Endpoints
if FASTAPI_AVAILABLE:
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {
            "message": "MARK Legal AI System API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    @app.post("/feedback")
    async def submit_feedback(feedback: Dict[str, Any]):
        """Submit user feedback for safe SFT training"""
        try:
            # Initialize feedback worker if needed
            if state.feedback_worker is None:
                state.feedback_worker = FeedbackWorker()
            
            # Save feedback atomically
            success = state.feedback_worker.save_feedback(feedback)
            
            if success:
                buffer_size = state.feedback_worker._get_buffer_size()
                return {
                    "status": "accepted",
                    "message": "Feedback recorded. Model won't be redeployed automatically; your feedback goes to training buffer for safe review.",
                    "buffer_size": buffer_size,
                    "threshold": FeedbackWorker.SFT_BUFFER_THRESHOLD
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to save feedback")
        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/feedback/review")
    async def get_feedback_review(limit: int = 50):
        """Get feedback items needing human review"""
        try:
            if state.feedback_worker is None:
                state.feedback_worker = FeedbackWorker()
            
            review_items = state.feedback_worker.get_review_queue(limit=limit)
            
            return {
                "count": len(review_items),
                "items": review_items
            }
        except Exception as e:
            logger.error(f"Review queue error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/retrain/trigger")
    async def trigger_retrain():
        """Trigger SFT bundle creation (operator only)"""
        try:
            if state.feedback_worker is None:
                state.feedback_worker = FeedbackWorker()
            
            buffer_size = state.feedback_worker._get_buffer_size()
            
            if buffer_size == 0:
                return {
                    "status": "skipped",
                    "message": "No training examples in buffer"
                }
            
            bundle_path = state.feedback_worker.create_sft_bundle()
            
            if bundle_path:
                return {
                    "status": "ready",
                    "message": f"SFT bundle created with {buffer_size} examples",
                    "bundle_path": bundle_path,
                    "command": f"python -m src.training.sft_train --data {bundle_path} --out checkpoints/training/sft/sft_incremental_$(date +%Y%m%d_%H%M%S).pt"
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to create SFT bundle")
        except Exception as e:
            logger.error(f"Retrain trigger error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        registry = get_registry()
        models = list(registry.list_models().keys())
        
        # Check ChromaDB status
        chroma_status = "loaded" if state.chroma_manager is not None else "not_loaded"
        stats = state.chroma_manager.get_collection_stats() if state.chroma_manager else {}
        doc_count = stats.get('document_count', 0)
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=models,
            timestamp=datetime.now().isoformat()
        )
    
    @app.get("/models")
    async def get_models():
        """Get available models"""
        registry = get_registry()
        models = list(registry.list_models().keys())
        
        # Check ChromaDB status
        chroma_loaded = state.chroma_manager is not None
        stats = state.chroma_manager.get_collection_stats() if state.chroma_manager else {}
        doc_count = stats.get('document_count', 0)
        
        return {
            "models": models,
            "chroma_loaded": chroma_loaded,
            "document_count": doc_count,
            "chroma_path": "db_store/chroma"
        }
    
    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """
        Process a query using RAG + Generation
        
        This endpoint:
        1. Retrieves relevant documents
        2. Generates an answer using the specified model
        3. Returns the answer with metadata
        """
        try:
            logger.info(f"Query request: {request.query[:100]}... (auto pipeline)")
            
            # Validate input
            if not request.query.strip():
                raise HTTPException(status_code=400, detail="Query cannot be empty.")
            
            # Check if ChromaDB is initialized
            if state.chroma_manager is None:
                initialize_chromadb()
            if state.chroma_manager is None:
                raise HTTPException(status_code=404, detail="ChromaDB not initialized. Please check system status.")
            
            # Get fusion pipeline for auto pipeline initialization
            if state.fusion_pipeline is None:
                state.fusion_pipeline = get_fusion_pipeline("rl_trained")
            
            if state.fusion_pipeline is None:
                raise HTTPException(status_code=500, detail="Training model not loaded. Please load checkpoint.")
            
            # Process query with auto pipeline
            if state.retriever is not None:
                logger.info("Processing query with AUTO pipeline")
                
                # Initialize auto pipeline if not exists
                if state.auto_pipeline is None:
                    state.auto_pipeline = AutoPipeline(state.fusion_pipeline, state.retriever)
                    logger.info("Auto pipeline initialized")
                
                # Use fully automated pipeline
                try:
                    # Run fully automated pipeline
                    auto_result = state.auto_pipeline.process_query(
                        query=request.query,
                        top_k=request.top_k
                    )
                    
                    logger.info(f"Auto pipeline result - Model: {auto_result['auto_model_used']}, Confidence: {auto_result['confidence']:.3f}")
                    
                    # Format response in ChatGPT style
                    formatted_answer = format_chatgpt_response(
                        query=request.query,
                        answer=auto_result["answer"],
                        retrieved_docs=auto_result.get("sources", []),
                        confidence=auto_result["confidence"]
                    )
                    
                    return QueryResponse(
                        answer=formatted_answer,
                        query=request.query,
                        model=auto_result["model"],
                        auto_model_used=auto_result["auto_model_used"],
                        retrieved_docs=auto_result["retrieved_docs"],
                        confidence=auto_result["confidence"],
                        latency=auto_result["latency"],
                        ensemble=auto_result["ensemble"],
                        metadata=auto_result["metadata"],
                        timestamp=datetime.now().isoformat()
                    )
                    
                except Exception as pipeline_error:
                    logger.error(f"Pipeline query failed: {pipeline_error}", exc_info=True)
                    
                    # Fallback: Direct retrieval + simple response
                    retrieval_result = state.retriever.retrieve(
                        query=request.query,
                        top_k=request.top_k
                    )
                    retrieved_docs = retrieval_result.documents
                    
                    fallback_answer = f"Auto pipeline error. Retrieved {len(retrieved_docs)} documents but generation failed: {str(pipeline_error)[:100]}..."
                    
                    return QueryResponse(
                        answer=fallback_answer,
                        query=request.query,
                        model="auto",
                        auto_model_used="Error - Pipeline Failed",
                        retrieved_docs=len(retrieved_docs),
                        confidence=0.2,
                        latency=0.0,
                        ensemble={},
                        metadata={},
                        timestamp=datetime.now().isoformat()
                    )
            else:
                # No retriever available
                return QueryResponse(
                    answer="No documents available for retrieval.",
                    query=request.query,
                    model="auto",
                    auto_model_used="No Retriever Available",
                    retrieved_docs=0,
                    confidence=0.1,
                    latency=0.0,
                    ensemble={},
                    metadata={},
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/rag-search", response_model=RAGSearchResponse)
    async def rag_search(request: RAGSearchRequest):
        """
        Search for documents using RAG retrieval
        
        This endpoint retrieves relevant documents without generation.
        """
        try:
            logger.info(f"RAG search: {request.query[:100]}...")
            
            # Load retriever if not loaded
            if state.retriever is None:
                state.retriever, _, _ = load_model("rag_encoder", device="cpu")
            
            # Search
            results = state.retriever.search(request.query, top_k=request.top_k)
            
            # Format results
            formatted_results = []
            for r in results:
                formatted_results.append({
                    'text': r['document'].get('text', ''),
                    'score': r.get('score', 0.0),
                    'metadata': r.get('metadata', {}),
                    'index': r.get('index', -1)
                })
            
            return RAGSearchResponse(
                query=request.query,
                results=formatted_results,
                num_results=len(formatted_results),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generate text using a specified model
        
        This endpoint generates text without retrieval.
        """
        try:
            logger.info(f"Generate request with {request.model}")
            
            # Load model if not cached
            if request.model not in state.models_cache:
                model, tokenizer, device = load_model(request.model, device="cpu")
                state.models_cache[request.model] = (model, tokenizer, device)
            else:
                model, tokenizer, device = state.models_cache[request.model]
            
            # Generate
            if request.model == "mamba":
                import torch
                encoding = tokenizer.encode(request.prompt, return_tensors=False)
                input_ids = torch.tensor([encoding['input_ids'][:request.max_length]]).to(device)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids, task="generation")
                    logits = outputs['logits']
                    predicted_ids = torch.argmax(logits, dim=-1)
                    generated_text = tokenizer.decode(predicted_ids[0].cpu().tolist())
            
            elif request.model == "transformer":
                import torch
                encoding = tokenizer.tokenize(
                    request.prompt,
                    max_length=request.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0, predicted_class].item()
                    generated_text = f"Class {predicted_class} (confidence: {confidence:.2f})"
            
            else:
                raise ValueError(f"Unknown model: {request.model}")
            
            return GenerateResponse(
                generated_text=generated_text,
                model=request.model,
                prompt_length=len(request.prompt),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models", response_model=Dict[str, Any])
    async def list_models():
        """List available models"""
        registry = get_registry()
        models = registry.list_models()
        
        models_info = {}
        for name, info in models.items():
            models_info[name] = {
                'architecture': info.architecture,
                'description': info.description,
                'config_path': info.config_path,
                'checkpoint_path': info.checkpoint_path
            }
        
        return {
            'models': models_info,
            'count': len(models_info)
        }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server"""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    import uvicorn
    
    logger.info(f"Starting MARK API server on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MARK API server")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument('--port', type=int, default=8000, help="Port to bind to")
    parser.add_argument('--reload', action='store_true', help="Enable auto-reload")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, reload=args.reload)
