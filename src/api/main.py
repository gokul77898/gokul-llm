"""FastAPI Production Server for MARK System"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

from src.inference.embedding_service import get_embedding
from src.common import init_logger
from src.core.chroma_manager import get_chroma_manager
from src.graph.graph_rag_filter import GraphRAGFilter
from src.graph.legal_graph_traverser import LegalGraphTraverser
from src.graph.legal_graph_builder import LegalGraphBuilder
import os
import requests

logger = init_logger("api_server")

# Remote service URLs
EMBEDDING_SERVICE_URL = "https://omilosaisolutions-indian-legal-encoder-8b.hf.space/encode"
HF_ENDPOINT_URL = os.environ.get("HF_ENDPOINT_URL", "")

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
    graph_allowed_docs: int = 0
    graph_excluded_docs: int = 0
    grounded: bool = True
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


# Initialize state
state = None





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
        Process a query using RAG + Graph-RAG + Decoder
        
        CANONICAL PIPELINE:
        1. Validate input (query, top_k)
        2. (Optional) Call encoder → embedding
        3. Retrieve documents from ChromaDB → REFUSE if zero
        4. Run Graph-RAG filtering → REFUSE if zero allowed
        5. Build evidence block from allowed_chunks
        6. Inject evidence into USER message
        7. Call decoder (UNCHANGED)
        8. Validate answer grounding
        9. Return structured response
        """
        import time
        start_time = time.time()
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: VALIDATE INPUT
        # ═══════════════════════════════════════════════════════════════
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
        top_k = min(request.top_k, 10)  # Cap at 10
        logger.info(f"Query: {request.query[:100]}... (RAG + Graph-RAG pipeline)")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: (OPTIONAL) ENCODER EMBEDDING
        # ═══════════════════════════════════════════════════════════════
        embedding_status = "skipped"
        try:
            embedding = get_embedding(request.query)
            embedding_status = "used"
            logger.info(f"Encoder: OK (dim={len(embedding)})")
        except Exception as e:
            logger.warning(f"Encoder failed (non-blocking): {e}")
            embedding_status = "failed"
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: RAG RETRIEVAL FROM CHROMADB (REQUIRED)
        # ═══════════════════════════════════════════════════════════════
        chroma_manager = get_chroma_manager()
        if not chroma_manager.is_ready():
            logger.error("ChromaDB not initialized - HARD FAILURE")
            raise HTTPException(status_code=500, detail="ChromaDB not initialized. Cannot proceed without retrieval.")
        
        retriever = chroma_manager.get_retriever()
        if retriever is None:
            logger.error("ChromaDB retriever not available - HARD FAILURE")
            raise HTTPException(status_code=500, detail="ChromaDB retriever not available.")
        
        # Retrieve documents
        retrieved_results = retriever.query(request.query, top_k=top_k)
        
        if not retrieved_results:
            logger.warning(f"Zero documents retrieved for query: {request.query[:50]}")
            raise HTTPException(status_code=400, detail="No relevant legal documents found. Please refine your query.")
        
        logger.info(f"Retrieved {len(retrieved_results)} documents from ChromaDB")
        
        # Convert to dict format for Graph-RAG
        retrieved_chunks = []
        for r in retrieved_results:
            chunk = {
                "id": r.id,
                "text": r.text,
                "score": r.score,
                "act": r.metadata.get("act", ""),
                "section": r.metadata.get("section", ""),
                "source": r.metadata.get("source", ""),
                "doc_type": r.metadata.get("doc_type", ""),
                "metadata": r.metadata
            }
            retrieved_chunks.append(chunk)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: GRAPH-RAG FILTERING (REQUIRED)
        # ═══════════════════════════════════════════════════════════════
        try:
            # Initialize graph components
            graph_builder = LegalGraphBuilder()
            traverser = LegalGraphTraverser(graph_builder.graph)
            graph_filter = GraphRAGFilter(traverser)
            
            # Filter chunks using graph constraints
            filter_result = graph_filter.filter_chunks(request.query, retrieved_chunks)
            
            allowed_chunks = filter_result.allowed_chunks
            excluded_count = filter_result.total_excluded
            
            logger.info(f"Graph-RAG: {len(allowed_chunks)} allowed, {excluded_count} excluded")
            
        except Exception as graph_error:
            logger.error(f"Graph-RAG filter failed - HARD FAILURE: {graph_error}")
            raise HTTPException(status_code=500, detail=f"Graph-RAG filter failed: {str(graph_error)}")
        
        # Check if all chunks were filtered out
        if not allowed_chunks:
            logger.warning(f"All chunks filtered by Graph-RAG for query: {request.query[:50]}")
            raise HTTPException(
                status_code=400, 
                detail="Query blocked by legal consistency rules. All retrieved documents were filtered due to section/case inconsistencies."
            )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: BUILD EVIDENCE BLOCK (STRICT FORMAT)
        # ═══════════════════════════════════════════════════════════════
        evidence_lines = ["[LEGAL EVIDENCE]"]
        for i, chunk in enumerate(allowed_chunks[:5]):  # Max 5 chunks
            source_info = chunk.get("source", "Unknown")
            act = chunk.get("act", "")
            section = chunk.get("section", "")
            
            source_label = f"Source {i+1}"
            if act and section:
                source_label += f" ({act} Section {section})"
            elif source_info:
                source_label += f" ({source_info})"
            
            evidence_lines.append(f"\n{source_label}:")
            evidence_lines.append(chunk["text"])
        
        evidence_block = "\n".join(evidence_lines)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: BUILD USER MESSAGE WITH EVIDENCE
        # ═══════════════════════════════════════════════════════════════
        user_message_with_evidence = f"""Answer the question strictly using the legal evidence below.
If the answer is not fully supported by the evidence, say so explicitly.

{evidence_block}

Question:
{request.query}"""
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 7: CALL DECODER (UNCHANGED PARAMETERS)
        # ═══════════════════════════════════════════════════════════════
        hf_token = os.environ.get("HF_TOKEN", "")
        if not HF_ENDPOINT_URL:
            raise HTTPException(status_code=500, detail="HF_ENDPOINT_URL not configured")
        if not hf_token:
            raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in Indian criminal law. Answer strictly according to the Code of Criminal Procedure, 1973 and Indian statutes. Do NOT confuse CrPC sections. If the question refers to a specific section, explain ONLY that section. If unsure or if the section does not match the explanation, clearly say you are unsure."
                },
                {
                    "role": "system",
                    "content": "Important rules: Section 436 CrPC = bail in bailable offences (absolute right). Section 437 = bail in non-bailable offences. Section 389 = suspension of sentence and bail pending appeal. Never mix these sections."
                },
                {"role": "user", "content": user_message_with_evidence}
            ]
            
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "Qwen/Qwen2.5-32B-Instruct",
                "lora": "nyayamitra",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            decoder_response = requests.post(
                f"{HF_ENDPOINT_URL}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            decoder_response.raise_for_status()
            result = decoder_response.json()
            
            answer = result["choices"][0]["message"]["content"]
            
        except Exception as decoder_error:
            logger.error(f"Decoder failed: {decoder_error}")
            raise HTTPException(status_code=500, detail=f"Decoder failed: {str(decoder_error)}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 8: POST-GENERATION VALIDATION (GROUNDING CHECK)
        # ═══════════════════════════════════════════════════════════════
        grounded = True
        
        # Check if answer references evidence (basic heuristic)
        answer_lower = answer.lower()
        evidence_keywords = []
        for chunk in allowed_chunks[:5]:
            # Extract key terms from evidence
            text_words = chunk["text"].lower().split()[:20]
            evidence_keywords.extend(text_words)
        
        # Count how many evidence keywords appear in answer
        keyword_matches = sum(1 for kw in evidence_keywords if kw in answer_lower and len(kw) > 4)
        
        if keyword_matches < 3:
            # Answer may not be grounded in evidence
            grounded = False
            logger.warning(f"Answer may not be grounded (keyword_matches={keyword_matches})")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 9: RETURN STRUCTURED RESPONSE
        # ═══════════════════════════════════════════════════════════════
        latency = time.time() - start_time
        
        logger.info(f"Query completed: retrieved={len(retrieved_chunks)}, allowed={len(allowed_chunks)}, grounded={grounded}, latency={latency:.2f}s")
        
        return QueryResponse(
            answer=answer,
            query=request.query,
            model="Qwen2.5-32B-Instruct + LoRA",
            auto_model_used="nyayamitra",
            retrieved_docs=len(retrieved_chunks),
            graph_allowed_docs=len(allowed_chunks),
            graph_excluded_docs=excluded_count,
            grounded=grounded,
            confidence=0.9 if grounded else 0.6,
            latency=latency,
            ensemble={},
            metadata={
                "encoder_status": embedding_status,
                "decoder": "hf_endpoint",
                "lora": "nyayamitra",
                "rag": "chromadb",
                "graph_filter": "active",
                "evidence_chunks": len(allowed_chunks)
            },
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/rag-search", response_model=RAGSearchResponse)
    async def rag_search(request: RAGSearchRequest):
        """
        Search for documents using RAG retrieval
        
        This endpoint retrieves relevant documents without generation.
        """
        try:
            logger.info(f"RAG search: {request.query[:100]}...")
            
            # Ensure ChromaDB retriever is initialized
            if state.retriever is None:
                initialize_chromadb()
            
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
            
            # Load model if not cached - use local model loader
            if request.model not in state.models_cache:
                loader = ModelLoader(device="cpu")
                # For local inference, models must be explicitly loaded
                raise HTTPException(
                    status_code=501, 
                    detail=f"Direct model generation not supported. Use /query endpoint with RAG pipeline."
                )
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


if FASTAPI_AVAILABLE:
    
    class E2ETestRequest(BaseModel):
        """E2E test request - bypasses RAG"""
        query: str = Field(..., description="User query")
    
    class E2ETestResponse(BaseModel):
        """E2E test response"""
        query: str
        embedding_dim: int
        answer: str
        embedding_status: str
        decoder_status: str
        timestamp: str
    
    @app.post("/test-e2e", response_model=E2ETestResponse)
    async def test_e2e(request: E2ETestRequest):
        """
        End-to-end test endpoint - bypasses RAG
        
        1. Calls encoder Space for embedding (validates encoder)
        2. Calls HF Inference Endpoint for generation (validates decoder)
        3. Returns both results
        """
        embedding_status = "not_called"
        decoder_status = "not_called"
        embedding_dim = 0
        answer = ""
        
        # Step 1: Call embedding service
        try:
            embedding_response = requests.post(
                EMBEDDING_SERVICE_URL,
                json={"query": request.query},
                timeout=30
            )
            embedding_response.raise_for_status()
            embedding_data = embedding_response.json()
            embedding_dim = embedding_data.get("embedding_dim", 0)
            embedding_status = f"ok (dim={embedding_dim})"
        except Exception as e:
            embedding_status = f"error: {str(e)[:100]}"
        
        # Step 2: Get embedding and call HF Inference Endpoint
        hf_token = os.environ.get("HF_TOKEN", "")
        if not HF_ENDPOINT_URL:
            decoder_status = "error: HF_ENDPOINT_URL not set"
        elif not hf_token:
            decoder_status = "error: HF_TOKEN not set"
        else:
            try:
                # Get embedding from encoder
                embedding = get_embedding(request.query)
                
                # Create debug context
                encoder_debug_context = f"""
[ENCODER DEBUG]
embedding_length={len(embedding)}
embedding_preview={embedding[:5]}
"""
                
                # Build messages with strict legal grounding
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert in Indian criminal law. Answer strictly according to the Code of Criminal Procedure, 1973 and Indian statutes. Do NOT confuse CrPC sections. If the question refers to a specific section, explain ONLY that section. If unsure or if the section does not match the explanation, clearly say you are unsure."
                    },
                    {
                        "role": "system",
                        "content": "Important rules: Section 436 CrPC = bail in bailable offences (absolute right). Section 437 = bail in non-bailable offences. Section 389 = suspension of sentence and bail pending appeal. Never mix these sections."
                    },
                    {"role": "system", "content": encoder_debug_context},
                    {"role": "user", "content": request.query}
                ]
                
                headers = {
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "Qwen/Qwen2.5-32B-Instruct",
                    "lora": "nyayamitra",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 200
                }
                decoder_response = requests.post(
                    f"{HF_ENDPOINT_URL}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                decoder_response.raise_for_status()
                result = decoder_response.json()
                
                # Extract answer from OpenAI format
                answer = result["choices"][0]["message"]["content"]
                
                decoder_status = "ok"
            except Exception as e:
                decoder_status = f"error: {str(e)[:100]}"
        
        return E2ETestResponse(
            query=request.query,
            embedding_dim=embedding_dim,
            answer=answer,
            embedding_status=embedding_status,
            decoder_status=decoder_status,
            timestamp=datetime.now().isoformat()
        )


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
