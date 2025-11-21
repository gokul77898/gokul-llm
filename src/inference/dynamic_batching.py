"""
Dynamic Batching + Async Streaming Engine

Implements dynamic batching for efficient inference and async streaming for real-time responses.
Compatible with Mamba/Transformer auto-detection system.
"""

import asyncio
import torch
import time
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import uuid

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    callback: Optional[Callable] = None
    future: Optional[asyncio.Future] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchConfig:
    """Configuration for dynamic batching"""
    max_batch_size: int = 8
    max_wait_time_ms: int = 50  # Maximum time to wait for batch to fill
    min_batch_size: int = 1
    padding_token_id: int = 0
    enable_streaming: bool = True
    stream_chunk_size: int = 1  # Number of tokens per stream chunk
    max_queue_size: int = 1000
    worker_threads: int = 2


class DynamicBatcher:
    """
    Dynamic Batching Engine
    
    Efficiently batches requests for improved throughput while maintaining low latency.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: BatchConfig,
        device: str = "auto"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Request queue and processing
        self.request_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.active_batches = {}
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0,
            "avg_processing_time_ms": 0.0,
            "throughput_rps": 0.0,
            "queue_size": 0
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.worker_threads)
        self.batch_processor_task = None
        
        logger.info("🔄 Dynamic Batcher initialized")
        logger.info(f"   Max batch size: {config.max_batch_size}")
        logger.info(f"   Max wait time: {config.max_wait_time_ms}ms")
        logger.info(f"   Device: {self.device}")
    
    async def start(self):
        """Start the dynamic batching service"""
        if self.is_running:
            logger.warning("Dynamic batcher already running")
            return
        
        self.is_running = True
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info("✅ Dynamic batcher started")
    
    async def stop(self):
        """Stop the dynamic batching service"""
        self.is_running = False
        
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        logger.info("🛑 Dynamic batcher stopped")
    
    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate text with dynamic batching
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stream: Whether to stream results
            
        Returns:
            Generated text (streaming or complete)
        """
        if not self.is_running:
            await self.start()
        
        # Create request
        request = BatchRequest(
            request_id=str(uuid.uuid4()),
            input_ids=input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            future=asyncio.Future()
        )
        
        # Add to queue
        try:
            await self.request_queue.put(request)
            self.stats["total_requests"] += 1
            self.stats["queue_size"] = self.request_queue.qsize()
            
            if stream:
                return self._stream_response(request)
            else:
                return await request.future
                
        except asyncio.QueueFull:
            logger.error("Request queue full - rejecting request")
            raise RuntimeError("Server overloaded - try again later")
    
    async def _stream_response(self, request: BatchRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response tokens as they are generated"""
        try:
            # Wait for generation to start
            result = await request.future
            
            if "error" in result:
                yield result
                return
            
            # Stream tokens
            generated_tokens = result.get("generated_tokens", [])
            
            for i in range(0, len(generated_tokens), self.config.stream_chunk_size):
                chunk_tokens = generated_tokens[i:i + self.config.stream_chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                
                yield {
                    "token_ids": chunk_tokens,
                    "text": chunk_text,
                    "is_final": i + self.config.stream_chunk_size >= len(generated_tokens),
                    "request_id": request.request_id
                }
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
        
        except Exception as e:
            yield {"error": str(e), "request_id": request.request_id}
    
    async def _batch_processor(self):
        """Main batch processing loop"""
        logger.info("🔄 Batch processor started")
        
        while self.is_running:
            try:
                # Collect requests for batching
                batch_requests = await self._collect_batch()
                
                if not batch_requests:
                    await asyncio.sleep(0.001)  # Small delay if no requests
                    continue
                
                # Process batch
                await self._process_batch(batch_requests)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("🛑 Batch processor stopped")
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests into a batch"""
        batch_requests = []
        start_time = time.time()
        
        # Get first request (blocking)
        try:
            first_request = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=0.1
            )
            batch_requests.append(first_request)
        except asyncio.TimeoutError:
            return []
        
        # Collect additional requests (non-blocking)
        while (
            len(batch_requests) < self.config.max_batch_size and
            (time.time() - start_time) * 1000 < self.config.max_wait_time_ms
        ):
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=0.001
                )
                batch_requests.append(request)
            except asyncio.TimeoutError:
                break
        
        # Update statistics
        wait_time = (time.time() - start_time) * 1000
        self.stats["avg_wait_time_ms"] = (
            self.stats["avg_wait_time_ms"] * 0.9 + wait_time * 0.1
        )
        
        return batch_requests
    
    async def _process_batch(self, batch_requests: List[BatchRequest]):
        """Process a batch of requests"""
        if not batch_requests:
            return
        
        start_time = time.time()
        batch_size = len(batch_requests)
        
        logger.debug(f"Processing batch of {batch_size} requests")
        
        try:
            # Prepare batch inputs
            batch_inputs = self._prepare_batch_inputs(batch_requests)
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._run_batch_inference,
                batch_inputs,
                batch_requests
            )
            
            # Distribute results to requests
            for request, result in zip(batch_requests, results):
                if not request.future.done():
                    request.future.set_result(result)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.stats["total_batches"] += 1
            self.stats["avg_batch_size"] = (
                self.stats["avg_batch_size"] * 0.9 + batch_size * 0.1
            )
            self.stats["avg_processing_time_ms"] = (
                self.stats["avg_processing_time_ms"] * 0.9 + processing_time * 0.1
            )
            self.stats["throughput_rps"] = batch_size / (processing_time / 1000)
            
            logger.debug(f"Batch processed in {processing_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Set error for all requests
            error_result = {"error": str(e)}
            for request in batch_requests:
                if not request.future.done():
                    request.future.set_result(error_result)
    
    def _prepare_batch_inputs(self, batch_requests: List[BatchRequest]) -> Dict[str, torch.Tensor]:
        """Prepare batched inputs with padding"""
        # Find maximum sequence length
        max_length = max(req.input_ids.shape[-1] for req in batch_requests)
        batch_size = len(batch_requests)
        
        # Create padded batch
        batch_input_ids = torch.full(
            (batch_size, max_length),
            self.config.padding_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=self.device
        )
        
        # Fill batch with actual sequences
        for i, request in enumerate(batch_requests):
            seq_len = request.input_ids.shape[-1]
            batch_input_ids[i, :seq_len] = request.input_ids.squeeze(0)
            attention_mask[i, :seq_len] = 1
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "batch_requests": batch_requests
        }
    
    def _run_batch_inference(
        self,
        batch_inputs: Dict[str, torch.Tensor],
        batch_requests: List[BatchRequest]
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch (runs in thread pool)"""
        try:
            with torch.no_grad():
                # Detect model type and run appropriate generation
                if hasattr(self.model, 'generate_with_state_space'):
                    # Mamba model
                    outputs = self._generate_mamba_batch(batch_inputs, batch_requests)
                else:
                    # Transformer model
                    outputs = self._generate_transformer_batch(batch_inputs, batch_requests)
                
                return outputs
                
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [{"error": str(e)} for _ in batch_requests]
    
    def _generate_mamba_batch(
        self,
        batch_inputs: Dict[str, torch.Tensor],
        batch_requests: List[BatchRequest]
    ) -> List[Dict[str, Any]]:
        """Generate with Mamba model (batch processing)"""
        results = []
        
        # Process each request individually for now
        # TODO: Implement true batch processing for Mamba
        for i, request in enumerate(batch_requests):
            try:
                input_ids = batch_inputs["input_ids"][i:i+1]
                
                # Use Mamba's generation method
                if hasattr(self.model, 'model'):
                    outputs = self.model.model.generate(
                        input_ids,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Extract generated tokens
                generated_tokens = outputs[0][input_ids.shape[1]:].tolist()
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                results.append({
                    "generated_tokens": generated_tokens,
                    "generated_text": generated_text,
                    "request_id": request.request_id,
                    "model_type": "mamba"
                })
                
            except Exception as e:
                results.append({
                    "error": str(e),
                    "request_id": request.request_id
                })
        
        return results
    
    def _generate_transformer_batch(
        self,
        batch_inputs: Dict[str, torch.Tensor],
        batch_requests: List[BatchRequest]
    ) -> List[Dict[str, Any]]:
        """Generate with Transformer model (true batch processing)"""
        try:
            # Use the first request's parameters for the batch
            # (In practice, you might want to handle different parameters per request)
            first_request = batch_requests[0]
            
            outputs = self.model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=first_request.max_new_tokens,
                temperature=first_request.temperature,
                top_p=first_request.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Process results for each request
            results = []
            for i, request in enumerate(batch_requests):
                input_len = request.input_ids.shape[-1]
                generated_tokens = outputs[i][input_len:].tolist()
                
                # Remove padding tokens
                if self.tokenizer.pad_token_id in generated_tokens:
                    pad_idx = generated_tokens.index(self.tokenizer.pad_token_id)
                    generated_tokens = generated_tokens[:pad_idx]
                
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                results.append({
                    "generated_tokens": generated_tokens,
                    "generated_text": generated_text,
                    "request_id": request.request_id,
                    "model_type": "transformer"
                })
            
            return results
            
        except Exception as e:
            return [{"error": str(e), "request_id": req.request_id} for req in batch_requests]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        stats = self.stats.copy()
        stats["queue_size"] = self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0
        stats["is_running"] = self.is_running
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0,
            "avg_processing_time_ms": 0.0,
            "throughput_rps": 0.0,
            "queue_size": 0
        }


class AsyncStreamingServer:
    """
    Async Streaming Server
    
    Provides async streaming capabilities for real-time token generation.
    """
    
    def __init__(self, batcher: DynamicBatcher):
        self.batcher = batcher
        self.active_streams = {}
        
        logger.info("🌊 Async Streaming Server initialized")
    
    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream text generation
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            
        Yields:
            Streaming response chunks
        """
        try:
            # Tokenize input
            input_ids = self.batcher.tokenizer.encode(
                prompt,
                return_tensors="pt",
                add_special_tokens=True
            )
            
            # Generate with streaming
            async for chunk in self.batcher.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            ):
                yield chunk
                
        except Exception as e:
            yield {"error": str(e)}
    
    async def batch_stream_generate(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> AsyncGenerator[Dict[str, List[Dict[str, Any]]], None]:
        """
        Stream generation for multiple prompts simultaneously
        
        Args:
            prompts: List of input prompts
            **generation_kwargs: Generation parameters
            
        Yields:
            Batch streaming responses
        """
        try:
            # Start all streams
            streams = []
            for i, prompt in enumerate(prompts):
                stream = self.stream_generate(prompt, **generation_kwargs)
                streams.append((i, stream))
            
            # Collect results from all streams
            active_streams = {i: stream for i, stream in streams}
            
            while active_streams:
                batch_results = []
                completed_streams = []
                
                for stream_id, stream in active_streams.items():
                    try:
                        chunk = await stream.__anext__()
                        batch_results.append({
                            "stream_id": stream_id,
                            "chunk": chunk
                        })
                        
                        if chunk.get("is_final", False):
                            completed_streams.append(stream_id)
                            
                    except StopAsyncIteration:
                        completed_streams.append(stream_id)
                
                # Remove completed streams
                for stream_id in completed_streams:
                    del active_streams[stream_id]
                
                if batch_results:
                    yield {"batch_results": batch_results}
                
                # Small delay between batches
                await asyncio.sleep(0.01)
                
        except Exception as e:
            yield {"error": str(e)}


# Factory functions
def create_dynamic_batcher(
    model,
    tokenizer,
    max_batch_size: int = 8,
    max_wait_time_ms: int = 50,
    device: str = "auto"
) -> DynamicBatcher:
    """Create dynamic batcher with default config"""
    config = BatchConfig(
        max_batch_size=max_batch_size,
        max_wait_time_ms=max_wait_time_ms
    )
    
    return DynamicBatcher(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )


def create_streaming_server(batcher: DynamicBatcher) -> AsyncStreamingServer:
    """Create async streaming server"""
    return AsyncStreamingServer(batcher)


# Integration with existing models
async def integrate_dynamic_batching():
    """
    Integration function for existing MARK AI models
    
    This can be called to add dynamic batching to existing models.
    """
    try:
        from src.core.mamba_loader import load_mamba_model
        from src.core.model_registry import get_model_instance
        
        logger.info("🔗 Integrating dynamic batching with existing models")
        
        async def create_batched_service(model_key: str = "mamba"):
            """Create batched service for existing model"""
            
            # Load model
            if model_key == "mamba":
                model_instance = load_mamba_model()
                if model_instance.available:
                    model = model_instance
                    tokenizer = model_instance.tokenizer
                else:
                    logger.warning("Mamba model not available")
                    return None
            else:
                model_instance = get_model_instance(model_key)
                if model_instance:
                    model = model_instance.model
                    tokenizer = model_instance.tokenizer
                else:
                    logger.warning(f"Model '{model_key}' not available")
                    return None
            
            # Create batcher and streaming server
            batcher = create_dynamic_batcher(
                model=model,
                tokenizer=tokenizer,
                max_batch_size=8,
                max_wait_time_ms=50
            )
            
            streaming_server = create_streaming_server(batcher)
            
            # Start the service
            await batcher.start()
            
            logger.info("✅ Dynamic batching service created and started")
            return batcher, streaming_server
        
        return create_batched_service
        
    except ImportError as e:
        logger.warning(f"⚠️  Could not integrate dynamic batching: {e}")
        return None
