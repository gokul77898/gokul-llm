"""
Low-Latency Token Streaming (SSE + WebSocket)

Implements real-time token streaming via Server-Sent Events (SSE) and WebSocket
for ultra-low latency text generation responses.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, List, Callable
from dataclasses import dataclass, field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
import websockets
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for token streaming"""
    max_connections: int = 100
    heartbeat_interval: int = 30  # seconds
    buffer_size: int = 1024
    compression_enabled: bool = True
    rate_limit_per_second: int = 1000  # tokens per second per connection
    enable_sse: bool = True
    enable_websocket: bool = True


@dataclass
class StreamSession:
    """Individual streaming session"""
    session_id: str
    connection_type: str  # "sse" or "websocket"
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    tokens_sent: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenStreamer:
    """
    Low-Latency Token Streaming Engine
    
    Provides real-time token streaming with minimal latency via SSE and WebSocket.
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.active_sessions: Dict[str, StreamSession] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.sse_generators: Dict[str, AsyncGenerator] = {}
        
        # Statistics
        self.stats = {
            "total_sessions": 0,
            "active_connections": 0,
            "total_tokens_streamed": 0,
            "avg_latency_ms": 0.0,
            "peak_connections": 0,
            "uptime": time.time()
        }
        
        logger.info("🌊 Token Streamer initialized")
        logger.info(f"   Max connections: {config.max_connections}")
        logger.info(f"   SSE enabled: {config.enable_sse}")
        logger.info(f"   WebSocket enabled: {config.enable_websocket}")
    
    async def create_sse_stream(
        self,
        request: Request,
        generator_func: Callable,
        **generation_kwargs
    ) -> StreamingResponse:
        """
        Create Server-Sent Events stream
        
        Args:
            request: FastAPI request object
            generator_func: Function that generates tokens
            **generation_kwargs: Arguments for generation
            
        Returns:
            StreamingResponse for SSE
        """
        session_id = str(uuid.uuid4())
        session = StreamSession(
            session_id=session_id,
            connection_type="sse"
        )
        
        self.active_sessions[session_id] = session
        self.stats["total_sessions"] += 1
        self.stats["active_connections"] += 1
        self.stats["peak_connections"] = max(
            self.stats["peak_connections"],
            self.stats["active_connections"]
        )
        
        logger.info(f"📡 SSE stream created: {session_id}")
        
        async def event_stream():
            try:
                # Send initial connection event
                yield self._format_sse_event("connected", {
                    "session_id": session_id,
                    "timestamp": time.time()
                })
                
                # Generate and stream tokens
                async for token_data in self._generate_tokens(
                    session, generator_func, **generation_kwargs
                ):
                    if not session.is_active:
                        break
                    
                    yield self._format_sse_event("token", token_data)
                    
                    # Rate limiting
                    await self._apply_rate_limit(session)
                
                # Send completion event
                yield self._format_sse_event("completed", {
                    "session_id": session_id,
                    "total_tokens": session.tokens_sent,
                    "duration": time.time() - session.created_at
                })
                
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                yield self._format_sse_event("error", {
                    "error": str(e),
                    "session_id": session_id
                })
            finally:
                await self._cleanup_session(session_id)
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def handle_websocket(
        self,
        websocket: WebSocket,
        generator_func: Callable
    ):
        """
        Handle WebSocket connection for token streaming
        
        Args:
            websocket: FastAPI WebSocket connection
            generator_func: Function that generates tokens
        """
        session_id = str(uuid.uuid4())
        session = StreamSession(
            session_id=session_id,
            connection_type="websocket"
        )
        
        await websocket.accept()
        
        self.active_sessions[session_id] = session
        self.websocket_connections[session_id] = websocket
        self.stats["total_sessions"] += 1
        self.stats["active_connections"] += 1
        self.stats["peak_connections"] = max(
            self.stats["peak_connections"],
            self.stats["active_connections"]
        )
        
        logger.info(f"🔌 WebSocket connected: {session_id}")
        
        try:
            # Send connection confirmation
            await websocket.send_json({
                "type": "connected",
                "session_id": session_id,
                "timestamp": time.time()
            })
            
            # Listen for messages and handle generation
            while session.is_active:
                try:
                    # Wait for generation request
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=1.0
                    )
                    
                    if message.get("type") == "generate":
                        await self._handle_websocket_generation(
                            session, websocket, generator_func, message
                        )
                    elif message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message.get("type") == "disconnect":
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": time.time()
                    })
                except WebSocketDisconnect:
                    break
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "session_id": session_id
                })
            except:
                pass
        finally:
            await self._cleanup_session(session_id)
    
    async def _generate_tokens(
        self,
        session: StreamSession,
        generator_func: Callable,
        **generation_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate tokens and yield them for streaming"""
        try:
            # Call the generator function (could be Mamba or Transformer)
            async for token_data in generator_func(**generation_kwargs):
                if not session.is_active:
                    break
                
                # Add streaming metadata
                enhanced_data = {
                    **token_data,
                    "session_id": session.session_id,
                    "timestamp": time.time(),
                    "sequence_number": session.tokens_sent
                }
                
                session.tokens_sent += 1
                session.last_activity = time.time()
                self.stats["total_tokens_streamed"] += 1
                
                yield enhanced_data
                
        except Exception as e:
            logger.error(f"Token generation error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "session_id": session.session_id
            }
    
    async def _handle_websocket_generation(
        self,
        session: StreamSession,
        websocket: WebSocket,
        generator_func: Callable,
        message: Dict[str, Any]
    ):
        """Handle WebSocket generation request"""
        try:
            # Extract generation parameters
            generation_kwargs = message.get("parameters", {})
            
            # Send generation start event
            await websocket.send_json({
                "type": "generation_started",
                "session_id": session.session_id,
                "timestamp": time.time()
            })
            
            # Stream tokens
            async for token_data in self._generate_tokens(
                session, generator_func, **generation_kwargs
            ):
                await websocket.send_json({
                    "type": "token",
                    **token_data
                })
                
                # Rate limiting
                await self._apply_rate_limit(session)
            
            # Send completion event
            await websocket.send_json({
                "type": "generation_completed",
                "session_id": session.session_id,
                "total_tokens": session.tokens_sent,
                "duration": time.time() - session.created_at
            })
            
        except Exception as e:
            logger.error(f"WebSocket generation error: {e}")
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "session_id": session.session_id
            })
    
    def _format_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event"""
        event_data = json.dumps(data)
        return f"event: {event_type}\ndata: {event_data}\n\n"
    
    async def _apply_rate_limit(self, session: StreamSession):
        """Apply rate limiting to prevent overwhelming clients"""
        if session.tokens_sent > 0:
            elapsed = time.time() - session.created_at
            rate = session.tokens_sent / elapsed
            
            if rate > self.config.rate_limit_per_second:
                # Calculate delay needed
                delay = (session.tokens_sent / self.config.rate_limit_per_second) - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            
            duration = time.time() - session.created_at
            logger.info(f"🧹 Session cleanup: {session_id}")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   Tokens sent: {session.tokens_sent}")
            
            del self.active_sessions[session_id]
            self.stats["active_connections"] -= 1
        
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]
        
        if session_id in self.sse_generators:
            del self.sse_generators[session_id]
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        disconnected_sessions = []
        
        for session_id, websocket in self.websocket_connections.items():
            try:
                await websocket.send_json({
                    "type": "broadcast",
                    **message
                })
            except Exception as e:
                logger.warning(f"Failed to send broadcast to {session_id}: {e}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            await self._cleanup_session(session_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        current_time = time.time()
        uptime = current_time - self.stats["uptime"]
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "active_sessions": len(self.active_sessions),
            "websocket_connections": len(self.websocket_connections),
            "sse_connections": len([s for s in self.active_sessions.values() if s.connection_type == "sse"])
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for streaming service"""
        return {
            "status": "healthy",
            "active_connections": len(self.active_sessions),
            "max_connections": self.config.max_connections,
            "uptime": time.time() - self.stats["uptime"],
            "total_tokens_streamed": self.stats["total_tokens_streamed"]
        }


class StreamingIntegration:
    """
    Integration layer for existing MARK AI models
    
    Connects token streaming with Mamba/Transformer models.
    """
    
    def __init__(self, streamer: TokenStreamer):
        self.streamer = streamer
        self.model_cache = {}
        
        logger.info("🔗 Streaming Integration initialized")
    
    async def create_model_generator(self, model_key: str = "mamba"):
        """Create generator function for specific model"""
        try:
            from src.core.mamba_loader import load_mamba_model
            from src.core.model_registry import get_model_instance
            
            # Load model if not cached
            if model_key not in self.model_cache:
                if model_key == "mamba":
                    model_instance = load_mamba_model()
                    if model_instance.available:
                        self.model_cache[model_key] = model_instance
                    else:
                        logger.warning("Mamba model not available")
                        return None
                else:
                    model_instance = get_model_instance(model_key)
                    if model_instance:
                        self.model_cache[model_key] = model_instance
                    else:
                        logger.warning(f"Model '{model_key}' not available")
                        return None
            
            model_instance = self.model_cache[model_key]
            
            async def model_generator(
                prompt: str,
                max_new_tokens: int = 100,
                temperature: float = 0.7,
                top_p: float = 0.9,
                **kwargs
            ):
                """Generator function that yields tokens from the model"""
                try:
                    # Tokenize input
                    if hasattr(model_instance, 'tokenizer'):
                        tokenizer = model_instance.tokenizer
                    else:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")
                    
                    # Generate tokens one by one for streaming
                    current_ids = input_ids
                    
                    for step in range(max_new_tokens):
                        # Generate next token
                        if hasattr(model_instance, 'generate_with_state_space'):
                            # Mamba model
                            outputs = model_instance.model.generate(
                                current_ids,
                                max_new_tokens=1,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                        else:
                            # Transformer model
                            outputs = model_instance.model.generate(
                                current_ids,
                                max_new_tokens=1,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                        
                        # Extract new token
                        if outputs.shape[1] > current_ids.shape[1]:
                            new_token = outputs[0, -1].item()
                            new_text = tokenizer.decode([new_token], skip_special_tokens=True)
                            
                            # Yield token data
                            yield {
                                "token_id": new_token,
                                "token_text": new_text,
                                "step": step,
                                "is_final": step == max_new_tokens - 1 or new_token == tokenizer.eos_token_id,
                                "model_type": getattr(model_instance, 'backend', 'unknown')
                            }
                            
                            # Update current sequence
                            current_ids = outputs
                            
                            # Check for EOS token
                            if new_token == tokenizer.eos_token_id:
                                break
                        else:
                            break
                        
                        # Small delay to prevent overwhelming
                        await asyncio.sleep(0.001)
                
                except Exception as e:
                    logger.error(f"Model generation error: {e}")
                    yield {
                        "error": str(e),
                        "step": -1,
                        "is_final": True
                    }
            
            return model_generator
            
        except Exception as e:
            logger.error(f"Failed to create model generator: {e}")
            return None


# Factory functions
def create_token_streamer(
    max_connections: int = 100,
    enable_sse: bool = True,
    enable_websocket: bool = True
) -> TokenStreamer:
    """Create token streamer with configuration"""
    config = StreamConfig(
        max_connections=max_connections,
        enable_sse=enable_sse,
        enable_websocket=enable_websocket
    )
    return TokenStreamer(config)


def create_streaming_integration(streamer: TokenStreamer) -> StreamingIntegration:
    """Create streaming integration"""
    return StreamingIntegration(streamer)


# FastAPI integration helpers
def add_streaming_routes(app: FastAPI, streamer: TokenStreamer, integration: StreamingIntegration):
    """Add streaming routes to FastAPI app"""
    
    @app.get("/stream/sse/{model_key}")
    async def sse_stream(request: Request, model_key: str = "mamba"):
        """SSE streaming endpoint"""
        generator_func = await integration.create_model_generator(model_key)
        if generator_func is None:
            return {"error": f"Model '{model_key}' not available"}
        
        return await streamer.create_sse_stream(request, generator_func)
    
    @app.websocket("/stream/ws/{model_key}")
    async def websocket_stream(websocket: WebSocket, model_key: str = "mamba"):
        """WebSocket streaming endpoint"""
        generator_func = await integration.create_model_generator(model_key)
        if generator_func is None:
            await websocket.close(code=1000, reason=f"Model '{model_key}' not available")
            return
        
        await streamer.handle_websocket(websocket, generator_func)
    
    @app.get("/stream/stats")
    async def streaming_stats():
        """Get streaming statistics"""
        return streamer.get_stats()
    
    @app.get("/stream/health")
    async def streaming_health():
        """Streaming service health check"""
        return await streamer.health_check()
    
    logger.info("✅ Streaming routes added to FastAPI app")
    logger.info("   SSE: GET /stream/sse/{model_key}")
    logger.info("   WebSocket: WS /stream/ws/{model_key}")
    logger.info("   Stats: GET /stream/stats")
    logger.info("   Health: GET /stream/health")
