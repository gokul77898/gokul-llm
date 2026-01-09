"""
Local Model Registry

GPU-Ready Local Model Execution Layer

Provides:
- Lazy model loading (no auto-download)
- GPU-ready execution when models are loaded
- Graceful failure when models not loaded
- GPU hygiene and resource guards (Phase P1)
- Execution isolation and fault containment (Phase P2)
- Runtime health, readiness & warmup (Phase P4)

CONSTRAINTS:
- No auto-loading
- No downloads triggered at import
- GPU only when models explicitly loaded
- CPU compatible for RAG/validation
- Fail-fast on insufficient GPU resources

PHASE P1 SAFETY:
- CUDA availability check before any model load
- VRAM guards with minimum requirements
- Explicit torch backend policy (bfloat16, TF32)
- Hard generation limits (no sampling, max tokens)
- No raw CUDA traces exposed to API

PHASE P2 SAFETY:
- Encoder/decoder execution isolation
- Timeout enforcement (encoder: 10s, decoder: 20s)
- GPU state cleanup on failure only
- No partial outputs, no server crashes
- No retry logic

PHASE P4 HEALTH:
- Model state machine (UNINITIALIZED → LOADING → READY/DEGRADED/FAILED)
- Real model warmup (single forward pass)
- Warmup telemetry (encoder_warmup_ms, decoder_warmup_ms)
- GPU health endpoint support
"""

import logging
import signal
import threading
import time
from enum import Enum
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PHASE P4: MODEL STATE MACHINE
# ─────────────────────────────────────────────

class ModelState(str, Enum):
    """
    Model registry state machine.
    
    State transitions:
    - UNINITIALIZED → LOADING (on load_encoder/load_decoder)
    - LOADING → READY (warmup success, both models loaded)
    - LOADING → DEGRADED_ENCODER_ONLY (encoder loaded, decoder failed)
    - LOADING → DEGRADED_RAG_ONLY (RAG only, no models)
    - LOADING/READY/DEGRADED_* → FAILED (any fatal error)
    
    Phase P8: Extended degraded modes for graceful degradation.
    """
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"  # Generic degraded (backward compat)
    DEGRADED_ENCODER_ONLY = "degraded_encoder_only"  # P8: Encoder works, decoder failed
    DEGRADED_RAG_ONLY = "degraded_rag_only"  # P8: RAG only, no GPU models
    FAILED = "failed"


@dataclass
class WarmupTelemetry:
    """Telemetry data from model warmup."""
    encoder_warmup_ms: Optional[int] = None
    decoder_warmup_ms: Optional[int] = None
    first_token_latency_ms: Optional[int] = None
    encoder_warmed_up: bool = False
    decoder_warmed_up: bool = False
    warmup_timestamp: Optional[str] = None


# ─────────────────────────────────────────────
# PHASE P2: EXECUTION TIMEOUT LIMITS
# ─────────────────────────────────────────────

ENCODER_TIMEOUT_SECONDS: int = 10
DECODER_TIMEOUT_SECONDS: int = 20


# ─────────────────────────────────────────────
# PHASE P2: CUSTOM EXCEPTION TYPES
# ─────────────────────────────────────────────

class EncoderExecutionError(Exception):
    """
    Raised when encoder execution fails.
    
    Server should return:
        status="refused"
        reason="encoder_failed"
    """
    def __init__(self, reason: str, details: str = ""):
        self.reason = reason
        self.details = details
        super().__init__(f"Encoder execution failed: {reason}")


class DecoderExecutionError(Exception):
    """
    Raised when decoder execution fails.
    
    Server should return:
        status="refused"
        reason="decoder_failed"
    """
    def __init__(self, reason: str, details: str = ""):
        self.reason = reason
        self.details = details
        super().__init__(f"Decoder execution failed: {reason}")


class ExecutionTimeoutError(Exception):
    """Raised when execution exceeds timeout limit."""
    pass


# ─────────────────────────────────────────────
# PHASE P2: GPU STATE CLEANUP
# ─────────────────────────────────────────────

def _cleanup_gpu_on_failure() -> None:
    """
    Clean GPU state after a failure.
    
    ONLY called on failure - not on success (to avoid perf regression).
    Does NOT unload models or mutate global state.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("[GPU] Cache cleared after failure")
    except Exception as e:
        # Don't fail on cleanup failure
        logger.warning(f"GPU cleanup failed: {e}")


def _synchronize_gpu() -> None:
    """
    Synchronize GPU after execution.
    
    Called after both success and failure to ensure
    all GPU operations are complete.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass  # Ignore sync failures


# ─────────────────────────────────────────────
# PHASE P2: TIMEOUT ENFORCEMENT
# ─────────────────────────────────────────────

class TimeoutHandler:
    """
    Cross-platform timeout handler.
    
    Uses signal.alarm on Unix, threading.Timer as fallback.
    """
    
    def __init__(self, timeout_seconds: int, operation: str):
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self._timer: Optional[threading.Timer] = None
        self._timed_out = False
        self._use_signal = hasattr(signal, 'SIGALRM')
    
    def _timeout_handler(self, signum=None, frame=None):
        """Signal handler for timeout."""
        self._timed_out = True
        raise ExecutionTimeoutError(
            f"{self.operation} timed out after {self.timeout_seconds}s"
        )
    
    def _timer_callback(self):
        """Timer callback for non-signal platforms."""
        self._timed_out = True
        # Note: This won't interrupt the main thread, but we check the flag
    
    def __enter__(self):
        self._timed_out = False
        if self._use_signal:
            # Unix: use signal.alarm
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout_seconds)
        else:
            # Fallback: use threading.Timer (won't interrupt, but sets flag)
            self._timer = threading.Timer(
                self.timeout_seconds,
                self._timer_callback
            )
            self._timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._use_signal:
            signal.alarm(0)  # Cancel alarm
        elif self._timer:
            self._timer.cancel()
        return False  # Don't suppress exceptions
    
    def check_timeout(self):
        """Check if timeout occurred (for non-signal platforms)."""
        if self._timed_out:
            raise ExecutionTimeoutError(
                f"{self.operation} timed out after {self.timeout_seconds}s"
            )


# ─────────────────────────────────────────────
# PHASE P1: HARD GENERATION SAFETY LIMITS
# ─────────────────────────────────────────────
# These limits are IMMUTABLE and enforced server-side
# regardless of prompt content or API parameters.

MAX_NEW_TOKENS: int = 512
MAX_INPUT_TOKENS: int = 4096

# VRAM requirements (in GB)
ENCODER_MIN_VRAM_GB: float = 16.0  # Generic encoder
DECODER_MIN_VRAM_GB: float = 48.0  # Qwen 32B (prefer ≥80GB)
DECODER_PREFERRED_VRAM_GB: float = 80.0


# ─────────────────────────────────────────────
# PHASE P1: GPU RESOURCE GUARDS
# ─────────────────────────────────────────────

def check_cuda_available() -> None:
    """
    Check if CUDA is available.
    
    Raises:
        RuntimeError: If CUDA is not available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. GPU is required for model loading. "
                "Ensure NVIDIA drivers and CUDA toolkit are installed."
            )
    except ImportError:
        raise RuntimeError("PyTorch not installed. Cannot check CUDA availability.")


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information including device count and VRAM.
    
    Returns:
        Dictionary with GPU info per device
        
    Raises:
        RuntimeError: If CUDA is not available
    """
    check_cuda_available()
    
    import torch
    
    device_count = torch.cuda.device_count()
    devices = {}
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_vram_bytes = props.total_memory
        total_vram_gb = total_vram_bytes / (1024 ** 3)
        
        devices[i] = {
            "name": props.name,
            "total_vram_gb": round(total_vram_gb, 2),
            "total_vram_bytes": total_vram_bytes,
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
        }
    
    return {
        "device_count": device_count,
        "devices": devices,
        "cuda_version": torch.version.cuda,
    }


def check_vram_sufficient(required_gb: float, role: str, model_id: str) -> None:
    """
    Check if sufficient VRAM is available for model loading.
    
    Args:
        required_gb: Minimum required VRAM in GB
        role: Model role ("encoder" or "decoder")
        model_id: Model identifier for error message
        
    Raises:
        RuntimeError: If insufficient VRAM
    """
    check_cuda_available()
    
    import torch
    
    # Get total VRAM across all devices (for device_map="auto")
    total_vram_gb = 0.0
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_vram_gb += props.total_memory / (1024 ** 3)
    
    if total_vram_gb < required_gb:
        raise RuntimeError(
            f"Insufficient GPU memory for {model_id} ({role}): "
            f"required ≥{required_gb:.0f}GB, found {total_vram_gb:.1f}GB. "
            f"DO NOT attempt partial load."
        )
    
    # Log warning if below preferred for decoder
    if role == "decoder" and total_vram_gb < DECODER_PREFERRED_VRAM_GB:
        logger.warning(
            f"GPU memory ({total_vram_gb:.1f}GB) is below preferred "
            f"{DECODER_PREFERRED_VRAM_GB:.0f}GB for {model_id}. "
            f"Performance may be degraded."
        )


def configure_torch_backend() -> None:
    """
    Configure torch backend for optimal GPU performance.
    
    Sets:
    - TF32 for matmul and cuDNN
    - No implicit dtype inference
    """
    try:
        import torch
        
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Ensure deterministic behavior where possible
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
        
        logger.info("[GPU] Torch backend configured: TF32 enabled")
        
    except Exception as e:
        # Don't fail if torch not available - this is called during load
        logger.warning(f"Could not configure torch backend: {e}")


@dataclass
class ModelConfig:
    """Configuration for a loaded model."""
    model_id: str
    role: str  # "encoder" or "decoder"
    device: str
    dtype: str


class LocalModelRegistry:
    """
    Registry for locally loaded models.
    
    Models are NOT loaded automatically.
    Call load_encoder() or load_decoder() explicitly when GPU is available.
    
    PHASE P4: State machine and warmup support.
    """
    
    def __init__(self):
        self._models: Dict[str, Tuple[Any, Any]] = {}  # model_id -> (tokenizer, model)
        self._configs: Dict[str, ModelConfig] = {}
        
        # Phase P4: State machine
        self._state: ModelState = ModelState.UNINITIALIZED
        self._encoder_id: Optional[str] = None
        self._decoder_id: Optional[str] = None
        
        # Phase P4: Warmup telemetry
        self._warmup: WarmupTelemetry = WarmupTelemetry()
        
        # Phase P8: Failure tracking
        self._failure_reason: Optional[str] = None
        
        logger.info("[INFO] Local model registry initialized (no models loaded)")
    
    # ─────────────────────────────────────────────
    # PHASE P4: STATE MACHINE
    # ─────────────────────────────────────────────
    
    def get_state(self) -> ModelState:
        """Get current model state."""
        return self._state
    
    def get_warmup_telemetry(self) -> WarmupTelemetry:
        """Get warmup telemetry data."""
        return self._warmup
    
    def _update_state(self) -> None:
        """
        Update state based on loaded models.
        
        State rules (Phase P8):
        - Both encoder + decoder loaded & warmed → READY
        - Encoder-only loaded & warmed → DEGRADED_ENCODER_ONLY
        - No models but RAG available → DEGRADED_RAG_ONLY
        - Any fatal error → FAILED (set externally)
        """
        if self._state == ModelState.FAILED:
            return  # Don't recover from FAILED
        
        encoder_loaded = self._encoder_id is not None and self._encoder_id in self._models
        decoder_loaded = self._decoder_id is not None and self._decoder_id in self._models
        
        if encoder_loaded and decoder_loaded:
            if self._warmup.encoder_warmed_up and self._warmup.decoder_warmed_up:
                self._state = ModelState.READY
            else:
                self._state = ModelState.LOADING
        elif encoder_loaded:
            # Phase P8: Encoder-only mode
            if self._warmup.encoder_warmed_up:
                self._state = ModelState.DEGRADED_ENCODER_ONLY
            else:
                self._state = ModelState.LOADING
        elif decoder_loaded:
            # Decoder-only is unusual but handle it
            self._state = ModelState.DEGRADED
        else:
            # No GPU models - RAG-only mode
            self._state = ModelState.DEGRADED_RAG_ONLY
    
    def _set_failed(self, reason: str) -> None:
        """Set state to FAILED with reason."""
        self._state = ModelState.FAILED
        self._failure_reason = reason
        logger.error(f"[MODEL STATE] FAILED: {reason}")
    
    def _set_degraded_encoder_only(self, reason: str) -> None:
        """
        Phase P8: Set state to DEGRADED_ENCODER_ONLY.
        
        Called when decoder fails (e.g., OOM) but encoder is working.
        """
        self._state = ModelState.DEGRADED_ENCODER_ONLY
        logger.warning(f"[MODEL STATE] DEGRADED_ENCODER_ONLY: {reason}")
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        return model_id in self._models
    
    def load_encoder(self, model_id: str, device: str = "auto", dtype: str = "bfloat16") -> None:
        """
        DEPRECATED: Use remote embedding service instead.
        
        This method is kept for backward compatibility but should not be used.
        All encoder operations should use the remote HF Space service.
        """
        logger.warning("load_encoder is deprecated. Use remote embedding service instead.")
        raise RuntimeError("Local encoder loading is deprecated. Use remote embedding service.")
    
    def load_decoder(self, model_id: str, device: str = "auto", dtype: str = "bfloat16") -> None:
        """
        Load a decoder model for text generation.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to load on ("auto", "cuda") - CPU NOT allowed for large models
            dtype: Data type - MUST be "bfloat16" (enforced)
            
        Raises:
            RuntimeError: If CUDA not available, insufficient VRAM, or loading fails
            
        PHASE P4: Updates state machine and runs warmup after load.
        """
        # Phase P4: Set state to LOADING
        self._state = ModelState.LOADING
        
        try:
            # ─────────────────────────────────────────────
            # PHASE P1: GPU GUARDS (FAIL-FAST)
            # ─────────────────────────────────────────────
            
            # 1. Check CUDA availability
            check_cuda_available()
            
            # 2. Check VRAM requirements (decoder needs more VRAM)
            check_vram_sufficient(DECODER_MIN_VRAM_GB, "decoder", model_id)
            
            # 3. Configure torch backend (TF32, etc.)
            configure_torch_backend()
            
            # 4. Enforce bfloat16 - no fp32 fallback allowed
            if dtype != "bfloat16":
                logger.warning(f"Overriding dtype={dtype} to bfloat16 (enforced policy)")
                dtype = "bfloat16"
            
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Determine torch dtype (always bfloat16)
            torch_dtype = torch.bfloat16
            
            logger.info(f"Loading decoder: {model_id} (device={device}, dtype={dtype})")
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",  # Always use auto for GPU distribution
            )
            
            model.eval()
            
            self._models[model_id] = (tokenizer, model)
            self._configs[model_id] = ModelConfig(
                model_id=model_id,
                role="decoder",
                device=device,
                dtype=dtype,
            )
            self._decoder_id = model_id
            
            logger.info(f"Decoder loaded: {model_id}")
            
            # ─────────────────────────────────────────────
            # PHASE P4: DECODER WARMUP
            # ─────────────────────────────────────────────
            self._warmup_decoder(model_id)
            
            # Update state after successful load + warmup
            self._update_state()
            
        except RuntimeError as e:
            # Phase P4: Set state to FAILED on error
            self._set_failed(f"Decoder load failed: {e}")
            raise
        except Exception as e:
            # Wrap other exceptions - DO NOT expose raw CUDA traces
            error_msg = str(e)
            # Sanitize error message - remove CUDA stack traces
            if "CUDA" in error_msg or "cuda" in error_msg:
                error_msg = "GPU error during decoder load. Check VRAM and CUDA installation."
            logger.error(f"Failed to load decoder {model_id}: {error_msg}")
            self._set_failed(f"Decoder load failed: {error_msg}")
            raise RuntimeError(f"Failed to load decoder: {error_msg}")
    
    def get(self, model_id: str) -> Tuple[Any, Any]:
        """
        Get a loaded model.
        
        Args:
            model_id: Model ID to retrieve
            
        Returns:
            Tuple of (tokenizer, model)
            
        Raises:
            RuntimeError: If model not loaded
        """
        if model_id not in self._models:
            raise RuntimeError(
                f"Model not loaded: {model_id}. "
                "Load manually when GPU is available."
            )
        return self._models[model_id]
    
    def get_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a loaded model."""
        return self._configs.get(model_id)
    
    def list_loaded(self) -> Dict[str, str]:
        """List all loaded models with their roles."""
        return {
            model_id: config.role
            for model_id, config in self._configs.items()
        }
    
    def unload(self, model_id: str) -> bool:
        """
        Unload a model to free memory.
        
        Args:
            model_id: Model ID to unload
            
        Returns:
            True if unloaded, False if not found
        """
        if model_id not in self._models:
            return False
        
        del self._models[model_id]
        del self._configs[model_id]
        
        # Force garbage collection
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        logger.info(f"Model unloaded: {model_id}")
        return True
    
    def _get_torch_dtype(self, dtype: str):
        """Convert string dtype to torch dtype."""
        import torch
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.bfloat16)
    
    # ─────────────────────────────────────────────
    # PHASE P4: WARMUP METHODS
    # ─────────────────────────────────────────────
    
    def _warmup_encoder(self, model_id: str) -> None:
        """
        Run encoder warmup with a real forward pass.
        
        Warmup rules:
        - Single real forward pass
        - Short legal text (≤20 tokens)
        - Runs ONCE per model
        - Failure moves state → FAILED
        """
        if self._warmup.encoder_warmed_up:
            return  # Already warmed up
        
        logger.info(f"[WARMUP] Starting encoder warmup: {model_id}")
        start_time = time.time()
        
        try:
            import torch
            
            tokenizer, model = self._models[model_id]
            
            # Short legal text for warmup (≤20 tokens)
            warmup_text = "Section 420 IPC punishment for cheating."
            
            inputs = tokenizer(
                warmup_text,
                return_tensors="pt",
                truncation=True,
                max_length=32,
            )
            
            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Single forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Sync GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            warmup_ms = int((time.time() - start_time) * 1000)
            self._warmup.encoder_warmup_ms = warmup_ms
            self._warmup.encoder_warmed_up = True
            self._warmup.warmup_timestamp = datetime.utcnow().isoformat()
            
            logger.info(f"[WARMUP] Encoder warmup complete: {warmup_ms}ms")
            
        except Exception as e:
            logger.error(f"[WARMUP] Encoder warmup failed: {e}")
            self._set_failed(f"Encoder warmup failed: {e}")
            raise RuntimeError(f"Encoder warmup failed: {e}")
    
    def _warmup_decoder(self, model_id: str) -> None:
        """
        Run decoder warmup with a real generate call.
        
        Warmup rules:
        - Single real generate call
        - max_new_tokens ≤ 16
        - do_sample = False
        - Runs ONCE per model
        - Failure moves state → FAILED
        - Records first_token_latency_ms
        """
        if self._warmup.decoder_warmed_up:
            return  # Already warmed up
        
        logger.info(f"[WARMUP] Starting decoder warmup: {model_id}")
        start_time = time.time()
        
        try:
            import torch
            
            tokenizer, model = self._models[model_id]
            
            # Short prompt for warmup
            warmup_prompt = "Legal question: What is Section 420?"
            
            inputs = tokenizer(
                warmup_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=32,
            )
            
            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Record time before generation for first token latency
            gen_start = time.time()
            
            # Single generate call with minimal tokens
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=16,  # Minimal tokens for warmup
                    do_sample=False,  # Deterministic
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Sync GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            gen_end = time.time()
            
            warmup_ms = int((time.time() - start_time) * 1000)
            first_token_ms = int((gen_end - gen_start) * 1000)
            
            self._warmup.decoder_warmup_ms = warmup_ms
            self._warmup.first_token_latency_ms = first_token_ms
            self._warmup.decoder_warmed_up = True
            self._warmup.warmup_timestamp = datetime.utcnow().isoformat()
            
            logger.info(f"[WARMUP] Decoder warmup complete: {warmup_ms}ms (first_token: {first_token_ms}ms)")
            
        except Exception as e:
            logger.error(f"[WARMUP] Decoder warmup failed: {e}")
            self._set_failed(f"Decoder warmup failed: {e}")
            raise RuntimeError(f"Decoder warmup failed: {e}")


# ─────────────────────────────────────────────
# PHASE P5: GPU MEMORY SNAPSHOTS
# ─────────────────────────────────────────────

@dataclass
class GPUMemorySnapshot:
    """GPU memory snapshot for telemetry."""
    free_vram_gb: Optional[float] = None
    total_vram_gb: Optional[float] = None
    reserved_vram_gb: Optional[float] = None
    allocated_vram_gb: Optional[float] = None
    max_reserved_vram_gb: Optional[float] = None
    peak_allocated_vram_gb: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "free_vram_gb": self.free_vram_gb,
            "total_vram_gb": self.total_vram_gb,
            "reserved_vram_gb": self.reserved_vram_gb,
            "allocated_vram_gb": self.allocated_vram_gb,
            "max_reserved_vram_gb": self.max_reserved_vram_gb,
            "peak_allocated_vram_gb": self.peak_allocated_vram_gb,
            "timestamp": self.timestamp,
        }


def get_gpu_memory_snapshot() -> GPUMemorySnapshot:
    """
    Get GPU memory snapshot for telemetry.
    
    Returns snapshot with:
    - free_vram_gb
    - total_vram_gb
    - reserved_vram_gb
    - allocated_vram_gb
    - max_reserved_vram_gb
    - peak_allocated_vram_gb
    
    This function:
    - Works only if CUDA available
    - Never allocates memory
    - Never throws uncaught exceptions
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return GPUMemorySnapshot(timestamp=datetime.utcnow().isoformat())
        
        # Aggregate across all devices
        total_vram_gb = 0.0
        free_vram_gb = 0.0
        reserved_vram_gb = 0.0
        allocated_vram_gb = 0.0
        max_reserved_vram_gb = 0.0
        peak_allocated_vram_gb = 0.0
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb += props.total_memory / (1024 ** 3)
            
            # Get free memory
            free_mem, _ = torch.cuda.mem_get_info(i)
            free_vram_gb += free_mem / (1024 ** 3)
            
            # Get memory stats
            reserved_vram_gb += torch.cuda.memory_reserved(i) / (1024 ** 3)
            allocated_vram_gb += torch.cuda.memory_allocated(i) / (1024 ** 3)
            max_reserved_vram_gb += torch.cuda.max_memory_reserved(i) / (1024 ** 3)
            peak_allocated_vram_gb += torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        
        return GPUMemorySnapshot(
            free_vram_gb=round(free_vram_gb, 3),
            total_vram_gb=round(total_vram_gb, 3),
            reserved_vram_gb=round(reserved_vram_gb, 3),
            allocated_vram_gb=round(allocated_vram_gb, 3),
            max_reserved_vram_gb=round(max_reserved_vram_gb, 3),
            peak_allocated_vram_gb=round(peak_allocated_vram_gb, 3),
            timestamp=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        logger.warning(f"GPU memory snapshot failed: {e}")
        return GPUMemorySnapshot(timestamp=datetime.utcnow().isoformat())


class WarmupError(Exception):
    """
    Raised when model warmup fails.
    
    Specific reasons:
    - warmup_oom: GPU out of memory during warmup
    - warmup_failed: Other warmup failure
    """
    def __init__(self, reason: str, details: str = ""):
        self.reason = reason
        self.details = details
        super().__init__(f"Warmup failed: {reason}")


# ─────────────────────────────────────────────
# PHASE P4: GPU HEALTH INFO (SAFE)
# ─────────────────────────────────────────────

def get_gpu_health_info() -> Dict[str, Any]:
    """
    Get GPU health information for /health/gpu endpoint.
    
    This function:
    - Never loads models
    - Never allocates GPU memory
    - Never throws uncaught exceptions
    
    Returns:
        Dictionary with GPU health info
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # Get VRAM info
            total_vram_gb = 0.0
            free_vram_gb = 0.0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram_gb += props.total_memory / (1024 ** 3)
                
                # Get free memory
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_vram_gb += free_mem / (1024 ** 3)
            
            return {
                "cuda_available": True,
                "total_vram_gb": round(total_vram_gb, 2),
                "free_vram_gb": round(free_vram_gb, 2),
            }
        else:
            return {
                "cuda_available": False,
                "total_vram_gb": None,
                "free_vram_gb": None,
            }
    except Exception as e:
        logger.warning(f"GPU health check failed: {e}")
        return {
            "cuda_available": False,
            "total_vram_gb": None,
            "free_vram_gb": None,
        }


# ─────────────────────────────────────────────
# Local Execution Functions
# ─────────────────────────────────────────────

def run_local_encoder(
    registry: LocalModelRegistry,
    model_id: str,
    text: str,
) -> Dict[str, Any]:
    """
    Run local encoder for NER / entity extraction.
    
    Args:
        registry: LocalModelRegistry instance
        model_id: Model ID to use
        text: Input text
        
    Returns:
        Dictionary with extracted entities and gpu_memory snapshots
        
    Raises:
        EncoderExecutionError: On any execution failure
        RuntimeError: If model not loaded
        
    PHASE P2 ISOLATION:
    - Wrapped in try/except for ALL torch calls
    - Timeout enforcement (ENCODER_TIMEOUT_SECONDS)
    - GPU sync after execution
    - GPU cache cleared on failure ONLY
    - No retry, no partial output
    
    PHASE P5 TELEMETRY:
    - Memory snapshot before execution
    - Memory snapshot after success/failure
    - Specific OOM classification (encoder_oom)
    """
    execution_failed = False
    memory_before: Optional[GPUMemorySnapshot] = None
    memory_after: Optional[GPUMemorySnapshot] = None
    
    try:
        import torch
        
        # Get model (raises RuntimeError if not loaded)
        tokenizer, model = registry.get(model_id)
        
        # ─────────────────────────────────────────────
        # PHASE P5: MEMORY SNAPSHOT BEFORE
        # ─────────────────────────────────────────────
        memory_before = get_gpu_memory_snapshot()
        
        # ─────────────────────────────────────────────
        # PHASE P2: ENCODER ISOLATION
        # ─────────────────────────────────────────────
        
        with TimeoutHandler(ENCODER_TIMEOUT_SECONDS, "Encoder"):
            # Tokenization (can fail on invalid input)
            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                raise EncoderExecutionError("tokenization_failed", str(e))
            
            # Move to model device
            try:
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                raise EncoderExecutionError("device_transfer_failed", str(e))
            
            # Model inference
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except torch.cuda.OutOfMemoryError:
                execution_failed = True
                # Phase P5: Capture memory on OOM
                memory_after = get_gpu_memory_snapshot()
                raise EncoderExecutionError("encoder_oom", "GPU out of memory during encoder inference")
            except Exception as e:
                execution_failed = True
                error_msg = str(e)
                if "CUDA" in error_msg or "cuda" in error_msg:
                    raise EncoderExecutionError("cuda_error", "GPU error during inference")
                raise EncoderExecutionError("inference_failed", error_msg)
            
            # Validate output shape
            try:
                if not hasattr(outputs, 'logits'):
                    raise EncoderExecutionError("invalid_output", "Model output missing logits")
                if outputs.logits.dim() != 3:
                    raise EncoderExecutionError("invalid_logits_shape", f"Expected 3D logits, got {outputs.logits.dim()}D")
            except EncoderExecutionError:
                raise
            except Exception as e:
                raise EncoderExecutionError("output_validation_failed", str(e))
        
        # GPU sync after successful execution
        _synchronize_gpu()
        
        # ─────────────────────────────────────────────
        # PHASE P5: MEMORY SNAPSHOT AFTER SUCCESS
        # ─────────────────────────────────────────────
        memory_after = get_gpu_memory_snapshot()
        
        # Parse NER outputs and include memory telemetry
        result = parse_ner_outputs(tokenizer, outputs, inputs, text)
        result["gpu_memory"] = {
            "before": memory_before.to_dict() if memory_before else None,
            "after": memory_after.to_dict() if memory_after else None,
        }
        return result
        
    except ExecutionTimeoutError:
        execution_failed = True
        raise EncoderExecutionError("timeout", f"Encoder timed out after {ENCODER_TIMEOUT_SECONDS}s")
    except EncoderExecutionError:
        execution_failed = True
        raise
    except RuntimeError:
        # Model not loaded - re-raise as-is
        raise
    except Exception as e:
        execution_failed = True
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            error_msg = "GPU error during encoder execution"
        raise EncoderExecutionError("unexpected_error", error_msg)
    finally:
        # Clean GPU state on failure ONLY
        if execution_failed:
            _cleanup_gpu_on_failure()


def parse_ner_outputs(
    tokenizer,
    outputs,
    inputs,
    original_text: str,
) -> Dict[str, Any]:
    """
    Parse NER model outputs into structured entities.
    
    Extracts:
    - sections: Legal section references
    - acts: Act names
    - entities: Named entities
    """
    import torch
    import re
    
    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Extract entities from predictions
    entities = []
    current_entity = None
    
    for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
        pred_label = pred.item()
        
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
            continue
        
        # Simple entity extraction (B-I-O scheme)
        if pred_label > 0:  # Non-O label
            if current_entity is None:
                current_entity = {"tokens": [token], "label": pred_label}
            else:
                current_entity["tokens"].append(token)
        else:
            if current_entity is not None:
                entity_text = tokenizer.convert_tokens_to_string(current_entity["tokens"])
                entities.append(entity_text.strip())
                current_entity = None
    
    # Fallback: Extract sections and acts from text using regex
    sections = re.findall(r'section\s+(\d+[a-z]?)', original_text.lower())
    acts = []
    
    act_patterns = [
        (r'\bIPC\b', 'IPC'),
        (r'\bCrPC\b', 'CrPC'),
        (r'\bCPC\b', 'CPC'),
        (r'Indian Penal Code', 'IPC'),
        (r'Criminal Procedure Code', 'CrPC'),
        (r'Civil Procedure Code', 'CPC'),
    ]
    
    for pattern, act_name in act_patterns:
        if re.search(pattern, original_text, re.IGNORECASE):
            acts.append(act_name)
    
    return {
        "facts": {
            "sections": list(set(sections)),
            "acts": list(set(acts)),
            "entities": entities,
        },
        "model": "local",
    }


def run_local_decoder(
    registry: LocalModelRegistry,
    model_id: str,
    prompt: str,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Run local decoder for text generation.
    
    Args:
        registry: LocalModelRegistry instance
        model_id: Model ID to use
        prompt: Input prompt
        max_tokens: Maximum new tokens to generate (capped by MAX_NEW_TOKENS)
        
    Returns:
        Dictionary with generated text and gpu_memory snapshots
        
    Raises:
        DecoderExecutionError: On any execution failure
        RuntimeError: If model not loaded
        
    PHASE P1 SAFETY:
    - max_tokens capped at MAX_NEW_TOKENS (512)
    - Input truncated at MAX_INPUT_TOKENS (4096)
    - do_sample=False (deterministic, no temperature)
    - No raw CUDA traces exposed
    
    PHASE P2 ISOLATION:
    - Wrapped in try/except for ALL torch calls
    - Timeout enforcement (DECODER_TIMEOUT_SECONDS)
    - GPU sync after execution
    - GPU cache cleared on failure ONLY
    - No retry, no partial output
    - NEVER bypasses post-gen verification
    
    PHASE P5 TELEMETRY:
    - Memory snapshot before execution
    - Memory snapshot after success/failure
    - Specific OOM classification (decoder_oom)
    """
    execution_failed = False
    memory_before: Optional[GPUMemorySnapshot] = None
    memory_after: Optional[GPUMemorySnapshot] = None
    
    try:
        import torch
        
        # Get model (raises RuntimeError if not loaded)
        tokenizer, model = registry.get(model_id)
        
        # ─────────────────────────────────────────────
        # PHASE P1: HARD GENERATION LIMITS
        # ─────────────────────────────────────────────
        
        # Enforce max_tokens limit
        effective_max_tokens = min(max_tokens, MAX_NEW_TOKENS)
        if max_tokens > MAX_NEW_TOKENS:
            logger.warning(
                f"max_tokens={max_tokens} exceeds limit, capping to {MAX_NEW_TOKENS}"
            )
        
        # ─────────────────────────────────────────────
        # PHASE P5: MEMORY SNAPSHOT BEFORE
        # ─────────────────────────────────────────────
        memory_before = get_gpu_memory_snapshot()
        
        # ─────────────────────────────────────────────
        # PHASE P2: DECODER ISOLATION
        # ─────────────────────────────────────────────
        
        with TimeoutHandler(DECODER_TIMEOUT_SECONDS, "Decoder"):
            # Tokenization (can fail on invalid input)
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                )
            except Exception as e:
                raise DecoderExecutionError("tokenization_failed", str(e))
            
            # Check input length
            input_length = inputs["input_ids"].shape[1]
            if input_length >= MAX_INPUT_TOKENS:
                logger.warning(
                    f"Input truncated from longer text to {MAX_INPUT_TOKENS} tokens"
                )
            
            # Move to model device
            try:
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                raise DecoderExecutionError("device_transfer_failed", str(e))
            
            # Model generation
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=effective_max_tokens,
                        do_sample=False,  # MANDATORY: Deterministic, no sampling
                        # temperature is NOT configurable - do_sample=False ignores it
                        pad_token_id=tokenizer.eos_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                execution_failed = True
                # Phase P5: Capture memory on OOM
                memory_after = get_gpu_memory_snapshot()
                raise DecoderExecutionError("decoder_oom", "GPU out of memory during decoder generation")
            except Exception as e:
                execution_failed = True
                error_msg = str(e)
                if "CUDA" in error_msg or "cuda" in error_msg:
                    raise DecoderExecutionError("cuda_error", "GPU error during generation")
                raise DecoderExecutionError("generation_failed", error_msg)
            
            # Decode output
            try:
                generated_ids = output_ids[0][input_length:]
                result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            except Exception as e:
                raise DecoderExecutionError("decoding_failed", str(e))
        
        # GPU sync after successful execution
        _synchronize_gpu()
        
        # ─────────────────────────────────────────────
        # PHASE P5: MEMORY SNAPSHOT AFTER SUCCESS
        # ─────────────────────────────────────────────
        memory_after = get_gpu_memory_snapshot()
        
        return {
            "text": result_text,
            "gpu_memory": {
                "before": memory_before.to_dict() if memory_before else None,
                "after": memory_after.to_dict() if memory_after else None,
            },
        }
        
    except ExecutionTimeoutError:
        execution_failed = True
        raise DecoderExecutionError("timeout", f"Decoder timed out after {DECODER_TIMEOUT_SECONDS}s")
    except DecoderExecutionError:
        execution_failed = True
        raise
    except RuntimeError:
        # Model not loaded - re-raise as-is
        raise
    except Exception as e:
        execution_failed = True
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            error_msg = "GPU error during decoder execution"
        raise DecoderExecutionError("unexpected_error", error_msg)
    finally:
        # Clean GPU state on failure ONLY
        if execution_failed:
            _cleanup_gpu_on_failure()


# ─────────────────────────────────────────────
# Model Load Guard
# ─────────────────────────────────────────────

def ensure_model_loaded(
    registry: LocalModelRegistry,
    model_id: str,
    backend: str = "local",
) -> None:
    """
    Ensure a model is loaded before execution.
    
    Args:
        registry: LocalModelRegistry instance
        model_id: Model ID to check
        backend: Backend type ("local" or "api")
        
    Raises:
        RuntimeError: If local model not loaded
    """
    if backend != "local":
        return  # API backends don't need local loading
    
    if not registry.is_loaded(model_id):
        raise RuntimeError(
            f"Local model not loaded yet: {model_id}. "
            "Load manually when GPU is available."
        )
