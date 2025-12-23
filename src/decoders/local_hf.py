"""
Phase-4: Local HuggingFace Decoder
Phase-5: Inference Safety & Runtime Guards

Local decoder implementation using HuggingFace transformers.
Supports safe stub mode (no inference) by default.
Enforces runtime guards to prevent accidental inference.
"""

import logging
from typing import Optional
from .base import DecoderInterface

try:
    from src.runtime.guards import assert_inference_allowed
    GUARDS_AVAILABLE = True
except ImportError:
    GUARDS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalHFDecoder(DecoderInterface):
    """Local HuggingFace decoder with safe stub mode.
    
    When enable_inference=False (default), returns deterministic stub response.
    When enable_inference=True, loads and runs actual model (NOT IMPLEMENTED).
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-32B-Instruct")
        device: Device to use ("cpu", "cuda", "mps")
        dtype: Data type ("float32", "float16", "bfloat16")
        enable_inference: Enable actual inference (default: False for safety)
    """
    
    REFUSAL_MESSAGE = "I cannot answer based on the provided documents."
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "float32",
        enable_inference: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_inference = enable_inference
        
        # DO NOT load model in stub mode
        if self.enable_inference:
            raise NotImplementedError(
                "Inference mode not implemented. "
                "Phase-4 only implements safe stub mode (enable_inference=False)."
            )
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        """Generate text from prompt.
        
        Phase-5: Enforces runtime guards before inference.
        
        In stub mode (enable_inference=False), returns deterministic refusal.
        In inference mode (enable_inference=True), checks runtime guards first.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (optional)
            temperature: Sampling temperature (default: 0.0)
        
        Returns:
            Generated text (stub response in safe mode or if guards fail)
        """
        if not self.enable_inference:
            # Safe stub mode: return deterministic refusal
            logger.debug("Stub mode: returning C3 refusal")
            return self.REFUSAL_MESSAGE
        
        # Phase-5: Enforce runtime guards
        if GUARDS_AVAILABLE:
            allowed, reason = assert_inference_allowed(
                enable_inference=self.enable_inference,
                model_name=self.model_name,
                device=self.device
            )
            
            if not allowed:
                # Guard failed: log and return C3 refusal
                logger.warning(f"Inference guard blocked generation: {reason}")
                return self.REFUSAL_MESSAGE
            
            # Guards passed: would proceed with inference
            logger.info(f"Inference guards passed: {reason}")
        else:
            logger.warning("Runtime guards not available - proceeding without safety checks")
        
        # Inference mode not implemented (Phase-4 scope)
        raise NotImplementedError("Inference mode not implemented in Phase-4")
    
    @property
    def name(self) -> str:
        """Return decoder name."""
        if self.enable_inference:
            return f"LocalHF({self.model_name})"
        else:
            return f"LocalHF-Stub({self.model_name})"
    
    def __repr__(self) -> str:
        return (
            f"LocalHFDecoder(model_name={self.model_name}, "
            f"device={self.device}, dtype={self.dtype}, "
            f"enable_inference={self.enable_inference})"
        )
