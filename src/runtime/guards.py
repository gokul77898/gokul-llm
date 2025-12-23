"""
Phase-5: Inference Safety & Runtime Guards

Prevents accidental inference, GPU usage, or model loading.
"""

import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class InferenceGuardResult:
    """Result of inference guard check."""
    allowed: bool
    reason: str


class InferenceGuard:
    """Runtime guard to prevent accidental inference.
    
    Checks multiple safety conditions before allowing inference:
    1. enable_inference must be True in config
    2. ALLOW_LLM_INFERENCE=1 environment variable must be set
    3. Large models require GPU (device != cpu)
    
    If any check fails, inference is blocked and C3 refusal is returned.
    """
    
    # Models that require GPU (>10B parameters)
    LARGE_MODELS = [
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3-70B-Instruct",
        "meta-llama/Llama-2-70B-chat",
    ]
    
    @staticmethod
    def check_inference_allowed(
        enable_inference: bool,
        model_name: str,
        device: str
    ) -> InferenceGuardResult:
        """Check if inference is allowed.
        
        Args:
            enable_inference: Config flag for inference
            model_name: Model name to check
            device: Device to use (cpu, cuda, mps)
        
        Returns:
            InferenceGuardResult with allowed flag and reason
        """
        
        # Check 1: enable_inference must be True
        if not enable_inference:
            return InferenceGuardResult(
                allowed=False,
                reason="Inference disabled by config (enable_inference=false)"
            )
        
        # Check 2: Environment variable must be set
        env_flag = os.environ.get("ALLOW_LLM_INFERENCE", "0")
        if env_flag != "1":
            return InferenceGuardResult(
                allowed=False,
                reason="Inference blocked: ALLOW_LLM_INFERENCE environment variable not set to 1"
            )
        
        # Check 3: Large models require GPU
        if model_name in InferenceGuard.LARGE_MODELS:
            if device == "cpu":
                return InferenceGuardResult(
                    allowed=False,
                    reason=f"GPU required for large model {model_name} (device=cpu not allowed)"
                )
        
        # All checks passed
        return InferenceGuardResult(
            allowed=True,
            reason="Inference allowed: all safety checks passed"
        )
    
    @staticmethod
    def assert_inference_allowed(
        enable_inference: bool,
        model_name: str,
        device: str
    ) -> Tuple[bool, str]:
        """Assert that inference is allowed.
        
        This is the main entry point for checking inference safety.
        
        Args:
            enable_inference: Config flag for inference
            model_name: Model name to check
            device: Device to use (cpu, cuda, mps)
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        result = InferenceGuard.check_inference_allowed(
            enable_inference, model_name, device
        )
        
        if not result.allowed:
            logger.warning(f"Inference blocked: {result.reason}")
        else:
            logger.info(f"Inference allowed: {result.reason}")
        
        return result.allowed, result.reason


def assert_inference_allowed(
    enable_inference: bool,
    model_name: str,
    device: str
) -> Tuple[bool, str]:
    """Convenience function for inference guard check.
    
    Args:
        enable_inference: Config flag for inference
        model_name: Model name to check
        device: Device to use (cpu, cuda, mps)
    
    Returns:
        Tuple of (allowed: bool, reason: str)
    """
    return InferenceGuard.assert_inference_allowed(
        enable_inference, model_name, device
    )
