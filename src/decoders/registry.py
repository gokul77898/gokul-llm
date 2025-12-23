"""
Phase-4: Decoder Registry

Central registry for decoder selection and instantiation.
"""

from typing import Dict, Any
from .base import DecoderInterface
from .local_hf import LocalHFDecoder


def get_decoder(config: Dict[str, Any]) -> DecoderInterface:
    """Get decoder instance from configuration.
    
    Supports decoder types:
    - "qwen": Qwen models (e.g., Qwen2.5-32B-Instruct)
    - "llama": Llama models (e.g., Llama-3-8B-Instruct)
    - "stub": Safe stub decoder (always returns refusal)
    
    Args:
        config: Decoder configuration dict with keys:
            - type: Decoder type ("qwen", "llama", "stub")
            - model_name: HuggingFace model name
            - device: Device ("cpu", "cuda", "mps")
            - dtype: Data type ("float32", "float16", "bfloat16")
            - enable_inference: Enable actual inference (default: False)
    
    Returns:
        DecoderInterface instance
    
    Example:
        config = {
            "type": "qwen",
            "model_name": "Qwen/Qwen2.5-32B-Instruct",
            "device": "cpu",
            "dtype": "float32",
            "enable_inference": False
        }
        decoder = get_decoder(config)
        answer = decoder.generate(prompt)
    """
    
    decoder_type = config.get("type", "stub")
    model_name = config.get("model_name", "Qwen/Qwen2.5-32B-Instruct")
    device = config.get("device", "cpu")
    dtype = config.get("dtype", "float32")
    enable_inference = config.get("enable_inference", False)
    
    # Map decoder types to model names
    if decoder_type == "qwen":
        # Qwen models
        if not model_name or model_name == "default":
            model_name = "Qwen/Qwen2.5-32B-Instruct"
        
        return LocalHFDecoder(
            model_name=model_name,
            device=device,
            dtype=dtype,
            enable_inference=enable_inference
        )
    
    elif decoder_type == "llama":
        # Llama models
        if not model_name or model_name == "default":
            model_name = "meta-llama/Llama-3-8B-Instruct"
        
        return LocalHFDecoder(
            model_name=model_name,
            device=device,
            dtype=dtype,
            enable_inference=enable_inference
        )
    
    elif decoder_type == "stub":
        # Safe stub decoder (always returns refusal)
        return LocalHFDecoder(
            model_name="stub",
            device="cpu",
            dtype="float32",
            enable_inference=False
        )
    
    else:
        raise ValueError(
            f"Unknown decoder type: {decoder_type}. "
            f"Supported types: qwen, llama, stub"
        )


def list_supported_decoders() -> Dict[str, str]:
    """List supported decoder types and their default models.
    
    Returns:
        Dict mapping decoder type to default model name
    """
    return {
        "qwen": "Qwen/Qwen2.5-32B-Instruct",
        "llama": "meta-llama/Llama-3-8B-Instruct",
        "stub": "stub (safe mode, no inference)"
    }
