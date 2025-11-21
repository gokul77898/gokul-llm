"""
Unified Answer Generator

Provides generation interface for both Mamba and Transformer models
with automatic fallback handling.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_answer(
    model_key: str,
    prompt: str,
    context: str = "",
    generation_params: Optional[Dict[str, Any]] = None,
    model_instance = None,
    fallback_enabled: bool = True
) -> Dict[str, Any]:
    """
    Generate answer using specified model
    
    Args:
        model_key: "mamba" or "transformer"
        prompt: Query prompt
        context: Retrieved context
        generation_params: Generation parameters (temperature, max_tokens, etc.)
        model_instance: Preloaded model instance (optional)
        fallback_enabled: Enable fallback to transformer on mamba failure
        
    Returns:
        dict with 'answer', 'model_used', 'fallback_used', 'error'
    """
    generation_params = generation_params or {}
    
    # Default generation params
    max_new_tokens = generation_params.get("max_new_tokens", 256)
    temperature = generation_params.get("temperature", 0.7)
    top_p = generation_params.get("top_p", 0.9)
    top_k = generation_params.get("top_k", 50)
    
    result = {
        "answer": "",
        "model_used": model_key,
        "fallback_used": False,
        "error": None
    }
    
    # Try Mamba generation (auto-detects backend)
    if model_key == "mamba":
        try:
            if model_instance is None:
                # Load Mamba on demand (auto-detects backend)
                logger.info("🎯 Loading Mamba for generation (auto-detect backend)")
                from src.core.mamba_loader import load_mamba_model
                model_instance = load_mamba_model()
            
            # Check if Mamba is actually available
            if not getattr(model_instance, "available", True):
                raise RuntimeError(f"Mamba not available: {model_instance.reason}")
            
            # Detect which backend and use appropriate generation method
            backend = getattr(model_instance, "backend", "unknown")
            
            if backend == "real-mamba":
                logger.info("⚡ Generating with REAL Mamba SSM (CUDA optimized)")
            elif backend == "mamba2":
                logger.info("🍎 Generating with Mamba2 (Mac optimized)")
            else:
                logger.info(f"⚡ Generating with Mamba (backend: {backend})")
            
            # Both backends use the same interface
            answer = model_instance.generate_with_state_space(
                prompt=prompt,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            result["answer"] = answer
            result["backend"] = backend
            logger.info(f"✅ Mamba generation successful ({len(answer)} chars, backend: {backend})")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  Mamba generation failed: {e}")
            result["error"] = str(e)
            
            # Fallback to transformer if enabled
            if fallback_enabled:
                logger.info("→ Attempting fallback to transformer...")
                result["fallback_used"] = True
                result["model_used"] = "transformer"
                model_key = "transformer"
            else:
                result["answer"] = f"Mamba generation failed: {str(e)}"
                return result
    
    # Transformer generation (or fallback)
    if model_key == "transformer":
        try:
            answer = _generate_transformer(
                prompt=prompt,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                model_instance=model_instance
            )
            
            result["answer"] = answer
            result["model_used"] = "transformer"
            logger.info(f"Transformer generation successful ({len(answer)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Transformer generation failed: {e}")
            result["error"] = str(e)
            result["answer"] = f"Generation failed: {str(e)}"
            return result
    
    # Unknown model
    result["error"] = f"Unknown model: {model_key}"
    result["answer"] = f"Error: Unknown model key '{model_key}'"
    return result


def _generate_transformer(
    prompt: str,
    context: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    model_instance = None
) -> str:
    """
    Generate answer using transformer model
    
    Args:
        prompt: Query prompt
        context: Retrieved context
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling
        top_k: Top-k sampling
        model_instance: Preloaded model (optional)
        
    Returns:
        Generated answer string
    """
    import torch
    
    # Load model if not provided
    if model_instance is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        model = model.to(device)
        model.eval()
    else:
        # Use provided model
        if hasattr(model_instance, "tokenizer"):
            tokenizer = model_instance.tokenizer
            model = model_instance.model
            device = model_instance.device
        else:
            # Assume HF model
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model = model_instance
            device = next(model.parameters()).device
    
    # Build input text (shorter context for transformer)
    max_context_length = 512
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    full_text = f"{context}\n\nQuestion: {prompt}\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.strip()
    
    return answer


def get_generation_params(model_key: str) -> Dict[str, Any]:
    """
    Get default generation parameters for model
    
    Args:
        model_key: "mamba" or "transformer"
        
    Returns:
        dict of generation parameters
    """
    if model_key == "mamba":
        return {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
    elif model_key == "transformer":
        return {
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40
        }
    else:
        return {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
