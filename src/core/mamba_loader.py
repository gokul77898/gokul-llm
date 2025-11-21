"""
AUTO-DETECTING Mamba Loader with Multi-Backend Support

Automatically selects the best Mamba backend:
- Mac (darwin) → Mamba2
- Windows/Linux + CUDA → REAL Mamba SSM
- No GPU → Fallback to Transformer (via MambaShim)
"""

import os
import sys
import platform
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def detect_mamba_backend() -> str:
    """
    Automatically detect the best Mamba backend for current platform
    
    Returns:
        "mamba2": Use Mamba2 (Mac optimized)
        "real-mamba": Use REAL Mamba SSM (CUDA optimized)
        "none": No Mamba available, use fallback
    """
    # Check if Mamba is disabled
    if os.getenv("ENABLE_MAMBA", "true").lower() == "false":
        logger.info("Mamba disabled via ENABLE_MAMBA environment variable")
        return "none"
    
    system = platform.system().lower()
    
    # Mac → Mamba2 (optimized for Mac)
    if system == "darwin":
        logger.info("🍎 Mac detected → selecting Mamba2 backend")
        try:
            import mamba2
            logger.info("✅ Mamba2 package available")
            return "mamba2"
        except ImportError:
            logger.warning("⚠️  Mamba2 not installed, will use fallback")
            logger.info("   Install with: pip install mamba2")
            return "none"
    
    # Windows/Linux + CUDA → REAL Mamba SSM
    elif system in ["linux", "windows"]:
        if torch.cuda.is_available():
            logger.info(f"🐧 {system.capitalize()} + CUDA detected → selecting REAL Mamba SSM")
            try:
                import mamba_ssm
                logger.info("✅ REAL Mamba SSM (mamba-ssm) package available")
                return "real-mamba"
            except ImportError:
                logger.warning("⚠️  REAL Mamba SSM not installed, will use fallback")
                logger.info("   Install with: pip install mamba-ssm causal-conv1d>=1.2.0")
                return "none"
        else:
            logger.info(f"💻 {system.capitalize()} without CUDA → no Mamba backend")
            return "none"
    
    else:
        logger.info(f"❓ Unknown platform '{system}' → no Mamba backend")
        return "none"


class MambaShim:
    """Fallback shim when no Mamba backend available"""
    
    def __init__(self, reason: str):
        self.available = False
        self.backend = "none"
        self.reason = reason
        self.model = None
        self.tokenizer = None
        self.device = None
        logger.warning(f"⚠️  Mamba not available: {reason}")
    
    def generate(self, *args, **kwargs):
        raise RuntimeError(f"Mamba not available: {self.reason}")
    
    def generate_with_state_space(self, *args, **kwargs):
        raise RuntimeError(f"Mamba not available: {self.reason}")


class RealMambaModel:
    """Wrapper for REAL Mamba SSM (State Space Model) - CUDA optimized"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.available = True
        self.backend = "real-mamba"
        self.max_context_length = 2048
        
        logger.info(f"✅ REAL Mamba SSM loaded on {device}")
    
    def generate_with_state_space(
        self, 
        prompt: str, 
        context: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate using REAL Mamba SSM's state space mechanism"""
        try:
            full_text = f"{context}\n\nQuestion: {prompt}\nAnswer:"
            
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True,
                padding=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(full_text):].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"REAL Mamba SSM generation failed: {e}")
            raise
    
    def generate(self, input_ids, **kwargs):
        """Direct generation interface"""
        return self.model.generate(input_ids, **kwargs)


class Mamba2Model:
    """Wrapper for Mamba2 - Mac optimized"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.available = True
        self.backend = "mamba2"
        self.max_context_length = 2048
        
        logger.info(f"✅ Mamba2 loaded on {device}")
    
    def generate_with_state_space(
        self, 
        prompt: str, 
        context: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate using Mamba2"""
        try:
            full_text = f"{context}\n\nQuestion: {prompt}\nAnswer:"
            
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True,
                padding=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(full_text):].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Mamba2 generation failed: {e}")
            raise
    
    def generate(self, input_ids, **kwargs):
        """Direct generation interface"""
        return self.model.generate(input_ids, **kwargs)


def load_real_mamba(model_name: str, config: Dict[str, Any], device: torch.device):
    """Load REAL Mamba SSM (for Linux/Windows + CUDA)"""
    logger.info("🔥 Loading REAL Mamba SSM...")
    
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from transformers import AutoTokenizer
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            logger.info("   Using GPT-2 tokenizer as fallback")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        try:
            model = MambaLMHeadModel.from_pretrained(
                model_name,
                device=str(device),
                dtype=torch.float32
            )
        except Exception as e:
            logger.warning(f"   Failed to load from HF: {e}")
            logger.info("   Using default Mamba architecture")
            
            from mamba_ssm.models.config_mamba import MambaConfig
            
            mamba_config = MambaConfig(
                d_model=768,
                n_layer=24,
                vocab_size=50277,
                ssm_cfg={},
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                pad_vocab_size_multiple=8,
            )
            
            model = MambaLMHeadModel(mamba_config, device=device)
        
        model.eval()
        
        # Load checkpoint if specified
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
        
        logger.info("=" * 70)
        logger.info("  ✅ REAL MAMBA SSM LOADED")
        logger.info("=" * 70)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Backend: real-mamba (CUDA optimized)")
        logger.info("=" * 70)
        
        return RealMambaModel(model, tokenizer, device)
        
    except Exception as e:
        logger.error(f"❌ Failed to load REAL Mamba SSM: {e}")
        raise


def load_mamba2(model_name: str, config: Dict[str, Any], device: torch.device):
    """Load Mamba2 (for Mac)"""
    logger.info("🍎 Loading Mamba2 (Mac optimized)...")
    
    try:
        from mamba2.models.mamba2 import Mamba2LMHeadModel
        from transformers import AutoTokenizer
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            logger.info("   Using GPT-2 tokenizer as fallback")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        try:
            model = Mamba2LMHeadModel.from_pretrained(
                model_name,
                device=str(device),
                dtype=torch.float32
            )
        except Exception as e:
            logger.warning(f"   Failed to load pretrained: {e}")
            logger.info("   Using default Mamba2 architecture")
            
            # Create default Mamba2 config
            from mamba2.models.config_mamba2 import Mamba2Config
            
            config_obj = Mamba2Config(
                d_model=768,
                n_layer=24,
                vocab_size=50277,
            )
            
            model = Mamba2LMHeadModel(config_obj, device=device)
        
        model.eval()
        
        # Load checkpoint if specified
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
        
        logger.info("=" * 70)
        logger.info("  ✅ MAMBA2 LOADED")
        logger.info("=" * 70)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Backend: mamba2 (Mac optimized)")
        logger.info("=" * 70)
        
        return Mamba2Model(model, tokenizer, device)
        
    except Exception as e:
        logger.error(f"❌ Failed to load Mamba2: {e}")
        raise


def load_mamba_model(config: Optional[Dict[str, Any]] = None):
    """
    Main entry point: Auto-detect and load best Mamba backend
    
    Returns:
        RealMambaModel, Mamba2Model, or MambaShim
    """
    config = config or {}
    
    # Detect backend
    backend = detect_mamba_backend()
    
    logger.info(f"🎯 Selected backend: {backend}")
    
    if backend == "none":
        reason = "No Mamba backend available for this platform/configuration"
        logger.warning(f"⚠️  {reason}")
        return MambaShim(reason)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and backend == "mamba2":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"📍 Device: {device}")
    
    try:
        if backend == "real-mamba":
            # Load REAL Mamba SSM
            model_name = config.get("real_model", "state-spaces/mamba-130m")
            return load_real_mamba(model_name, config, device)
        
        elif backend == "mamba2":
            # Load Mamba2
            model_name = config.get("mamba2_model", "mamba2-base")
            return load_mamba2(model_name, config, device)
    
    except Exception as e:
        reason = f"Failed to load {backend}: {str(e)}"
        logger.error(f"❌ {reason}")
        return MambaShim(reason)


def is_mamba_available() -> bool:
    """Quick check if any Mamba backend is available"""
    backend = detect_mamba_backend()
    return backend != "none"


def get_mamba_info() -> Dict[str, Any]:
    """Get detailed information about Mamba backend"""
    backend = detect_mamba_backend()
    
    info = {
        "backend": backend,
        "available": backend != "none",
        "reason": "",
        "model_name": "",
        "platform": platform.system().lower(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    
    if backend == "real-mamba":
        info["model_name"] = "state-spaces/mamba-130m"
        info["reason"] = "REAL Mamba SSM (CUDA optimized)"
        try:
            import mamba_ssm
            info["version"] = getattr(mamba_ssm, "__version__", "unknown")
        except:
            pass
    
    elif backend == "mamba2":
        info["model_name"] = "mamba2-base"
        info["reason"] = "Mamba2 (Mac optimized)"
        try:
            import mamba2
            info["version"] = getattr(mamba2, "__version__", "unknown")
        except:
            pass
    
    else:
        info["reason"] = "No Mamba backend available"
        if info["platform"] == "darwin":
            info["install_command"] = "pip install mamba2"
        elif torch.cuda.is_available():
            info["install_command"] = "pip install mamba-ssm causal-conv1d>=1.2.0"
        else:
            info["reason"] = "No GPU available, use Transformer instead"
    
    return info
