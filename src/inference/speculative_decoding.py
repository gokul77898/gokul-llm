"""
Speculative Decoding Engine

Implements speculative decoding for faster inference by using a smaller draft model
to generate candidate tokens that are then verified by the main model.

Compatible with both Mamba and Transformer models via auto-detection.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import time
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    draft_model_name: str = "gpt2"  # Smaller, faster draft model
    num_speculative_tokens: int = 4  # Number of tokens to speculate ahead
    acceptance_threshold: float = 0.8  # Threshold for accepting speculated tokens
    max_draft_length: int = 32  # Maximum draft sequence length
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    enable_fallback: bool = True  # Fallback to regular decoding if speculation fails


class SpeculativeDecoder:
    """
    Speculative Decoding Engine
    
    Uses a fast draft model to generate candidate tokens, then verifies
    them with the main model for faster overall generation.
    """
    
    def __init__(
        self,
        main_model,
        main_tokenizer,
        config: SpeculativeConfig,
        device: str = "auto"
    ):
        """
        Args:
            main_model: Main generation model (Mamba or Transformer)
            main_tokenizer: Tokenizer for main model
            config: Speculative decoding configuration
            device: Device to run on
        """
        self.main_model = main_model
        self.main_tokenizer = main_tokenizer
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
        
        # Load draft model
        self.draft_model = None
        self.draft_tokenizer = None
        self._load_draft_model()
        
        # Statistics
        self.stats = {
            "total_tokens_generated": 0,
            "total_speculative_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "acceptance_rate": 0.0,
            "speedup_ratio": 1.0,
            "fallback_count": 0
        }
        
        logger.info(f"✅ Speculative Decoder initialized")
        logger.info(f"   Main model: {type(main_model).__name__}")
        logger.info(f"   Draft model: {config.draft_model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Speculative tokens: {config.num_speculative_tokens}")
    
    def _load_draft_model(self):
        """Load the smaller draft model for speculation"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading draft model: {self.config.draft_model_name}")
            
            self.draft_tokenizer = AutoTokenizer.from_pretrained(
                self.config.draft_model_name
            )
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.config.draft_model_name,
                torch_dtype=torch.float32,
                device_map=None
            ).to(self.device)
            
            # Set padding tokens
            if self.draft_tokenizer.pad_token is None:
                self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
            
            self.draft_model.eval()
            logger.info("✅ Draft model loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load draft model: {e}")
            logger.info("   Speculative decoding will be disabled")
            self.draft_model = None
            self.draft_tokenizer = None
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using speculative decoding
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated tokens and statistics
        """
        start_time = time.time()
        
        # Check if speculative decoding is available
        if self.draft_model is None or not self._is_speculative_viable(input_ids):
            logger.info("🔄 Falling back to regular decoding")
            return self._fallback_generate(input_ids, max_new_tokens, **generation_kwargs)
        
        logger.info("⚡ Starting speculative decoding")
        
        # Initialize generation
        current_ids = input_ids.clone()
        generated_tokens = []
        total_accepted = 0
        total_rejected = 0
        
        for step in range(max_new_tokens // self.config.num_speculative_tokens + 1):
            if len(generated_tokens) >= max_new_tokens:
                break
            
            # Generate speculative tokens with draft model
            draft_tokens = self._generate_draft_tokens(current_ids)
            
            if len(draft_tokens) == 0:
                # No draft tokens generated, fall back
                logger.warning("No draft tokens generated, falling back")
                break
            
            # Verify draft tokens with main model
            accepted_tokens, num_accepted = self._verify_draft_tokens(
                current_ids, draft_tokens
            )
            
            if num_accepted > 0:
                # Accept verified tokens
                generated_tokens.extend(accepted_tokens)
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor(accepted_tokens, device=self.device).unsqueeze(0)
                ], dim=1)
                total_accepted += num_accepted
                total_rejected += len(draft_tokens) - num_accepted
                
                logger.debug(f"Step {step}: Accepted {num_accepted}/{len(draft_tokens)} tokens")
            else:
                # No tokens accepted, generate one token normally
                next_token = self._generate_single_token(current_ids)
                if next_token is not None:
                    generated_tokens.append(next_token)
                    current_ids = torch.cat([
                        current_ids,
                        torch.tensor([[next_token]], device=self.device)
                    ], dim=1)
                total_rejected += len(draft_tokens)
        
        # Update statistics
        generation_time = time.time() - start_time
        self._update_stats(total_accepted, total_rejected, generation_time)
        
        # Prepare result
        result = {
            "generated_ids": torch.tensor(generated_tokens, device=self.device),
            "full_sequence": current_ids,
            "num_tokens_generated": len(generated_tokens),
            "accepted_tokens": total_accepted,
            "rejected_tokens": total_rejected,
            "acceptance_rate": total_accepted / (total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else 0.0,
            "generation_time": generation_time,
            "tokens_per_second": len(generated_tokens) / generation_time if generation_time > 0 else 0.0,
            "speculative_enabled": True
        }
        
        logger.info(f"✅ Speculative decoding complete")
        logger.info(f"   Generated: {len(generated_tokens)} tokens")
        logger.info(f"   Accepted: {total_accepted}, Rejected: {total_rejected}")
        logger.info(f"   Acceptance rate: {result['acceptance_rate']:.2%}")
        logger.info(f"   Speed: {result['tokens_per_second']:.1f} t/s")
        
        return result
    
    def _generate_draft_tokens(self, input_ids: torch.Tensor) -> List[int]:
        """Generate draft tokens using the smaller model"""
        try:
            with torch.no_grad():
                draft_outputs = self.draft_model.generate(
                    input_ids,
                    max_new_tokens=self.config.num_speculative_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.draft_tokenizer.pad_token_id,
                    eos_token_id=self.draft_tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Extract new tokens
            new_tokens = draft_outputs.sequences[0][input_ids.shape[1]:].tolist()
            return new_tokens
            
        except Exception as e:
            logger.warning(f"Draft generation failed: {e}")
            return []
    
    def _verify_draft_tokens(
        self, 
        input_ids: torch.Tensor, 
        draft_tokens: List[int]
    ) -> Tuple[List[int], int]:
        """Verify draft tokens using the main model"""
        if len(draft_tokens) == 0:
            return [], 0
        
        try:
            # Create sequence with draft tokens
            draft_sequence = torch.cat([
                input_ids,
                torch.tensor(draft_tokens, device=self.device).unsqueeze(0)
            ], dim=1)
            
            # Get main model probabilities for the draft sequence
            with torch.no_grad():
                if hasattr(self.main_model, 'generate_with_state_space'):
                    # Mamba model - use state space generation
                    # For verification, we need to get logits, so we'll use forward pass
                    outputs = self.main_model.model(draft_sequence)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                else:
                    # Transformer model
                    outputs = self.main_model(draft_sequence)
                    logits = outputs.logits
            
            # Verify each draft token
            accepted_tokens = []
            for i, draft_token in enumerate(draft_tokens):
                token_pos = input_ids.shape[1] + i - 1  # Position of the token being verified
                
                if token_pos >= 0 and token_pos < logits.shape[1]:
                    # Get probability distribution at this position
                    token_logits = logits[0, token_pos, :]
                    token_probs = F.softmax(token_logits, dim=-1)
                    
                    # Check if draft token probability is above threshold
                    draft_prob = token_probs[draft_token].item()
                    
                    if draft_prob >= self.config.acceptance_threshold:
                        accepted_tokens.append(draft_token)
                    else:
                        # Token rejected, stop verification
                        break
                else:
                    break
            
            return accepted_tokens, len(accepted_tokens)
            
        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            return [], 0
    
    def _generate_single_token(self, input_ids: torch.Tensor) -> Optional[int]:
        """Generate a single token using the main model"""
        try:
            with torch.no_grad():
                if hasattr(self.main_model, 'generate_with_state_space'):
                    # Mamba model
                    outputs = self.main_model.model.generate(
                        input_ids,
                        max_new_tokens=1,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.main_tokenizer.pad_token_id,
                        eos_token_id=self.main_tokenizer.eos_token_id
                    )
                else:
                    # Transformer model
                    outputs = self.main_model.generate(
                        input_ids,
                        max_new_tokens=1,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.main_tokenizer.pad_token_id,
                        eos_token_id=self.main_tokenizer.eos_token_id
                    )
            
            # Extract the new token
            if outputs.shape[1] > input_ids.shape[1]:
                return outputs[0, -1].item()
            return None
            
        except Exception as e:
            logger.warning(f"Single token generation failed: {e}")
            return None
    
    def _is_speculative_viable(self, input_ids: torch.Tensor) -> bool:
        """Check if speculative decoding is viable for this input"""
        # Check sequence length (too short sequences may not benefit)
        if input_ids.shape[1] < 10:
            return False
        
        # Check if models are compatible
        if self.draft_model is None:
            return False
        
        # Check device compatibility
        if input_ids.device != self.device:
            return False
        
        return True
    
    def _fallback_generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int, 
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to regular generation without speculation"""
        start_time = time.time()
        self.stats["fallback_count"] += 1
        
        try:
            with torch.no_grad():
                if hasattr(self.main_model, 'generate_with_state_space'):
                    # Mamba model
                    outputs = self.main_model.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.main_tokenizer.pad_token_id,
                        eos_token_id=self.main_tokenizer.eos_token_id
                    )
                else:
                    # Transformer model
                    outputs = self.main_model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.main_tokenizer.pad_token_id,
                        eos_token_id=self.main_tokenizer.eos_token_id
                    )
            
            generation_time = time.time() - start_time
            generated_tokens = outputs[0][input_ids.shape[1]:].tolist()
            
            return {
                "generated_ids": outputs[0][input_ids.shape[1]:],
                "full_sequence": outputs,
                "num_tokens_generated": len(generated_tokens),
                "accepted_tokens": 0,
                "rejected_tokens": 0,
                "acceptance_rate": 0.0,
                "generation_time": generation_time,
                "tokens_per_second": len(generated_tokens) / generation_time if generation_time > 0 else 0.0,
                "speculative_enabled": False
            }
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            raise
    
    def _update_stats(self, accepted: int, rejected: int, generation_time: float):
        """Update generation statistics"""
        self.stats["total_speculative_tokens"] += accepted + rejected
        self.stats["accepted_tokens"] += accepted
        self.stats["rejected_tokens"] += rejected
        self.stats["total_tokens_generated"] += accepted
        
        if self.stats["total_speculative_tokens"] > 0:
            self.stats["acceptance_rate"] = (
                self.stats["accepted_tokens"] / self.stats["total_speculative_tokens"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get speculative decoding statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_tokens_generated": 0,
            "total_speculative_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "acceptance_rate": 0.0,
            "speedup_ratio": 1.0,
            "fallback_count": 0
        }


def create_speculative_decoder(
    main_model,
    main_tokenizer,
    draft_model_name: str = "gpt2",
    num_speculative_tokens: int = 4,
    device: str = "auto"
) -> SpeculativeDecoder:
    """
    Factory function to create a speculative decoder
    
    Args:
        main_model: Main generation model (from mamba_loader or model_registry)
        main_tokenizer: Tokenizer for main model
        draft_model_name: Name of smaller draft model
        num_speculative_tokens: Number of tokens to speculate
        device: Device to run on
        
    Returns:
        Configured SpeculativeDecoder
    """
    config = SpeculativeConfig(
        draft_model_name=draft_model_name,
        num_speculative_tokens=num_speculative_tokens
    )
    
    return SpeculativeDecoder(
        main_model=main_model,
        main_tokenizer=main_tokenizer,
        config=config,
        device=device
    )


# Integration with existing model loading
def integrate_speculative_decoding():
    """
    Integration function to add speculative decoding to existing models
    
    This can be called from the main generation pipeline to enable
    speculative decoding for supported models.
    """
    try:
        from src.core.mamba_loader import load_mamba_model
        from src.core.model_registry import get_model_instance
        
        logger.info("🔗 Integrating speculative decoding with existing models")
        
        # Example integration - this would be called from the generator
        def create_speculative_generator(model_key: str = "mamba"):
            """Create a generator with speculative decoding enabled"""
            
            # Load main model
            if model_key == "mamba":
                main_model = load_mamba_model()
                main_tokenizer = main_model.tokenizer if hasattr(main_model, 'tokenizer') else None
            else:
                main_model = get_model_instance(model_key)
                main_tokenizer = main_model.tokenizer if hasattr(main_model, 'tokenizer') else None
            
            if main_model and main_model.available and main_tokenizer:
                # Create speculative decoder
                spec_decoder = create_speculative_decoder(
                    main_model=main_model,
                    main_tokenizer=main_tokenizer,
                    draft_model_name="gpt2",  # Fast draft model
                    num_speculative_tokens=4
                )
                
                logger.info("✅ Speculative decoding integrated successfully")
                return spec_decoder
            else:
                logger.warning("⚠️  Main model not available for speculative decoding")
                return None
        
        return create_speculative_generator
        
    except ImportError as e:
        logger.warning(f"⚠️  Could not integrate speculative decoding: {e}")
        return None
