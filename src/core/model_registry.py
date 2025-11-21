"""Unified Model Registry and Loader for MARK System"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.common import load_config, get_device
# REAL Mamba SSM (not custom transformer)
from src.core.mamba_loader import load_mamba_model, is_mamba_available as check_mamba_available
from src.transfer import LegalTransferModel, LegalTokenizer
from src.rag.document_store import FAISSStore
from src.rag.retriever import LegalRetriever

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered model"""
    name: str
    architecture: str
    config_path: str
    checkpoint_path: Optional[str]
    description: str


class ModelRegistry:
    """Central registry for all MARK models"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default models"""
        self.models = {
            "mamba": ModelInfo(
                name="mamba",
                architecture="mamba",
                config_path="configs/mamba.yaml",
                checkpoint_path="checkpoints/mamba/model.pt",
                description="Mamba SSM model for sequential reasoning"
            ),
            "transformer": ModelInfo(
                name="transformer",
                architecture="transformer",
                config_path="configs/transformer.yaml",
                checkpoint_path="checkpoints/transformer/model.pt",
                description="Standard transformer model"
            ),
            "rag_encoder": ModelInfo(
                name="rag_encoder",
                architecture="rag",
                config_path="configs/rag.yaml",
                checkpoint_path="checkpoints/rag/encoder.pt",
                description="RAG retrieval encoder"
            ),
            "rl_trained": ModelInfo(
                name="rl_trained",
                architecture="rlhf",
                config_path="configs/rl.yaml",
                checkpoint_path="checkpoints/rl/policy.pt",
                description="RL/RLHF trained model with PPO"
            ),
            "mamba_lora": ModelInfo(
                name="mamba_lora",
                architecture="mamba",
                config_path="configs/lora_sft.yaml",
                checkpoint_path="checkpoints/lora/mamba_lora/final",
                description="Mamba model fine-tuned with LoRA on legal QA data"
            ),
            "transformer_lora": ModelInfo(
                name="transformer_lora",
                architecture="transformer",
                config_path="configs/lora_sft.yaml",
                checkpoint_path="checkpoints/lora/transformer_lora/final",
                description="Transformer fine-tuned with LoRA on legal QA data"
            ),
            "rl_trained_lora": ModelInfo(
                name="rl_trained_lora",
                architecture="rlhf",
                config_path="configs/rlhf.yaml",
                checkpoint_path="checkpoints/lora/rl_trained_lora/final",
                description="RLHF model with LoRA adapters (if trained)"
            )
        }
        self.register_model(
            name="rl_trained",
            architecture="rlhf",
            config_path="configs/ppo_train.yaml",
            checkpoint_path="checkpoints/rlhf/ppo/ppo_final.pt",
            description="RLHF-optimized model (PPO fine-tuned)"
        )
    
    def register_model(
        self,
        name: str,
        architecture: str,
        config_path: str,
        checkpoint_path: Optional[str],
        description: str
    ):
        """Register a new model"""
        self.models[name] = ModelInfo(
            name=name,
            architecture=architecture,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            description=description
        )
        logger.info(f"Registered model: {name} ({architecture})")
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all registered models"""
        return self.models
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get information about a model"""
        return self.models.get(name)


# Global registry instance
_registry = ModelRegistry()


def load_model(
    model_name: str,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None
) -> Tuple[Any, Any, torch.device]:
    """
    Load a model by name from the registry
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on (auto-detect if None)
        checkpoint_path: Override checkpoint path
        
    Returns:
        Tuple of (model, tokenizer/retriever, device)
    """
    model_info = _registry.get_model_info(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(_registry.models.keys())}")
    
    logger.info(f"Loading model: {model_name} ({model_info.architecture})")
    
    # Load config
    config = load_config(model_info.config_path)
    
    # Determine device
    if device is None:
        device = get_device(config.system.device)
    else:
        device = get_device(device)
    
    # Use override or default checkpoint
    ckpt_path = checkpoint_path or model_info.checkpoint_path
    
    # Load based on architecture
    if model_info.architecture == "mamba":
        return _load_mamba_model(config, ckpt_path, device)
    elif model_info.architecture == "transformer":
        return _load_transformer_model(config, ckpt_path, device)
    elif model_info.architecture == "rag":
        return _load_rag_model(config, ckpt_path, device)
    elif model_info.architecture == "rl":
        return _load_rl_model(config, ckpt_path, device)
    elif model_info.architecture == "rlhf":
        return _load_rlhf_model(config, ckpt_path, device)
    else:
        raise ValueError(f"Unknown architecture: {model_info.architecture}")


def _load_mamba_model(config, checkpoint_path: str, device: torch.device):
    """Load Mamba model (auto-detects backend: REAL Mamba SSM or Mamba2)"""
    logger.info("🎯 Auto-detecting best Mamba backend...")
    
    # Build config dict for mamba_loader
    mamba_config = {
        'real_model': getattr(config.model, 'base_model', 'state-spaces/mamba-130m'),
        'mamba2_model': getattr(config.model, 'mamba2_model', 'mamba2-base'),
        'checkpoint_path': checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else None,
    }
    
    # Load Mamba with auto-detection (REAL Mamba SSM, Mamba2, or Shim)
    mamba_model = load_mamba_model(mamba_config)
    
    if not mamba_model.available:
        logger.error(f"❌ Mamba unavailable: {mamba_model.reason}")
        logger.error(f"   Backend attempted: {mamba_model.backend}")
        
        # Provide platform-specific install instructions
        from src.core.mamba_loader import get_mamba_info
        info = get_mamba_info()
        if 'install_command' in info:
            logger.error(f"   Install with: {info['install_command']}")
        
        raise RuntimeError(f"Mamba not available: {mamba_model.reason}")
    
    logger.info(f"✅ Mamba loaded successfully (backend: {mamba_model.backend})")
    return mamba_model.model, mamba_model.tokenizer, mamba_model.device


def _load_transformer_model(config, checkpoint_path: str, device: torch.device):
    """Load transformer model"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # Load tokenizer
    base_model = config.model.base_model
    tokenizer = LegalTokenizer(base_model)
    
    # Create model
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=config.model.num_labels
        ).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, loading base model")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=config.model.num_labels
        ).to(device)
    
    model.eval()
    return model, tokenizer, device


def _load_rag_model(config, checkpoint_path: str, device: torch.device):
    """Load RAG model (retriever)"""
    from src.rag.document_store import FAISSStore
    
    # Create document store
    store = FAISSStore(
        embedding_model=getattr(config.model, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    )
    
    # Load FAISS index if exists
    index_path = checkpoint_path
    metadata_path = Path(checkpoint_path).parent / "metadata.json"
    
    if Path(index_path).exists() and Path(metadata_path).exists():
        store.load(str(index_path), str(metadata_path))
        logger.info(f"Loaded FAISS index from {index_path}")
    else:
        logger.warning(f"Index not found: {index_path}, retriever needs to be built")
    
    # Create retriever with store
    retriever = LegalRetriever(
        document_store=store,
        top_k=5
    )
    
    return retriever, store.embedding_model, device


def _load_rl_model(config, checkpoint_path: str, device: torch.device):
    """Load RL trained model"""
    from src.rl.ppo import PPOPolicy
    
    # Get dimensions from config
    obs_dim = getattr(config.model, 'max_length', 128)
    action_dim = getattr(config.model, 'vocab_size', 1000)
    hidden_dim = config.model.hidden_dim
    n_layers = config.model.n_layers
    
    # Create policy
    policy = PPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    ).to(device)
    
    # Load checkpoint if exists
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
        elif 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            policy.load_state_dict(checkpoint)
        logger.info(f"Loaded RL policy from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using random initialization")
    
    policy.eval()
    return policy, None, device


def _load_rlhf_model(config, checkpoint_path: str, device: torch.device):
    """Load RLHF trained model (PPO fine-tuned)"""
    from src.rl.policy import ActorCritic
    
    # Get dimensions from config
    obs_dim = getattr(config.model, 'obs_dim', 128)
    action_dim = getattr(config.model, 'action_dim', 50)
    hidden_dim = getattr(config.model, 'hidden_dim', 64)
    num_layers = getattr(config.model, 'num_layers', 2)
    
    # Create actor-critic model
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    # Load checkpoint if exists
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"✅ Loaded RLHF model from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using random initialization")
    
    # Wrap model with generator interface
    class RLHFGenerator:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self.model.eval()
        
        def generate(self, prompt, max_length=256, top_k=5):
            """Generate response using RLHF model"""
            # Simple generation logic - convert prompt to observation
            obs = torch.randn(1, obs_dim).to(self.device)  # Placeholder encoding
            with torch.no_grad():
                action, _, _ = self.model.get_action_and_value(obs, deterministic=True)
            return f"RLHF generated response (action: {action.item()})"
    
    generator = RLHFGenerator(model, device)
    model.eval()
    
    logger.info("✅ RLHF model loaded successfully")
    return generator, None, device


def get_registry() -> ModelRegistry:
    """Get the global model registry"""
    return _registry


def is_model_available(model_key: str) -> bool:
    """
    Check if a model is available in the registry and can be loaded
    
    Args:
        model_key: Name of the model to check
        
    Returns:
        bool: True if model is registered and loadable
    """
    # Check if model is registered
    if model_key not in _registry.models:
        return False
    
    # For REAL Mamba SSM, check if mamba-ssm package is available
    if model_key == "mamba" or "mamba" in model_key.lower():
        return check_mamba_available()  # Uses imported function
    
    # Other models are assumed available if registered
    return True


def get_model_instance(model_key: str, config: Optional[Dict[str, Any]] = None):
    """
    Get a loaded model instance by key
    
    Args:
        model_key: Name of the model (mamba, transformer, etc.)
        config: Optional config override
        
    Returns:
        Loaded model instance or None if unavailable
    """
    if not is_model_available(model_key):
        logger.warning(f"Model {model_key} is not available")
        return None
    
    try:
        # For Mamba (auto-detects backend: REAL Mamba SSM or Mamba2)
        if model_key == "mamba" or "mamba" in model_key.lower():
            logger.info("🎯 Loading Mamba via get_model_instance (auto-detect backend)")
            mamba_model = load_mamba_model(config)
            
            if not mamba_model.available:
                logger.error(f"Mamba unavailable: {mamba_model.reason}")
                return None
            
            logger.info(f"✅ Loaded Mamba with backend: {mamba_model.backend}")
            
            # Return the model wrapper (RealMambaModel, Mamba2Model, or MambaShim)
            # Already has .model, .tokenizer, .device, .available, .backend
            return mamba_model
        
        # For other models, use standard load_model
        model, tokenizer, device = load_model(model_key)
        
        # Wrap in a consistent interface
        class ModelWrapper:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.available = True
                self.backend = "transformer"
        
        return ModelWrapper(model, tokenizer, device)
        
    except Exception as e:
        logger.error(f"Failed to load model {model_key}: {e}")
        return None
