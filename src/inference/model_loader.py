"""
Canonical Local Model Loader

This is the ONLY abstraction for model loading in the MARK system.
All models are loaded locally via LocalModelRegistry.

NO remote inference. NO HF API. NO hosted models.
This system performs 100% local inference.
"""

from src.inference.local_models import LocalModelRegistry


class ModelLoader:
    """
    Canonical local-only model loader.
    
    Usage:
        loader = ModelLoader(device="cuda")
        encoder = loader.load_encoder("model-name")
        decoder = loader.load_decoder("model-name")
    
    Rules:
    - Encoder and decoder are loaded explicitly
    - No magic abstraction
    - No global state
    - No shared singleton
    - Device must be passed explicitly
    """
    
    def __init__(self, device="cpu"):
        self.registry = LocalModelRegistry()
        self.device = device

    def load_encoder(self, name):
        """Load encoder model locally."""
        return self.registry.load_encoder(name)

    def load_decoder(self, name):
        """Load decoder model locally."""
        return self.registry.load_decoder(name)
