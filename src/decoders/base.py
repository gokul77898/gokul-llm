"""
Phase-4: Decoder Base Interface

Abstract interface for all decoder implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional


class DecoderInterface(ABC):
    """Abstract base class for all decoder implementations.
    
    All decoders must implement:
    - generate(prompt: str) -> str
    - name property
    """
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (optional)
            temperature: Sampling temperature (default: 0.0 for deterministic)
        
        Returns:
            Generated text
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return decoder name."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
