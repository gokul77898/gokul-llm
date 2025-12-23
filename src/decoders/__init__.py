"""
Phase-4: Local Decoder Abstraction

Swappable decoder interface for local LLM inference.
"""

from .base import DecoderInterface
from .registry import get_decoder

__all__ = ['DecoderInterface', 'get_decoder']
