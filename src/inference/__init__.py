"""
Inference Engine Package - Phase 0: Minimal MoE Inference

Only exports MoERouter for clean inference path.
Optimization modules disabled for Phase 0.
"""

from .moe_router import MoERouter

__all__ = ['MoERouter']
