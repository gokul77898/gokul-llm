"""
Generation Module - Phase 4: Graph-Aware Grounded Generation

Orchestrates graph-constrained retrieval with grounded generation.
NO graph logic inside LLM, NO filtering after generation.
"""

from .graph_grounded_generator import (
    GraphGroundedGenerator,
    GroundedAnswerResult,
)

__all__ = [
    "GraphGroundedGenerator",
    "GroundedAnswerResult",
]
