"""
RAG Context - Phase R4: Context Assembler

Assembles validated evidence into a STRICT, bounded, auditable context
that is safe to send to a decoder later.

Controls:
- Token budget
- Evidence ordering
- Citation formatting
- Evidence completeness

Phase R4 does NOT:
- Perform retrieval
- Perform filtering
- Call LLMs/decoders
- Integrate MoE

The decoder never sees raw documents — only assembled evidence blocks.
"""

from .token_budget import TokenBudget, BudgetConfig
from .citation import CitationFormatter, Citation
from .formatter import EvidenceFormatter, FormattedEvidence
from .assembler import ContextAssembler, ContextResult, ContextRefusalReason

__all__ = [
    "TokenBudget",
    "BudgetConfig",
    "CitationFormatter",
    "Citation",
    "EvidenceFormatter",
    "FormattedEvidence",
    "ContextAssembler",
    "ContextResult",
    "ContextRefusalReason",
]
