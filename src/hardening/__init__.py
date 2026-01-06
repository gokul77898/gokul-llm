"""
Hardening Module - Phase 8

System hardening with resource limits, adversarial defense,
refusal consistency, and observability.

NO NEW REASONING.
NO ML/LLM.
GUARDRAILS ONLY.
"""

from .resource_limits import (
    ResourceLimits,
    ResourceLimitViolation,
    LimitType,
)

from .adversarial_defense import (
    AdversarialDefense,
    AdversarialPattern,
    AdversarialDetectionResult,
)

from .observability import (
    ObservabilityLogger,
    PhaseType,
)

__all__ = [
    # Resource limits
    "ResourceLimits",
    "ResourceLimitViolation",
    "LimitType",
    # Adversarial defense
    "AdversarialDefense",
    "AdversarialPattern",
    "AdversarialDetectionResult",
    # Observability
    "ObservabilityLogger",
    "PhaseType",
]
