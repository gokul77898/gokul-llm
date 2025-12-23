"""
Phase-7: Observability & Contracts - Structured Logging

Structured logging for retrieval, synthesis, grounding failures, and guard triggers.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any


logger = logging.getLogger(__name__)


class StructuredLogger:
    """Structured logger for RAG operations."""
    
    @staticmethod
    def log_structured(
        event_type: str,
        data: Dict[str, Any],
        level: int = logging.INFO
    ) -> None:
        """Log structured event.
        
        Args:
            event_type: Type of event (retrieval, synthesis, etc.)
            data: Event data dictionary
            level: Logging level
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **data
        }
        
        logger.log(level, json.dumps(log_entry))


def log_retrieval(
    query: str,
    retrieved_count: int,
    semantic_ids: List[str],
    top_k: int,
    collection_name: str,
    duration_ms: Optional[float] = None
) -> None:
    """Log retrieval operation.
    
    Args:
        query: User query
        retrieved_count: Number of chunks retrieved
        semantic_ids: List of retrieved semantic IDs
        top_k: Top-K parameter
        collection_name: ChromaDB collection name
        duration_ms: Retrieval duration in milliseconds
    """
    data = {
        "query": query,
        "retrieved_count": retrieved_count,
        "semantic_ids": semantic_ids,
        "top_k": top_k,
        "collection_name": collection_name,
    }
    
    if duration_ms is not None:
        data["duration_ms"] = duration_ms
    
    StructuredLogger.log_structured("retrieval", data, logging.INFO)
    logger.info(f"Retrieved {retrieved_count} chunks for query: {query[:50]}...")


def log_synthesis(
    query: str,
    phase: str,
    is_sufficient: bool,
    is_grounded: bool,
    cited_sources: List[str],
    retrieved_semantic_ids: List[str],
    answer_length: int,
    duration_ms: Optional[float] = None
) -> None:
    """Log synthesis operation.
    
    Args:
        query: User query
        phase: Phase identifier (C2, C3, C3+)
        is_sufficient: Evidence sufficiency status
        is_grounded: Grounding validation status
        cited_sources: List of cited semantic IDs
        retrieved_semantic_ids: List of retrieved semantic IDs
        answer_length: Length of generated answer
        duration_ms: Synthesis duration in milliseconds
    """
    data = {
        "query": query,
        "phase": phase,
        "is_sufficient": is_sufficient,
        "is_grounded": is_grounded,
        "cited_sources": cited_sources,
        "retrieved_semantic_ids": retrieved_semantic_ids,
        "answer_length": answer_length,
        "citation_count": len(cited_sources),
    }
    
    if duration_ms is not None:
        data["duration_ms"] = duration_ms
    
    StructuredLogger.log_structured("synthesis", data, logging.INFO)
    logger.info(f"Synthesis complete: phase={phase}, grounded={is_grounded}, citations={len(cited_sources)}")


def log_grounding_failure(
    query: str,
    phase: str,
    failure_type: str,
    reason: str,
    retrieved_semantic_ids: List[str],
    cited_sources: Optional[List[str]] = None,
    invalid_citations: Optional[List[str]] = None,
    uncovered_claims: Optional[List[str]] = None
) -> None:
    """Log grounding failure.
    
    Args:
        query: User query
        phase: Phase identifier (C2, C3, C3+)
        failure_type: Type of failure (insufficient_evidence, invalid_citations, uncovered_claims)
        reason: Detailed failure reason
        retrieved_semantic_ids: List of retrieved semantic IDs
        cited_sources: List of cited semantic IDs (if any)
        invalid_citations: List of invalid citations
        uncovered_claims: List of uncovered claims
    """
    data = {
        "query": query,
        "phase": phase,
        "failure_type": failure_type,
        "reason": reason,
        "retrieved_semantic_ids": retrieved_semantic_ids,
    }
    
    if cited_sources is not None:
        data["cited_sources"] = cited_sources
    
    if invalid_citations is not None:
        data["invalid_citations"] = invalid_citations
    
    if uncovered_claims is not None:
        data["uncovered_claims"] = uncovered_claims
        data["uncovered_count"] = len(uncovered_claims)
    
    StructuredLogger.log_structured("grounding_failure", data, logging.WARNING)
    logger.warning(f"Grounding failure: {failure_type} - {reason}")


def log_inference_guard_trigger(
    guard_type: str,
    reason: str,
    model_name: str,
    device: str,
    enable_inference: bool,
    env_var_set: bool
) -> None:
    """Log inference guard trigger.
    
    Args:
        guard_type: Type of guard (config_disabled, env_var_missing, gpu_required)
        reason: Detailed reason for guard trigger
        model_name: Model name
        device: Device configuration
        enable_inference: Inference enabled flag
        env_var_set: ALLOW_LLM_INFERENCE environment variable status
    """
    data = {
        "guard_type": guard_type,
        "reason": reason,
        "model_name": model_name,
        "device": device,
        "enable_inference": enable_inference,
        "env_var_set": env_var_set,
    }
    
    StructuredLogger.log_structured("inference_guard_trigger", data, logging.WARNING)
    logger.warning(f"Inference guard triggered: {guard_type} - {reason}")
