"""
Retrieval Validator Pipeline

Phase R3: Retrieval Validation & Filtering

Main validation pipeline combining:
- Statute validation
- Evidence filtering
- Threshold checking

Returns structured validation results with refusal reasons.

NO LLMs used in this module. Failure = refusal, never guess.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

from .threshold import EvidenceThreshold, ThresholdConfig, ThresholdStatus
from .statute_validator import StatuteValidator, StatuteMatch
from .evidence_filter import EvidenceFilter, FilterResult, FilterReason


class RefusalReason(str, Enum):
    """Machine-readable refusal reasons."""
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STATUTE_MISMATCH = "statute_mismatch"
    SECTION_MISMATCH = "section_mismatch"
    REPEALED_LAW = "repealed_law"
    LOW_CONFIDENCE = "low_confidence"
    NO_VALID_CHUNKS = "no_valid_chunks"


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASS = "pass"
    REFUSE = "refuse"


@dataclass
class RejectedChunk:
    """Information about a rejected chunk."""
    chunk_id: str
    reason: str
    details: Optional[str] = None


@dataclass
class AcceptedChunk:
    """Information about an accepted chunk."""
    chunk_id: str
    text: str
    section: Optional[str]
    act: Optional[str]
    score: float
    adjusted_score: float  # After statute penalty


@dataclass
class ValidationResult:
    """
    Result of validation pipeline.
    
    Attributes:
        status: pass or refuse
        accepted_chunks: List of chunks that passed validation
        rejected_chunks: List of chunks that failed with reasons
        refusal_reason: Machine-readable reason if refused
        refusal_message: Human-readable message if refused
    """
    status: ValidationStatus
    accepted_chunks: List[AcceptedChunk] = field(default_factory=list)
    rejected_chunks: List[RejectedChunk] = field(default_factory=list)
    refusal_reason: Optional[RefusalReason] = None
    refusal_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "accepted_chunks": [asdict(c) for c in self.accepted_chunks],
            "rejected_chunks": [asdict(c) for c in self.rejected_chunks],
            "refusal_reason": self.refusal_reason.value if self.refusal_reason else None,
            "refusal_message": self.refusal_message,
        }


class RetrievalValidator:
    """
    Main validation pipeline for retrieved chunks.
    
    Validates:
    1. Statute consistency (IPC vs CrPC etc.)
    2. Section consistency
    3. Repealed/invalid law guards
    4. Evidence threshold
    
    If validation fails → structured refusal, never guess.
    """
    
    def __init__(
        self,
        threshold_config: Optional[ThresholdConfig] = None,
        strict_statute: bool = True,
        strict_section: bool = True,
        log_dir: str = "logs",
    ):
        """
        Initialize validator.
        
        Args:
            threshold_config: Evidence threshold configuration
            strict_statute: Reject statute mismatches (vs penalize)
            strict_section: Require exact section match
            log_dir: Directory for validation logs
        """
        self.threshold = EvidenceThreshold(threshold_config)
        self.statute_validator = StatuteValidator(strict_mode=strict_statute)
        self.evidence_filter = EvidenceFilter(strict_section_match=strict_section)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rag_validation.jsonl"
    
    def validate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunk dictionaries
                Expected keys: chunk_id, text, section, act, score
            
        Returns:
            ValidationResult with pass/refuse status
        """
        accepted = []
        rejected = []
        
        for chunk in retrieved_chunks:
            chunk_id = chunk.get('chunk_id', '')
            text = chunk.get('text', '')
            section = chunk.get('section')
            act = chunk.get('act')
            score = chunk.get('score', 0.0)
            metadata = chunk.get('metadata', {})
            
            # Step 1: Statute validation
            statute_result = self.statute_validator.validate(query, act, text)
            
            if not statute_result.is_match and statute_result.penalty == 0:
                rejected.append(RejectedChunk(
                    chunk_id=chunk_id,
                    reason=RefusalReason.STATUTE_MISMATCH.value,
                    details=f"Query statute: {statute_result.query_statute}, Chunk statute: {statute_result.chunk_statute}",
                ))
                continue
            
            # Step 2: Evidence filtering (section consistency + law validity)
            filter_result = self.evidence_filter.filter_chunk(
                query=query,
                chunk_id=chunk_id,
                chunk_section=section,
                chunk_text=text,
                chunk_metadata=metadata,
            )
            
            if not filter_result.accepted:
                rejected.append(RejectedChunk(
                    chunk_id=chunk_id,
                    reason=filter_result.reason.value,
                    details=filter_result.details,
                ))
                continue
            
            # Chunk passed validation - apply statute penalty to score
            adjusted_score = score * statute_result.penalty
            
            accepted.append(AcceptedChunk(
                chunk_id=chunk_id,
                text=text,
                section=section,
                act=act,
                score=score,
                adjusted_score=adjusted_score,
            ))
        
        # Step 3: Check evidence threshold
        if not accepted:
            result = ValidationResult(
                status=ValidationStatus.REFUSE,
                accepted_chunks=[],
                rejected_chunks=rejected,
                refusal_reason=RefusalReason.NO_VALID_CHUNKS,
                refusal_message="No chunks passed validation filters.",
            )
            self._log_validation(query, retrieved_chunks, result)
            return result
        
        # Check threshold with adjusted scores
        adjusted_scores = [c.adjusted_score for c in accepted]
        threshold_result = self.threshold.check(adjusted_scores)
        
        if threshold_result.status == ThresholdStatus.FAIL:
            refusal_reason = RefusalReason.INSUFFICIENT_EVIDENCE
            if threshold_result.reason == "low_confidence":
                refusal_reason = RefusalReason.LOW_CONFIDENCE
            
            result = ValidationResult(
                status=ValidationStatus.REFUSE,
                accepted_chunks=accepted,
                rejected_chunks=rejected,
                refusal_reason=refusal_reason,
                refusal_message=f"Evidence threshold not met: {threshold_result.reason}. "
                               f"Found {threshold_result.chunk_count} chunks with max confidence {threshold_result.max_confidence:.2f}.",
            )
            self._log_validation(query, retrieved_chunks, result)
            return result
        
        # Sort accepted chunks by adjusted score
        accepted.sort(key=lambda x: x.adjusted_score, reverse=True)
        
        result = ValidationResult(
            status=ValidationStatus.PASS,
            accepted_chunks=accepted,
            rejected_chunks=rejected,
        )
        self._log_validation(query, retrieved_chunks, result)
        return result
    
    def _log_validation(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        result: ValidationResult,
    ) -> None:
        """
        Log validation run to JSONL file.
        
        Args:
            query: User query
            retrieved_chunks: Original retrieved chunks
            result: Validation result
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "total_retrieved": len(retrieved_chunks),
            "accepted_count": len(result.accepted_chunks),
            "rejected_count": len(result.rejected_chunks),
            "status": result.status.value,
            "refusal_reason": result.refusal_reason.value if result.refusal_reason else None,
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            # Don't fail validation due to logging error
            print(f"Warning: Failed to log validation: {e}")
    
    def get_refusal_response(self, result: ValidationResult) -> Dict[str, Any]:
        """
        Get structured refusal response.
        
        Args:
            result: Validation result
            
        Returns:
            Structured refusal dictionary
        """
        if result.status == ValidationStatus.PASS:
            return {"status": "pass"}
        
        return {
            "status": "refuse",
            "reason": result.refusal_reason.value if result.refusal_reason else "unknown",
            "message": result.refusal_message,
            "rejected_count": len(result.rejected_chunks),
            "rejection_details": [
                {"chunk_id": r.chunk_id, "reason": r.reason}
                for r in result.rejected_chunks[:5]  # Limit details
            ],
        }
