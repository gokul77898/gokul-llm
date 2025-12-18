"""
Evidence Threshold Definitions

Phase R3: Retrieval Validation & Filtering

Defines minimum evidence requirements for retrieval results.
If threshold not met → REFUSE.

NO LLMs used in this module.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ThresholdStatus(str, Enum):
    """Status of threshold check."""
    PASS = "pass"
    FAIL = "fail"


@dataclass
class ThresholdConfig:
    """
    Configuration for evidence thresholds.
    
    Attributes:
        min_chunks: Minimum number of chunks required
        min_confidence: Minimum confidence score for single chunk
        high_confidence_threshold: Score above which single chunk is sufficient
    """
    min_chunks: int = 2
    min_confidence: float = 0.5
    high_confidence_threshold: float = 0.75


@dataclass
class ThresholdResult:
    """Result of threshold check."""
    status: ThresholdStatus
    chunk_count: int
    max_confidence: float
    reason: Optional[str] = None


class EvidenceThreshold:
    """
    Evidence threshold checker.
    
    Rules:
    - At least 2 chunks OR
    - One chunk with confidence ≥ 0.75
    
    If threshold not met → REFUSE
    """
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize threshold checker.
        
        Args:
            config: Threshold configuration
        """
        self.config = config or ThresholdConfig()
    
    def check(self, scores: List[float]) -> ThresholdResult:
        """
        Check if evidence meets threshold.
        
        Args:
            scores: List of confidence scores for chunks
            
        Returns:
            ThresholdResult with pass/fail status
        """
        if not scores:
            return ThresholdResult(
                status=ThresholdStatus.FAIL,
                chunk_count=0,
                max_confidence=0.0,
                reason="no_evidence",
            )
        
        chunk_count = len(scores)
        max_confidence = max(scores)
        
        # Rule 1: At least min_chunks
        if chunk_count >= self.config.min_chunks:
            # Check minimum confidence for any chunk
            if max_confidence >= self.config.min_confidence:
                return ThresholdResult(
                    status=ThresholdStatus.PASS,
                    chunk_count=chunk_count,
                    max_confidence=max_confidence,
                )
            else:
                return ThresholdResult(
                    status=ThresholdStatus.FAIL,
                    chunk_count=chunk_count,
                    max_confidence=max_confidence,
                    reason="low_confidence",
                )
        
        # Rule 2: Single chunk with high confidence
        if chunk_count == 1 and max_confidence >= self.config.high_confidence_threshold:
            return ThresholdResult(
                status=ThresholdStatus.PASS,
                chunk_count=chunk_count,
                max_confidence=max_confidence,
            )
        
        # Threshold not met
        if chunk_count < self.config.min_chunks:
            return ThresholdResult(
                status=ThresholdStatus.FAIL,
                chunk_count=chunk_count,
                max_confidence=max_confidence,
                reason="insufficient_evidence",
            )
        
        return ThresholdResult(
            status=ThresholdStatus.FAIL,
            chunk_count=chunk_count,
            max_confidence=max_confidence,
            reason="low_confidence",
        )
    
    def meets_threshold(self, scores: List[float]) -> bool:
        """
        Quick check if evidence meets threshold.
        
        Args:
            scores: List of confidence scores
            
        Returns:
            True if threshold met
        """
        return self.check(scores).status == ThresholdStatus.PASS
