"""
Phase-7: Observability & Contracts - Audit Records

Audit records for all RAG operations with refusal tracking.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional


logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Audit record for RAG operation.
    
    Tracks all operations including refusals for compliance and debugging.
    
    Attributes:
        query: User query text
        retrieved_semantic_ids: List of retrieved chunk semantic IDs
        cited_ids: List of cited semantic IDs in answer
        refusal_reason: Reason for refusal (None if answer provided)
        phase: Phase identifier (C2, C3, C3+)
        timestamp: UTC timestamp of operation
        is_grounded: Whether answer is grounded
        is_sufficient: Whether evidence is sufficient (C3+ only)
        invalid_citations: List of invalid citations (if any)
        uncovered_claims: List of uncovered claims (if any)
    """
    
    query: str
    retrieved_semantic_ids: List[str]
    cited_ids: List[str]
    refusal_reason: Optional[str]
    phase: str
    timestamp: str
    is_grounded: bool
    is_sufficient: Optional[bool] = None
    invalid_citations: Optional[List[str]] = None
    uncovered_claims: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate audit record."""
        if self.phase not in ['C2', 'C3', 'C3+']:
            raise ValueError(f"Invalid phase: {self.phase}. Must be C2, C3, or C3+")
        
        # If refusal, ensure reason is provided
        if self.refusal_reason and not self.is_grounded:
            pass  # Valid refusal
        elif not self.refusal_reason and not self.is_grounded:
            raise ValueError("Ungrounded answer must have refusal_reason")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @property
    def is_refusal(self) -> bool:
        """Check if this is a refusal."""
        return self.refusal_reason is not None
    
    @property
    def citation_count(self) -> int:
        """Get number of citations."""
        return len(self.cited_ids)
    
    @property
    def retrieval_count(self) -> int:
        """Get number of retrieved chunks."""
        return len(self.retrieved_semantic_ids)


def create_audit_record(
    query: str,
    retrieved_semantic_ids: List[str],
    cited_ids: List[str],
    refusal_reason: Optional[str],
    phase: str,
    is_grounded: bool,
    is_sufficient: Optional[bool] = None,
    invalid_citations: Optional[List[str]] = None,
    uncovered_claims: Optional[List[str]] = None
) -> AuditRecord:
    """Create audit record with current timestamp.
    
    Args:
        query: User query
        retrieved_semantic_ids: Retrieved semantic IDs
        cited_ids: Cited semantic IDs
        refusal_reason: Refusal reason (None if answer provided)
        phase: Phase identifier (C2, C3, C3+)
        is_grounded: Grounding status
        is_sufficient: Evidence sufficiency (C3+ only)
        invalid_citations: Invalid citations
        uncovered_claims: Uncovered claims
    
    Returns:
        AuditRecord instance
    """
    return AuditRecord(
        query=query,
        retrieved_semantic_ids=retrieved_semantic_ids,
        cited_ids=cited_ids,
        refusal_reason=refusal_reason,
        phase=phase,
        timestamp=datetime.utcnow().isoformat(),
        is_grounded=is_grounded,
        is_sufficient=is_sufficient,
        invalid_citations=invalid_citations,
        uncovered_claims=uncovered_claims
    )


def log_audit_record(record: AuditRecord) -> None:
    """Log audit record to structured log.
    
    Args:
        record: AuditRecord to log
    """
    log_entry = {
        "event_type": "audit_record",
        **record.to_dict()
    }
    
    logger.info(json.dumps(log_entry))
    
    # Also log human-readable summary
    if record.is_refusal:
        logger.warning(
            f"AUDIT: Refusal - phase={record.phase}, "
            f"reason={record.refusal_reason}, "
            f"retrieved={record.retrieval_count}"
        )
    else:
        logger.info(
            f"AUDIT: Success - phase={record.phase}, "
            f"citations={record.citation_count}, "
            f"retrieved={record.retrieval_count}"
        )


def print_audit_record(record: AuditRecord, verbose: bool = False) -> None:
    """Print audit record in human-readable format.
    
    Args:
        record: AuditRecord to print
        verbose: Show detailed information
    """
    print("=" * 70)
    print("AUDIT RECORD")
    print("=" * 70)
    
    print(f"\nPhase: {record.phase}")
    print(f"Timestamp: {record.timestamp}")
    print(f"Query: {record.query}")
    
    print(f"\nRetrieval:")
    print(f"  Retrieved: {record.retrieval_count} chunks")
    if verbose and record.retrieved_semantic_ids:
        for sid in record.retrieved_semantic_ids:
            print(f"    - {sid}")
    
    print(f"\nGeneration:")
    print(f"  Grounded: {'✓' if record.is_grounded else '✗'}")
    
    if record.is_sufficient is not None:
        print(f"  Sufficient: {'✓' if record.is_sufficient else '✗'}")
    
    print(f"  Citations: {record.citation_count}")
    if verbose and record.cited_ids:
        for cid in record.cited_ids:
            print(f"    - {cid}")
    
    if record.is_refusal:
        print(f"\n⚠ REFUSAL")
        print(f"  Reason: {record.refusal_reason}")
    
    if record.invalid_citations:
        print(f"\n✗ Invalid Citations: {len(record.invalid_citations)}")
        if verbose:
            for cid in record.invalid_citations:
                print(f"    - {cid}")
    
    if record.uncovered_claims:
        print(f"\n✗ Uncovered Claims: {len(record.uncovered_claims)}")
        if verbose:
            for claim in record.uncovered_claims:
                print(f"    - {claim[:60]}...")
    
    print()
