"""
Evidence Filter

Phase R3: Retrieval Validation & Filtering

Filters retrieved chunks based on:
- Section consistency
- Repealed/invalid law guards

NO LLMs used in this module.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum


class FilterReason(str, Enum):
    """Reasons for filtering a chunk."""
    SECTION_MISMATCH = "section_mismatch"
    REPEALED_LAW = "repealed_law"
    INVALID_LAW = "invalid_law"
    SUPERSEDED_LAW = "superseded_law"
    ACCEPTED = "accepted"


@dataclass
class FilterResult:
    """Result of filtering a single chunk."""
    chunk_id: str
    accepted: bool
    reason: FilterReason
    details: Optional[str] = None


@dataclass
class FilterSummary:
    """Summary of filtering multiple chunks."""
    total: int
    accepted: int
    rejected: int
    results: List[FilterResult] = field(default_factory=list)
    
    @property
    def accepted_ids(self) -> List[str]:
        """Get list of accepted chunk IDs."""
        return [r.chunk_id for r in self.results if r.accepted]
    
    @property
    def rejected_ids(self) -> List[str]:
        """Get list of rejected chunk IDs."""
        return [r.chunk_id for r in self.results if not r.accepted]


class EvidenceFilter:
    """
    Filters evidence based on section consistency and law validity.
    
    Rules:
    1. Section Consistency:
       - If query mentions a section number, chunk MUST contain same section
       - Otherwise chunk is discarded
    
    2. Repealed/Invalid Law Guard:
       - Reject chunks marked: repealed=true, invalid=true, superseded=true
    """
    
    def __init__(self, strict_section_match: bool = True):
        """
        Initialize evidence filter.
        
        Args:
            strict_section_match: If True, require exact section match
        """
        self.strict_section_match = strict_section_match
    
    def extract_sections(self, text: str) -> Set[str]:
        """
        Extract section numbers from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of section numbers (as strings)
        """
        # Match various section formats
        patterns = [
            r'section\s+(\d+[a-z]?)',  # Section 420, Section 420A
            r'sec\.?\s*(\d+[a-z]?)',   # Sec. 420, Sec 420
            r's\.?\s*(\d+[a-z]?)',     # S. 420, S 420
        ]
        
        sections = set()
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            sections.update(matches)
        
        return sections
    
    def check_section_consistency(
        self,
        query: str,
        chunk_section: Optional[str],
        chunk_text: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if chunk section matches query section.
        
        Args:
            query: User query
            chunk_section: Section from chunk metadata
            chunk_text: Chunk text content
            
        Returns:
            Tuple of (is_consistent, detail_message)
        """
        # Extract sections from query
        query_sections = self.extract_sections(query)
        
        # If query doesn't mention specific sections, accept all
        if not query_sections:
            return True, None
        
        # Get chunk sections from metadata and text
        chunk_sections = set()
        if chunk_section:
            # Normalize section from metadata
            section_num = re.search(r'(\d+[a-z]?)', str(chunk_section).lower())
            if section_num:
                chunk_sections.add(section_num.group(1))
        
        # Also extract from text
        chunk_sections.update(self.extract_sections(chunk_text))
        
        # Check for overlap
        if self.strict_section_match:
            # Require exact match
            if query_sections & chunk_sections:
                return True, None
            else:
                return False, f"Query sections {query_sections} not in chunk sections {chunk_sections}"
        else:
            # Allow if any section found
            if chunk_sections:
                return True, None
            return False, "No sections found in chunk"
    
    def check_law_validity(
        self,
        chunk_metadata: dict,
    ) -> tuple[bool, Optional[FilterReason]]:
        """
        Check if law is valid (not repealed/invalid/superseded).
        
        Args:
            chunk_metadata: Chunk metadata dictionary
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check for repealed flag
        if chunk_metadata.get('repealed', False):
            return False, FilterReason.REPEALED_LAW
        
        # Check for invalid flag
        if chunk_metadata.get('invalid', False):
            return False, FilterReason.INVALID_LAW
        
        # Check for superseded flag
        if chunk_metadata.get('superseded', False):
            return False, FilterReason.SUPERSEDED_LAW
        
        return True, None
    
    def filter_chunk(
        self,
        query: str,
        chunk_id: str,
        chunk_section: Optional[str],
        chunk_text: str,
        chunk_metadata: Optional[dict] = None,
    ) -> FilterResult:
        """
        Filter a single chunk.
        
        Args:
            query: User query
            chunk_id: Chunk identifier
            chunk_section: Section from chunk metadata
            chunk_text: Chunk text content
            chunk_metadata: Additional chunk metadata
            
        Returns:
            FilterResult with accept/reject decision
        """
        metadata = chunk_metadata or {}
        
        # Check law validity first
        is_valid, invalid_reason = self.check_law_validity(metadata)
        if not is_valid:
            return FilterResult(
                chunk_id=chunk_id,
                accepted=False,
                reason=invalid_reason,
                details="Law is no longer valid",
            )
        
        # Check section consistency
        is_consistent, detail = self.check_section_consistency(
            query, chunk_section, chunk_text
        )
        if not is_consistent:
            return FilterResult(
                chunk_id=chunk_id,
                accepted=False,
                reason=FilterReason.SECTION_MISMATCH,
                details=detail,
            )
        
        # Chunk passes all filters
        return FilterResult(
            chunk_id=chunk_id,
            accepted=True,
            reason=FilterReason.ACCEPTED,
        )
    
    def filter_chunks(
        self,
        query: str,
        chunks: List[dict],
    ) -> FilterSummary:
        """
        Filter multiple chunks.
        
        Args:
            query: User query
            chunks: List of chunk dictionaries with keys:
                    chunk_id, section, text, and optional metadata
            
        Returns:
            FilterSummary with all results
        """
        results = []
        
        for chunk in chunks:
            result = self.filter_chunk(
                query=query,
                chunk_id=chunk.get('chunk_id', ''),
                chunk_section=chunk.get('section'),
                chunk_text=chunk.get('text', ''),
                chunk_metadata=chunk.get('metadata', {}),
            )
            results.append(result)
        
        accepted = sum(1 for r in results if r.accepted)
        
        return FilterSummary(
            total=len(chunks),
            accepted=accepted,
            rejected=len(chunks) - accepted,
            results=results,
        )
