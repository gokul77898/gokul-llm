"""
Citation Formatting Rules

Phase R4: Context Assembler

Formats citations for evidence blocks.
Enforces strict citation requirements.

NO LLMs used in this module.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SourceType(str, Enum):
    """Source type for citations."""
    BARE_ACT = "Bare Act"
    CASE_LAW = "Case Law"
    AMENDMENT = "Amendment"
    NOTIFICATION = "Notification"
    UNKNOWN = "Unknown"


@dataclass
class Citation:
    """
    Formatted citation for an evidence block.
    
    Attributes:
        act_or_court: Act name (for bare acts) or Court name (for case law)
        section: Section number (if applicable)
        year: Year of the law/judgment
        source_type: Type of source
        case_name: Case name (for case law only)
        is_valid: Whether citation has all required fields
        missing_fields: List of missing required fields
    """
    act_or_court: Optional[str]
    section: Optional[str]
    year: Optional[int]
    source_type: SourceType
    case_name: Optional[str] = None
    is_valid: bool = True
    missing_fields: list = None
    
    def __post_init__(self):
        if self.missing_fields is None:
            self.missing_fields = []


class CitationFormatter:
    """
    Formats citations for evidence blocks.
    
    Citation Rules:
    - Each chunk MUST include: Act/Court, Section (if applicable), Year, Source type
    - If any required metadata missing → chunk is dropped
    
    Format for Bare Acts:
        (IPC, Section 420, 1860)
        
    Format for Case Law:
        (Supreme Court, 2012, XYZ v State)
    """
    
    # Court hierarchy for ordering
    COURT_HIERARCHY = {
        "supreme court": 1,
        "sc": 1,
        "high court": 2,
        "hc": 2,
        "district court": 3,
        "sessions court": 3,
        "magistrate": 4,
        "tribunal": 5,
    }
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize citation formatter.
        
        Args:
            strict_mode: If True, drop chunks with missing required fields
        """
        self.strict_mode = strict_mode
    
    def format_citation(
        self,
        act: Optional[str],
        section: Optional[str],
        year: Optional[int],
        doc_type: str,
        court: Optional[str] = None,
        citation: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Citation:
        """
        Format a citation from chunk metadata.
        
        Args:
            act: Act name
            section: Section number
            year: Year
            doc_type: Document type (bare_act, case_law, etc.)
            court: Court name (for case law)
            citation: Legal citation string
            title: Document title (for case name extraction)
            
        Returns:
            Citation object
        """
        missing_fields = []
        source_type = self._get_source_type(doc_type)
        
        if source_type == SourceType.BARE_ACT:
            return self._format_bare_act_citation(act, section, year, missing_fields)
        elif source_type == SourceType.CASE_LAW:
            return self._format_case_law_citation(court, year, title, citation, missing_fields)
        else:
            return self._format_generic_citation(act, section, year, source_type, missing_fields)
    
    def _format_bare_act_citation(
        self,
        act: Optional[str],
        section: Optional[str],
        year: Optional[int],
        missing_fields: list,
    ) -> Citation:
        """Format citation for bare act."""
        # Required: act, year
        if not act:
            missing_fields.append("act")
        if not year:
            missing_fields.append("year")
        
        is_valid = len(missing_fields) == 0
        
        return Citation(
            act_or_court=act,
            section=section,
            year=year,
            source_type=SourceType.BARE_ACT,
            is_valid=is_valid,
            missing_fields=missing_fields,
        )
    
    def _format_case_law_citation(
        self,
        court: Optional[str],
        year: Optional[int],
        title: Optional[str],
        citation: Optional[str],
        missing_fields: list,
    ) -> Citation:
        """Format citation for case law."""
        # Required: court, year
        if not court:
            missing_fields.append("court")
        if not year:
            missing_fields.append("year")
        
        # Extract case name from title or citation
        case_name = self._extract_case_name(title, citation)
        
        is_valid = len(missing_fields) == 0
        
        return Citation(
            act_or_court=court,
            section=None,
            year=year,
            source_type=SourceType.CASE_LAW,
            case_name=case_name,
            is_valid=is_valid,
            missing_fields=missing_fields,
        )
    
    def _format_generic_citation(
        self,
        act: Optional[str],
        section: Optional[str],
        year: Optional[int],
        source_type: SourceType,
        missing_fields: list,
    ) -> Citation:
        """Format citation for other document types."""
        if not act:
            missing_fields.append("act")
        if not year:
            missing_fields.append("year")
        
        is_valid = len(missing_fields) == 0
        
        return Citation(
            act_or_court=act,
            section=section,
            year=year,
            source_type=source_type,
            is_valid=is_valid,
            missing_fields=missing_fields,
        )
    
    def _get_source_type(self, doc_type: str) -> SourceType:
        """Convert doc_type string to SourceType enum."""
        doc_type_lower = str(doc_type).lower()
        
        if "bare" in doc_type_lower or "act" in doc_type_lower:
            return SourceType.BARE_ACT
        elif "case" in doc_type_lower or "judgment" in doc_type_lower:
            return SourceType.CASE_LAW
        elif "amendment" in doc_type_lower:
            return SourceType.AMENDMENT
        elif "notification" in doc_type_lower:
            return SourceType.NOTIFICATION
        else:
            return SourceType.UNKNOWN
    
    def _extract_case_name(
        self,
        title: Optional[str],
        citation: Optional[str],
    ) -> Optional[str]:
        """Extract case name from title or citation."""
        if title and " v " in title.lower():
            return title
        if title and " vs " in title.lower():
            return title
        if citation:
            return citation
        return title
    
    def format_citation_line(self, citation: Citation) -> str:
        """
        Format citation as a single line for evidence block.
        
        Args:
            citation: Citation object
            
        Returns:
            Formatted citation string
        """
        if citation.source_type == SourceType.CASE_LAW:
            # Format: (Supreme Court, 2012, XYZ v State)
            parts = []
            if citation.act_or_court:
                parts.append(citation.act_or_court)
            if citation.year:
                parts.append(str(citation.year))
            if citation.case_name:
                parts.append(citation.case_name)
            return f"({', '.join(parts)})"
        else:
            # Format: (IPC, Section 420, 1860)
            parts = []
            if citation.act_or_court:
                parts.append(citation.act_or_court)
            if citation.section:
                parts.append(f"Section {citation.section}")
            if citation.year:
                parts.append(str(citation.year))
            return f"({', '.join(parts)})"
    
    def get_court_rank(self, court: Optional[str]) -> int:
        """
        Get court hierarchy rank (lower = higher court).
        
        Args:
            court: Court name
            
        Returns:
            Rank (1 = Supreme Court, higher = lower courts)
        """
        if not court:
            return 99
        
        court_lower = court.lower()
        
        for court_name, rank in self.COURT_HIERARCHY.items():
            if court_name in court_lower:
                return rank
        
        return 10  # Default for unknown courts
