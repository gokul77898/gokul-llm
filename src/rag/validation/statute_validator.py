"""
Statute Validation Rules

Phase R3: Retrieval Validation & Filtering

Validates that retrieved chunks match the statute mentioned in query.
Prevents cross-statute contamination (e.g., IPC chunks for CrPC query).

NO LLMs used in this module.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from enum import Enum


class StatuteType(str, Enum):
    """Known statute types."""
    IPC = "ipc"  # Indian Penal Code
    CRPC = "crpc"  # Code of Criminal Procedure
    CPC = "cpc"  # Code of Civil Procedure
    IEA = "iea"  # Indian Evidence Act
    CONSTITUTION = "constitution"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class StatuteMatch:
    """Result of statute matching."""
    query_statute: Optional[StatuteType]
    chunk_statute: Optional[StatuteType]
    is_match: bool
    penalty: float  # Score penalty (1.0 = no penalty, 0.0 = reject)
    reason: Optional[str] = None


class StatuteValidator:
    """
    Validates statute consistency between query and chunks.
    
    Rules:
    - If query mentions IPC → Reject CrPC-only chunks
    - If query mentions CrPC → Reject IPC-only chunks
    - If statute cannot be inferred → Allow both, but penalize score
    """
    
    # Statute detection patterns
    STATUTE_PATTERNS = {
        StatuteType.IPC: [
            r'\bipc\b',
            r'\bindian\s+penal\s+code\b',
            r'\bpenal\s+code\b',
        ],
        StatuteType.CRPC: [
            r'\bcrpc\b',
            r'\bcr\.?p\.?c\.?\b',
            r'\bcode\s+of\s+criminal\s+procedure\b',
            r'\bcriminal\s+procedure\s+code\b',
        ],
        StatuteType.CPC: [
            r'\bcpc\b',
            r'\bc\.?p\.?c\.?\b',
            r'\bcode\s+of\s+civil\s+procedure\b',
            r'\bcivil\s+procedure\s+code\b',
        ],
        StatuteType.IEA: [
            r'\biea\b',
            r'\bindian\s+evidence\s+act\b',
            r'\bevidence\s+act\b',
        ],
        StatuteType.CONSTITUTION: [
            r'\bconstitution\b',
            r'\barticle\s+\d+',
        ],
    }
    
    # IPC-specific section ranges (common sections)
    IPC_SECTIONS = set(range(1, 512))  # IPC has sections 1-511
    
    # CrPC-specific section ranges
    CRPC_SECTIONS = set(range(1, 485))  # CrPC has sections 1-484
    
    # Well-known IPC sections
    KNOWN_IPC_SECTIONS = {
        302, 304, 306, 307, 323, 324, 354, 376, 379, 380,
        384, 392, 395, 406, 415, 420, 467, 468, 471, 498,
        499, 500, 506, 509,
    }
    
    # Well-known CrPC sections (bail, arrest, etc.)
    KNOWN_CRPC_SECTIONS = {
        41, 154, 156, 161, 164, 167, 173, 190, 197, 200,
        204, 227, 228, 239, 240, 245, 313, 319, 354, 374,
        378, 389, 397, 401, 437, 438, 439, 482,
    }
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize statute validator.
        
        Args:
            strict_mode: If True, reject mismatched statutes. If False, penalize.
        """
        self.strict_mode = strict_mode
    
    def detect_statute(self, text: str) -> Tuple[Optional[StatuteType], Set[int]]:
        """
        Detect statute type from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (statute_type, set of section numbers found)
        """
        text_lower = text.lower()
        detected_statutes = []
        
        for statute, patterns in self.STATUTE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_statutes.append(statute)
                    break
        
        # Extract section numbers
        section_matches = re.findall(r'section\s+(\d+)', text_lower)
        sections = {int(s) for s in section_matches}
        
        # If explicit statute found, return it
        if detected_statutes:
            return detected_statutes[0], sections
        
        # Infer from section numbers
        if sections:
            # Check if sections are known IPC sections
            if sections & self.KNOWN_IPC_SECTIONS:
                return StatuteType.IPC, sections
            # Check if sections are known CrPC sections
            if sections & self.KNOWN_CRPC_SECTIONS:
                return StatuteType.CRPC, sections
        
        return None, sections
    
    def detect_statute_from_act(self, act: Optional[str]) -> Optional[StatuteType]:
        """
        Detect statute type from act name.
        
        Args:
            act: Act name from chunk metadata
            
        Returns:
            Detected statute type
        """
        if not act:
            return None
        
        act_lower = act.lower()
        
        if 'ipc' in act_lower or 'penal' in act_lower:
            return StatuteType.IPC
        if 'crpc' in act_lower or 'criminal procedure' in act_lower:
            return StatuteType.CRPC
        if 'cpc' in act_lower or 'civil procedure' in act_lower:
            return StatuteType.CPC
        if 'evidence' in act_lower:
            return StatuteType.IEA
        if 'constitution' in act_lower:
            return StatuteType.CONSTITUTION
        
        return StatuteType.OTHER
    
    def validate(
        self,
        query: str,
        chunk_act: Optional[str],
        chunk_text: str,
    ) -> StatuteMatch:
        """
        Validate statute consistency.
        
        Args:
            query: User query
            chunk_act: Act name from chunk metadata
            chunk_text: Chunk text content
            
        Returns:
            StatuteMatch with validation result
        """
        # Detect query statute
        query_statute, query_sections = self.detect_statute(query)
        
        # Detect chunk statute
        chunk_statute = self.detect_statute_from_act(chunk_act)
        if not chunk_statute:
            chunk_statute, _ = self.detect_statute(chunk_text)
        
        # If query statute is unknown, allow with slight penalty
        if not query_statute:
            return StatuteMatch(
                query_statute=None,
                chunk_statute=chunk_statute,
                is_match=True,
                penalty=0.9,  # Slight penalty for ambiguity
                reason="query_statute_unknown",
            )
        
        # If chunk statute is unknown, allow with penalty
        if not chunk_statute:
            return StatuteMatch(
                query_statute=query_statute,
                chunk_statute=None,
                is_match=True,
                penalty=0.8,
                reason="chunk_statute_unknown",
            )
        
        # Check for match
        if query_statute == chunk_statute:
            return StatuteMatch(
                query_statute=query_statute,
                chunk_statute=chunk_statute,
                is_match=True,
                penalty=1.0,  # No penalty
            )
        
        # Mismatch detected
        if self.strict_mode:
            return StatuteMatch(
                query_statute=query_statute,
                chunk_statute=chunk_statute,
                is_match=False,
                penalty=0.0,  # Reject
                reason="statute_mismatch",
            )
        else:
            return StatuteMatch(
                query_statute=query_statute,
                chunk_statute=chunk_statute,
                is_match=False,
                penalty=0.3,  # Heavy penalty but not reject
                reason="statute_mismatch_penalized",
            )
    
    def is_compatible(
        self,
        query: str,
        chunk_act: Optional[str],
        chunk_text: str,
    ) -> bool:
        """
        Quick check if chunk is compatible with query statute.
        
        Args:
            query: User query
            chunk_act: Act name from chunk
            chunk_text: Chunk text
            
        Returns:
            True if compatible
        """
        result = self.validate(query, chunk_act, chunk_text)
        return result.is_match or result.penalty > 0
