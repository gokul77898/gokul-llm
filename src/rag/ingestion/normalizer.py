"""
Text Normalizer for Legal Documents

Phase R0: RAG Foundations

Normalizes legal text to canonical forms:
- Section references
- Act names
- Court names
- Citation formats
"""

import re
from typing import Dict, List, Tuple


# Canonical act name mappings
ACT_NAME_MAPPINGS: Dict[str, str] = {
    # Indian Penal Code
    "indian penal code": "IPC",
    "ipc": "IPC",
    "i.p.c.": "IPC",
    "i.p.c": "IPC",
    
    # Code of Criminal Procedure
    "code of criminal procedure": "CrPC",
    "criminal procedure code": "CrPC",
    "crpc": "CrPC",
    "cr.p.c.": "CrPC",
    "cr.p.c": "CrPC",
    
    # Code of Civil Procedure
    "code of civil procedure": "CPC",
    "civil procedure code": "CPC",
    "cpc": "CPC",
    "c.p.c.": "CPC",
    "c.p.c": "CPC",
    
    # Indian Evidence Act
    "indian evidence act": "IEA",
    "evidence act": "IEA",
    "iea": "IEA",
    
    # Constitution
    "constitution of india": "Constitution",
    "indian constitution": "Constitution",
    
    # Companies Act
    "companies act": "Companies Act",
    
    # Income Tax Act
    "income tax act": "IT Act",
    "income-tax act": "IT Act",
    
    # Minimum Wages Act
    "minimum wages act": "Minimum Wages Act",
    
    # Contract Act
    "indian contract act": "Contract Act",
    "contract act": "Contract Act",
}

# Section reference patterns
SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r'\bSec\.?\s*(\d+)', r'Section \1'),
    (r'\bsec\.?\s*(\d+)', r'Section \1'),
    (r'\bS\.?\s*(\d+)', r'Section \1'),
    (r'\bs\.?\s*(\d+)', r'Section \1'),
    (r'\bSection\s+(\d+)', r'Section \1'),
    (r'\bsection\s+(\d+)', r'Section \1'),
    (r'\bSECTION\s+(\d+)', r'Section \1'),
    (r'\b§\s*(\d+)', r'Section \1'),
]


def normalize_text(text: str) -> str:
    """
    Apply all normalization rules to legal text.
    
    Args:
        text: Raw legal text
        
    Returns:
        Normalized text
    """
    text = normalize_section_refs(text)
    text = normalize_act_names(text)
    return text


def normalize_section_refs(text: str) -> str:
    """
    Normalize section references to canonical form.
    
    Converts:
    - "Sec. 420" → "Section 420"
    - "section 420" → "Section 420"
    - "S.420" → "Section 420"
    - "§ 420" → "Section 420"
    
    Args:
        text: Legal text with section references
        
    Returns:
        Text with normalized section references
    """
    for pattern, replacement in SECTION_PATTERNS:
        text = re.sub(pattern, replacement, text)
    
    # Handle subsections: Section 420(1)(a)
    # Normalize spacing around subsection markers
    text = re.sub(r'Section\s+(\d+)\s*\(\s*(\d+)\s*\)', r'Section \1(\2)', text)
    text = re.sub(r'Section\s+(\d+)\s*\(\s*([a-z])\s*\)', r'Section \1(\2)', text)
    
    return text


def normalize_act_names(text: str) -> str:
    """
    Normalize act names to canonical abbreviations.
    
    Converts:
    - "Indian Penal Code" → "IPC"
    - "Code of Criminal Procedure" → "CrPC"
    
    Args:
        text: Legal text with act names
        
    Returns:
        Text with normalized act names
    """
    # Sort by length (longest first) to avoid partial matches
    sorted_mappings = sorted(
        ACT_NAME_MAPPINGS.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    for original, canonical in sorted_mappings:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        text = pattern.sub(canonical, text)
    
    return text


def normalize_court_names(text: str) -> str:
    """
    Normalize court names to canonical forms.
    
    Args:
        text: Legal text with court names
        
    Returns:
        Text with normalized court names
    """
    court_mappings = {
        "supreme court of india": "Supreme Court",
        "hon'ble supreme court": "Supreme Court",
        "hon'ble high court": "High Court",
        "district court": "District Court",
        "sessions court": "Sessions Court",
    }
    
    for original, canonical in court_mappings.items():
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        text = pattern.sub(canonical, text)
    
    return text


def extract_act_name(text: str) -> str | None:
    """
    Extract the primary act name from document text.
    
    Args:
        text: Document text
        
    Returns:
        Extracted act name or None
    """
    # Look for common patterns
    patterns = [
        r'(?:THE\s+)?([A-Z][A-Za-z\s]+(?:ACT|CODE)),?\s*(?:19|20)\d{2}',
        r'(?:under|of)\s+(?:the\s+)?([A-Z][A-Za-z\s]+(?:Act|Code))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            act_name = match.group(1).strip()
            # Normalize if known
            act_lower = act_name.lower()
            if act_lower in ACT_NAME_MAPPINGS:
                return ACT_NAME_MAPPINGS[act_lower]
            return act_name
    
    return None


def extract_year(text: str) -> int | None:
    """
    Extract the year from document text.
    
    Args:
        text: Document text
        
    Returns:
        Extracted year or None
    """
    # Look for year patterns
    patterns = [
        r'(?:ACT|Act|CODE|Code),?\s*((?:19|20)\d{2})',
        r'(?:of|dated)\s+((?:19|20)\d{2})',
        r'\b((?:19|20)\d{2})\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return None
