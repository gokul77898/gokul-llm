"""
Text Cleaning Utilities

Phase R0: RAG Foundations

Pure text processing functions for cleaning legal documents.
No embeddings, no LLMs, no external dependencies.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Apply all cleaning operations to text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    text = remove_headers_footers(text)
    text = remove_page_numbers(text)
    text = normalize_whitespace(text)
    return text.strip()


def remove_headers_footers(text: str) -> str:
    """
    Remove common header and footer patterns from legal documents.
    
    Patterns removed:
    - Page headers with document titles
    - Footer disclaimers
    - Copyright notices
    - "Page X of Y" patterns
    
    Args:
        text: Raw text
        
    Returns:
        Text with headers/footers removed
    """
    lines = text.split('\n')
    cleaned_lines: List[str] = []
    
    # Common header/footer patterns
    header_patterns = [
        r'^[-_=]{10,}$',  # Separator lines
        r'^\s*Page\s+\d+\s*$',
        r'^\s*-\s*\d+\s*-\s*$',
        r'^\s*\[\s*\d+\s*\]\s*$',
        r'^\s*©.*\d{4}.*$',  # Copyright
        r'^\s*All\s+[Rr]ights\s+[Rr]eserved.*$',
        r'^\s*DISCLAIMER.*$',
        r'^\s*This\s+document\s+is\s+for\s+informational.*$',
    ]
    
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in header_patterns]
    
    for line in lines:
        is_header_footer = False
        for pattern in compiled_patterns:
            if pattern.match(line):
                is_header_footer = True
                break
        
        if not is_header_footer:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_page_numbers(text: str) -> str:
    """
    Remove page number patterns from text.
    
    Patterns removed:
    - "Page 1", "Page 1 of 10"
    - "[1]", "(1)"
    - Standalone numbers at line boundaries
    
    Args:
        text: Raw text
        
    Returns:
        Text with page numbers removed
    """
    # Remove "Page X" or "Page X of Y" patterns
    text = re.sub(r'\bPage\s+\d+(\s+of\s+\d+)?\b', '', text, flags=re.IGNORECASE)
    
    # Remove standalone page numbers in brackets
    text = re.sub(r'^\s*[\[\(]\s*\d+\s*[\]\)]\s*$', '', text, flags=re.MULTILINE)
    
    # Remove standalone numbers that are likely page numbers (at start/end of lines)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    
    Operations:
    - Replace multiple spaces with single space
    - Normalize line endings
    - Preserve paragraph breaks (double newlines)
    - Remove trailing whitespace
    
    Args:
        text: Raw text
        
    Returns:
        Text with normalized whitespace
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple spaces with single space (within lines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    # Normalize multiple newlines to max 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from lines
    text = re.sub(r' +\n', '\n', text)
    
    # Remove leading whitespace from lines (but preserve indentation structure)
    # Only remove excessive leading spaces
    text = re.sub(r'\n {4,}', '\n    ', text)
    
    return text


def extract_sections(text: str) -> List[str]:
    """
    Extract section boundaries from legal text.
    
    This is a helper for future chunking (Phase R1).
    Currently returns the full text as a single section.
    
    Args:
        text: Cleaned legal text
        
    Returns:
        List of section texts
    """
    # Phase R0: No chunking - return full text
    return [text]
