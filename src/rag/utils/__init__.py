"""RAG Utilities - Text processing helpers"""

from .text_cleaning import (
    clean_text,
    remove_headers_footers,
    normalize_whitespace,
    remove_page_numbers,
)

__all__ = [
    "clean_text",
    "remove_headers_footers",
    "normalize_whitespace",
    "remove_page_numbers",
]
