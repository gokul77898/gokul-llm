"""
Context Builder for Model-Specific Context Preparation

Handles context assembly and compression for Mamba (long context)
and Transformer (short context) models.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def build_context_for_model(
    model_key: str,
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build context string optimized for specific model
    
    Args:
        model_key: "mamba" or "transformer"
        query: User query
        retrieved_docs: List of retrieved document dicts
        config: Optional config with max_tokens, etc.
        
    Returns:
        Formatted context string
    """
    config = config or {}
    
    if model_key == "mamba":
        # Long context for Mamba (SSM can handle it)
        max_tokens = config.get("max_context_tokens", 8192)
        return _build_long_context(retrieved_docs, max_tokens)
    
    elif model_key == "transformer":
        # Short context for Transformer
        max_tokens = config.get("max_context_tokens", 1024)
        top_k = config.get("top_k_docs", 3)
        return _build_short_context(retrieved_docs, max_tokens, top_k)
    
    else:
        # Default: medium context
        max_tokens = config.get("max_context_tokens", 2048)
        return _build_long_context(retrieved_docs, max_tokens)


def _build_long_context(
    docs: List[Dict[str, Any]],
    max_tokens: int
) -> str:
    """
    Build long context for Mamba
    
    Concatenates documents in order, optionally with page metadata.
    If exceeds max_tokens, compresses intelligently.
    
    Args:
        docs: Retrieved documents
        max_tokens: Maximum token budget
        
    Returns:
        Concatenated context string
    """
    context_parts = []
    total_chars = 0
    
    # Estimate: ~4 chars per token on average
    max_chars = max_tokens * 4
    
    for i, doc in enumerate(docs):
        # Extract document content
        content = doc.get("content") or doc.get("text") or str(doc)
        metadata = doc.get("metadata", {})
        
        # Add document header with metadata
        page_info = ""
        if "page" in metadata:
            page_info = f" (Page {metadata['page']})"
        elif "page_number" in metadata:
            page_info = f" (Page {metadata['page_number']})"
        
        doc_header = f"\n\n--- Document {i+1}{page_info} ---\n"
        
        # Check if adding this doc exceeds budget
        if total_chars + len(doc_header) + len(content) > max_chars:
            logger.info(f"Context budget exceeded at doc {i+1}, compressing...")
            # Add remaining docs in compressed form
            remaining_docs = docs[i:]
            compressed = compress_long_context(remaining_docs, max_tokens - (total_chars // 4))
            if compressed:
                context_parts.append("\n\n--- Compressed Additional Documents ---\n")
                context_parts.append(compressed)
            break
        
        context_parts.append(doc_header)
        context_parts.append(content)
        total_chars += len(doc_header) + len(content)
    
    return "".join(context_parts)


def _build_short_context(
    docs: List[Dict[str, Any]],
    max_tokens: int,
    top_k: int = 3
) -> str:
    """
    Build short context for Transformer
    
    Selects top-k most relevant documents and truncates.
    
    Args:
        docs: Retrieved documents
        max_tokens: Maximum token budget
        top_k: Number of top documents to include
        
    Returns:
        Short context string
    """
    # Take only top-k docs
    top_docs = docs[:top_k]
    
    context_parts = []
    total_chars = 0
    max_chars = max_tokens * 4
    
    for i, doc in enumerate(top_docs):
        content = doc.get("content") or doc.get("text") or str(doc)
        
        # Truncate individual docs to fit budget
        remaining_budget = max_chars - total_chars
        if remaining_budget < 100:
            break
        
        if len(content) > remaining_budget:
            content = content[:remaining_budget - 3] + "..."
        
        context_parts.append(f"\n[{i+1}] {content}")
        total_chars += len(content) + 10
    
    return "".join(context_parts)


def compress_long_context(
    docs: List[Dict[str, Any]],
    max_tokens: int
) -> str:
    """
    Compress long context using extractive summarization
    
    Strategy:
    - Keep document headers (section titles, page numbers)
    - Keep first N and last M tokens of each doc
    - Indicate omissions with [...]
    
    Args:
        docs: Documents to compress
        max_tokens: Target token count
        
    Returns:
        Compressed context string
    """
    max_chars = max_tokens * 4
    compressed_parts = []
    total_chars = 0
    
    # Tokens to keep per document
    chars_per_doc = max_chars // max(len(docs), 1)
    keep_start = min(200, chars_per_doc // 2)
    keep_end = min(100, chars_per_doc // 2)
    
    for doc in docs:
        content = doc.get("content") or doc.get("text") or str(doc)
        metadata = doc.get("metadata", {})
        
        if total_chars >= max_chars:
            break
        
        # Extract metadata
        page_info = metadata.get("page") or metadata.get("page_number", "")
        section = metadata.get("section", "")
        
        # Build compressed version
        if len(content) <= keep_start + keep_end + 20:
            # Short doc, keep all
            compressed = content
        else:
            # Long doc, keep start and end
            start_part = content[:keep_start].strip()
            end_part = content[-keep_end:].strip()
            compressed = f"{start_part}\n[...]\n{end_part}"
        
        # Add header if metadata available
        header = ""
        if page_info:
            header += f"Page {page_info}"
        if section:
            header += f" - {section}"
        
        if header:
            compressed = f"[{header}]\n{compressed}"
        
        compressed_parts.append(compressed)
        total_chars += len(compressed)
    
    return "\n\n".join(compressed_parts)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text
    
    Rule of thumb: ~0.75 tokens per word, or ~4 chars per token
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    word_count = len(text.split())
    return int(word_count * 0.75)


def get_total_pages(docs: List[Dict[str, Any]]) -> int:
    """
    Get total page count from documents
    
    Args:
        docs: Retrieved documents with metadata
        
    Returns:
        Total number of unique pages
    """
    pages = set()
    for doc in docs:
        metadata = doc.get("metadata", {})
        page = metadata.get("page") or metadata.get("page_number")
        if page:
            pages.add(page)
    
    return len(pages) if pages else 0
