"""
Token Budget Management

Phase R4: Context Assembler

Manages token budget for context assembly.
Estimates token counts and enforces limits.

NO LLMs used in this module.
"""

from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class BudgetConfig:
    """
    Configuration for token budget.
    
    Attributes:
        max_tokens: Maximum tokens allowed in context
        min_tokens: Minimum tokens required for valid context
        chars_per_token: Estimated characters per token (for estimation)
        overhead_tokens: Tokens reserved for EVIDENCE_START/END markers
    """
    max_tokens: int = 2500
    min_tokens: int = 50
    chars_per_token: float = 4.0
    overhead_tokens: int = 20  # For EVIDENCE_START/END markers


@dataclass
class BudgetResult:
    """Result of budget check."""
    fits: bool
    estimated_tokens: int
    remaining_tokens: int
    reason: str = ""


class TokenBudget:
    """
    Token budget manager for context assembly.
    
    Rules:
    - Default max tokens: 2,500
    - If overflow: drop lowest-ranked chunks first
    - Never split a chunk mid-sentence
    - Minimum: at least 1 full chunk
    """
    
    def __init__(self, config: BudgetConfig = None):
        """
        Initialize token budget manager.
        
        Args:
            config: Budget configuration
        """
        self.config = config or BudgetConfig()
        self._used_tokens = self.config.overhead_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses simple character-based estimation.
        More accurate tokenization would require model-specific tokenizer.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Simple estimation: chars / chars_per_token
        # Add small overhead for special tokens
        char_count = len(text)
        estimated = int(char_count / self.config.chars_per_token)
        
        # Minimum 1 token for non-empty text
        return max(1, estimated)
    
    def estimate_chunk_tokens(self, chunk_text: str, citation_overhead: int = 15) -> int:
        """
        Estimate tokens for a formatted chunk including citation.
        
        Args:
            chunk_text: Chunk text content
            citation_overhead: Estimated tokens for citation line
            
        Returns:
            Total estimated tokens
        """
        text_tokens = self.estimate_tokens(chunk_text)
        return text_tokens + citation_overhead
    
    def check_budget(self, text: str) -> BudgetResult:
        """
        Check if text fits in remaining budget.
        
        Args:
            text: Text to check
            
        Returns:
            BudgetResult with fit status
        """
        estimated = self.estimate_tokens(text)
        remaining = self.config.max_tokens - self._used_tokens
        fits = estimated <= remaining
        
        return BudgetResult(
            fits=fits,
            estimated_tokens=estimated,
            remaining_tokens=remaining,
            reason="" if fits else "token_budget_exceeded",
        )
    
    def allocate(self, tokens: int) -> bool:
        """
        Allocate tokens from budget.
        
        Args:
            tokens: Number of tokens to allocate
            
        Returns:
            True if allocation successful
        """
        if self._used_tokens + tokens > self.config.max_tokens:
            return False
        
        self._used_tokens += tokens
        return True
    
    def reset(self) -> None:
        """Reset budget to initial state."""
        self._used_tokens = self.config.overhead_tokens
    
    @property
    def used_tokens(self) -> int:
        """Get currently used tokens."""
        return self._used_tokens
    
    @property
    def remaining_tokens(self) -> int:
        """Get remaining tokens."""
        return self.config.max_tokens - self._used_tokens
    
    def select_chunks_within_budget(
        self,
        chunks: List[Tuple[str, int, dict]],
    ) -> Tuple[List[dict], List[Tuple[dict, str]]]:
        """
        Select chunks that fit within budget.
        
        Chunks should be pre-sorted by priority (highest first).
        Drops lowest-priority chunks first when budget exceeded.
        
        Args:
            chunks: List of (text, estimated_tokens, chunk_data) tuples
            
        Returns:
            Tuple of (selected_chunks, dropped_chunks_with_reasons)
        """
        self.reset()
        selected = []
        dropped = []
        
        for text, tokens, chunk_data in chunks:
            if self.allocate(tokens):
                selected.append(chunk_data)
            else:
                dropped.append((chunk_data, "token_budget_exceeded"))
        
        return selected, dropped
    
    def meets_minimum(self) -> bool:
        """
        Check if current usage meets minimum viable context.
        
        Returns:
            True if at least minimum tokens used (excluding overhead)
        """
        content_tokens = self._used_tokens - self.config.overhead_tokens
        return content_tokens >= self.config.min_tokens
