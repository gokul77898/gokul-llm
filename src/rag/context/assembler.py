"""
Context Assembler

Phase R4: Context Assembler

Main assembler that combines validated evidence into strict,
bounded, auditable context for decoder consumption.

NO LLMs used in this module.
NO retrieval here.
NO filtering here.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from pathlib import Path

from .token_budget import TokenBudget, BudgetConfig
from .citation import CitationFormatter
from .formatter import EvidenceFormatter, FormattedEvidence


class ContextRefusalReason(str, Enum):
    """Machine-readable refusal reasons for context assembly."""
    NO_VALID_EVIDENCE = "no_valid_evidence"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    MISSING_CITATION = "missing_citation"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class ContextStatus(str, Enum):
    """Context assembly status."""
    ASSEMBLED = "assembled"
    REFUSE = "refuse"


@dataclass
class DroppedChunk:
    """Information about a dropped chunk."""
    chunk_id: str
    reason: str
    details: Optional[str] = None


@dataclass
class UsedChunk:
    """Information about a used chunk."""
    chunk_id: str
    index: int
    section: Optional[str]
    act: Optional[str]
    token_estimate: int


@dataclass
class ContextResult:
    """
    Result of context assembly.
    
    Attributes:
        status: assembled or refuse
        context_text: Assembled context (if successful)
        used_chunks: List of chunks included in context
        dropped_chunks: List of chunks dropped with reasons
        token_count: Total token count
        refusal_reason: Machine-readable reason if refused
        refusal_message: Human-readable message if refused
    """
    status: ContextStatus
    context_text: str = ""
    used_chunks: List[UsedChunk] = field(default_factory=list)
    dropped_chunks: List[DroppedChunk] = field(default_factory=list)
    token_count: int = 0
    refusal_reason: Optional[ContextRefusalReason] = None
    refusal_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "context_text": self.context_text,
            "used_chunks": [asdict(c) for c in self.used_chunks],
            "dropped_chunks": [asdict(c) for c in self.dropped_chunks],
            "token_count": self.token_count,
            "refusal_reason": self.refusal_reason.value if self.refusal_reason else None,
            "refusal_message": self.refusal_message,
        }


class ContextAssembler:
    """
    Assembles validated evidence into bounded, auditable context.
    
    Features:
    - Token budget enforcement
    - Evidence ordering by relevance
    - Citation formatting
    - Strict evidence completeness
    
    The decoder never sees raw documents — only assembled evidence blocks.
    """
    
    def __init__(
        self,
        budget_config: BudgetConfig = None,
        log_dir: str = "logs",
    ):
        """
        Initialize context assembler.
        
        Args:
            budget_config: Token budget configuration
            log_dir: Directory for assembly logs
        """
        self.budget = TokenBudget(budget_config)
        self.citation_formatter = CitationFormatter()
        self.evidence_formatter = EvidenceFormatter(self.citation_formatter)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rag_context.jsonl"
    
    def assemble(
        self,
        query: str,
        validated_chunks: List[Dict[str, Any]],
    ) -> ContextResult:
        """
        Assemble context from validated chunks.
        
        Args:
            query: User query (for ordering)
            validated_chunks: List of validated chunk dictionaries
                Expected keys: chunk_id, text, section, act, score, doc_type, year
                Optional: court, citation, title
            
        Returns:
            ContextResult with assembled context or refusal
        """
        # Check for empty input
        if not validated_chunks:
            result = ContextResult(
                status=ContextStatus.REFUSE,
                refusal_reason=ContextRefusalReason.NO_VALID_EVIDENCE,
                refusal_message="No validated chunks provided for context assembly.",
            )
            self._log_assembly(query, 0, result)
            return result
        
        # Step 1: Order chunks by priority
        ordered_chunks = self._order_chunks(query, validated_chunks)
        
        # Step 2: Format evidence blocks and validate citations
        formatted_blocks = []
        dropped_citation = []
        
        for i, chunk in enumerate(ordered_chunks, start=1):
            block = self.evidence_formatter.format_evidence_block(
                index=i,
                chunk_id=chunk.get('chunk_id', ''),
                text=chunk.get('text', ''),
                act=chunk.get('act'),
                section=chunk.get('section'),
                year=chunk.get('year'),
                doc_type=chunk.get('doc_type', 'unknown'),
                court=chunk.get('court'),
                citation=chunk.get('citation'),
                title=chunk.get('title'),
            )
            
            if block.is_valid:
                formatted_blocks.append((block, chunk))
            else:
                dropped_citation.append(DroppedChunk(
                    chunk_id=chunk.get('chunk_id', ''),
                    reason="missing_citation",
                    details=block.drop_reason,
                ))
        
        # Check if all chunks dropped due to citation issues
        if not formatted_blocks:
            result = ContextResult(
                status=ContextStatus.REFUSE,
                dropped_chunks=dropped_citation,
                refusal_reason=ContextRefusalReason.MISSING_CITATION,
                refusal_message="All chunks dropped due to missing citation metadata.",
            )
            self._log_assembly(query, len(validated_chunks), result)
            return result
        
        # Step 3: Apply token budget
        self.budget.reset()
        used_blocks = []
        dropped_budget = []
        
        for block, chunk in formatted_blocks:
            if self.budget.allocate(block.token_estimate):
                used_blocks.append((block, chunk))
            else:
                dropped_budget.append(DroppedChunk(
                    chunk_id=chunk.get('chunk_id', ''),
                    reason="token_budget_exceeded",
                    details=f"Token estimate: {block.token_estimate}, remaining: {self.budget.remaining_tokens}",
                ))
        
        # Check if any blocks fit
        if not used_blocks:
            all_dropped = dropped_citation + dropped_budget
            result = ContextResult(
                status=ContextStatus.REFUSE,
                dropped_chunks=all_dropped,
                refusal_reason=ContextRefusalReason.TOKEN_BUDGET_EXCEEDED,
                refusal_message="No chunks fit within token budget.",
            )
            self._log_assembly(query, len(validated_chunks), result)
            return result
        
        # Step 4: Reindex and assemble
        final_blocks = [block for block, _ in used_blocks]
        final_blocks = self.evidence_formatter.reindex_blocks(final_blocks)
        
        context_text = self.evidence_formatter.assemble_context(final_blocks)
        
        # Build result
        used_chunks = []
        for block in final_blocks:
            if block.is_valid:
                # Find original chunk data
                orig_chunk = next(
                    (c for b, c in used_blocks if b.chunk_id == block.chunk_id),
                    {}
                )
                used_chunks.append(UsedChunk(
                    chunk_id=block.chunk_id,
                    index=block.index,
                    section=orig_chunk.get('section'),
                    act=orig_chunk.get('act'),
                    token_estimate=block.token_estimate,
                ))
        
        all_dropped = dropped_citation + dropped_budget
        
        result = ContextResult(
            status=ContextStatus.ASSEMBLED,
            context_text=context_text,
            used_chunks=used_chunks,
            dropped_chunks=all_dropped,
            token_count=self.budget.used_tokens,
        )
        
        self._log_assembly(query, len(validated_chunks), result)
        return result
    
    def _order_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Order chunks by priority.
        
        Order:
        1. Exact section match
        2. Statute match
        3. Court hierarchy (SC > HC > others)
        4. Retrieval score (descending)
        
        Args:
            query: User query
            chunks: List of chunks
            
        Returns:
            Ordered list of chunks
        """
        # Extract query section
        query_section = self._extract_section(query)
        query_statute = self._extract_statute(query)
        
        def sort_key(chunk: Dict) -> Tuple:
            # Lower = higher priority
            
            # 1. Exact section match (0 if match, 1 if not)
            section_match = 0
            if query_section:
                chunk_section = str(chunk.get('section', '')).lower()
                if query_section.lower() in chunk_section:
                    section_match = 0
                else:
                    section_match = 1
            
            # 2. Statute match (0 if match, 1 if not)
            statute_match = 0
            if query_statute:
                chunk_act = str(chunk.get('act', '')).lower()
                if query_statute.lower() in chunk_act:
                    statute_match = 0
                else:
                    statute_match = 1
            
            # 3. Court hierarchy (lower = higher court)
            court_rank = self.citation_formatter.get_court_rank(chunk.get('court'))
            
            # 4. Score (negative for descending)
            score = -float(chunk.get('score', 0) or chunk.get('adjusted_score', 0))
            
            return (section_match, statute_match, court_rank, score)
        
        return sorted(chunks, key=sort_key)
    
    def _extract_section(self, text: str) -> Optional[str]:
        """Extract section number from text."""
        match = re.search(r'section\s+(\d+[a-z]?)', text.lower())
        return match.group(1) if match else None
    
    def _extract_statute(self, text: str) -> Optional[str]:
        """Extract statute name from text."""
        text_lower = text.lower()
        
        if 'ipc' in text_lower or 'penal code' in text_lower:
            return 'ipc'
        if 'crpc' in text_lower or 'criminal procedure' in text_lower:
            return 'crpc'
        if 'cpc' in text_lower or 'civil procedure' in text_lower:
            return 'cpc'
        
        return None
    
    def _log_assembly(
        self,
        query: str,
        input_count: int,
        result: ContextResult,
    ) -> None:
        """
        Log assembly run to JSONL file.
        
        Args:
            query: User query
            input_count: Number of input chunks
            result: Assembly result
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "input_chunk_count": input_count,
            "used_chunk_count": len(result.used_chunks),
            "dropped_chunk_count": len(result.dropped_chunks),
            "token_count": result.token_count,
            "status": result.status.value,
            "refusal_reason": result.refusal_reason.value if result.refusal_reason else None,
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log assembly: {e}")
