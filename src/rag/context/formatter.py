"""
Evidence Block Formatting

Phase R4: Context Assembler

Formats evidence blocks for context assembly.
Produces strict, auditable evidence format.

NO LLMs used in this module.
"""

from dataclasses import dataclass
from typing import Optional

from .citation import Citation, CitationFormatter, SourceType


@dataclass
class FormattedEvidence:
    """
    Formatted evidence block ready for context.
    
    Attributes:
        index: Evidence block index (1-based)
        citation_line: Formatted citation
        text: Evidence text content
        source_line: SOURCE: type line
        full_block: Complete formatted block
        token_estimate: Estimated token count
        chunk_id: Original chunk ID
        is_valid: Whether block is valid for inclusion
        drop_reason: Reason if dropped
    """
    index: int
    citation_line: str
    text: str
    source_line: str
    full_block: str
    token_estimate: int
    chunk_id: str
    is_valid: bool = True
    drop_reason: Optional[str] = None


class EvidenceFormatter:
    """
    Formats evidence blocks for context assembly.
    
    Output format:
    
    [1] (IPC, Section 420, 1860)
    Text...
    SOURCE: Bare Act
    
    [2] (Supreme Court, 2012, XYZ v State)
    Text...
    SOURCE: Case Law
    """
    
    # Context markers
    EVIDENCE_START = "EVIDENCE_START"
    EVIDENCE_END = "EVIDENCE_END"
    
    def __init__(self, citation_formatter: CitationFormatter = None):
        """
        Initialize evidence formatter.
        
        Args:
            citation_formatter: Citation formatter instance
        """
        self.citation_formatter = citation_formatter or CitationFormatter()
    
    def format_evidence_block(
        self,
        index: int,
        chunk_id: str,
        text: str,
        act: Optional[str],
        section: Optional[str],
        year: Optional[int],
        doc_type: str,
        court: Optional[str] = None,
        citation: Optional[str] = None,
        title: Optional[str] = None,
    ) -> FormattedEvidence:
        """
        Format a single evidence block.
        
        Args:
            index: Block index (1-based)
            chunk_id: Original chunk ID
            text: Evidence text
            act: Act name
            section: Section number
            year: Year
            doc_type: Document type
            court: Court name (for case law)
            citation: Legal citation
            title: Document title
            
        Returns:
            FormattedEvidence object
        """
        # Format citation
        cit = self.citation_formatter.format_citation(
            act=act,
            section=section,
            year=year,
            doc_type=doc_type,
            court=court,
            citation=citation,
            title=title,
        )
        
        # Check if citation is valid
        if not cit.is_valid:
            return FormattedEvidence(
                index=index,
                citation_line="",
                text=text,
                source_line="",
                full_block="",
                token_estimate=0,
                chunk_id=chunk_id,
                is_valid=False,
                drop_reason=f"missing_citation: {', '.join(cit.missing_fields)}",
            )
        
        # Format citation line
        citation_line = self.citation_formatter.format_citation_line(cit)
        
        # Format source line
        source_line = f"SOURCE: {cit.source_type.value}"
        
        # Build full block
        full_block = f"[{index}] {citation_line}\n{text.strip()}\n{source_line}"
        
        # Estimate tokens (rough: chars / 4)
        token_estimate = max(1, len(full_block) // 4)
        
        return FormattedEvidence(
            index=index,
            citation_line=citation_line,
            text=text.strip(),
            source_line=source_line,
            full_block=full_block,
            token_estimate=token_estimate,
            chunk_id=chunk_id,
            is_valid=True,
        )
    
    def assemble_context(self, evidence_blocks: list[FormattedEvidence]) -> str:
        """
        Assemble multiple evidence blocks into final context.
        
        Args:
            evidence_blocks: List of formatted evidence blocks
            
        Returns:
            Complete context string with EVIDENCE_START/END markers
        """
        if not evidence_blocks:
            return ""
        
        # Filter to valid blocks only
        valid_blocks = [b for b in evidence_blocks if b.is_valid]
        
        if not valid_blocks:
            return ""
        
        # Build context
        lines = [self.EVIDENCE_START]
        
        for block in valid_blocks:
            lines.append(block.full_block)
            lines.append("")  # Empty line between blocks
        
        # Remove trailing empty line and add end marker
        if lines[-1] == "":
            lines.pop()
        lines.append(self.EVIDENCE_END)
        
        return "\n".join(lines)
    
    def reindex_blocks(self, blocks: list[FormattedEvidence]) -> list[FormattedEvidence]:
        """
        Reindex evidence blocks after filtering/reordering.
        
        Args:
            blocks: List of evidence blocks
            
        Returns:
            Blocks with updated indices
        """
        reindexed = []
        
        for i, block in enumerate(blocks, start=1):
            if block.is_valid:
                # Rebuild full block with new index
                new_block = FormattedEvidence(
                    index=i,
                    citation_line=block.citation_line,
                    text=block.text,
                    source_line=block.source_line,
                    full_block=f"[{i}] {block.citation_line}\n{block.text}\n{block.source_line}",
                    token_estimate=block.token_estimate,
                    chunk_id=block.chunk_id,
                    is_valid=True,
                )
                reindexed.append(new_block)
            else:
                reindexed.append(block)
        
        return reindexed
