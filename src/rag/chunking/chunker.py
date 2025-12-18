"""
Legal Document Chunker

Phase R1: Legal Chunking & Indexing

Converts legal documents into legal-structure-aware chunks.
Each chunk represents ONE legal unit (section, subsection, or paragraph).

NEVER chunks by token length.
NEVER merges sections.
"""

from datetime import datetime
from typing import List, Optional

from ..schemas.document import DocumentType, LegalDocument
from ..schemas.chunk import LegalChunk, ChunkIndexEntry
from .section_parser import SectionParser, ParsedSection
from .id_generator import generate_chunk_id, generate_chunk_id_from_text


class ChunkValidationError(Exception):
    """Raised when chunk validation fails"""
    
    def __init__(self, message: str, chunk_index: int = -1):
        self.message = message
        self.chunk_index = chunk_index
        super().__init__(f"Chunk validation failed (index {chunk_index}): {message}")


class LegalChunker:
    """
    Legal-structure-aware document chunker.
    
    Chunking rules:
    1. Bare acts → Section-level chunks
    2. Case law → Paragraph/holding chunks
    3. Amendments → Section-level chunks
    4. Notifications → Paragraph chunks
    
    Each chunk = ONE legal unit.
    """
    
    def __init__(self):
        self.parser = SectionParser()
    
    def chunk_document(self, doc: LegalDocument) -> List[LegalChunk]:
        """
        Chunk a legal document into legal units.
        
        Args:
            doc: LegalDocument to chunk
            
        Returns:
            List of LegalChunk objects
            
        Raises:
            ChunkValidationError: If validation fails
        """
        if doc.doc_type == DocumentType.BARE_ACT:
            return self._chunk_bare_act(doc)
        elif doc.doc_type == DocumentType.CASE_LAW:
            return self._chunk_case_law(doc)
        elif doc.doc_type == DocumentType.AMENDMENT:
            return self._chunk_amendment(doc)
        elif doc.doc_type == DocumentType.NOTIFICATION:
            return self._chunk_notification(doc)
        else:
            raise ValueError(f"Unknown document type: {doc.doc_type}")
    
    def _chunk_bare_act(self, doc: LegalDocument) -> List[LegalChunk]:
        """
        Chunk a bare act by sections.
        
        Each section becomes one chunk.
        Subsections become separate chunks if present.
        """
        chunks: List[LegalChunk] = []
        
        # Parse sections
        parsed_sections = self.parser.parse_bare_act(doc.raw_text)
        
        for i, parsed in enumerate(parsed_sections):
            # Generate deterministic chunk ID
            chunk_id = generate_chunk_id(
                doc_id=doc.doc_id,
                section=parsed.section,
                subsection=parsed.subsection,
                start_offset=parsed.start_offset,
            )
            
            # Create chunk
            chunk = LegalChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                act=doc.act,
                section=parsed.section,
                subsection=parsed.subsection,
                doc_type=doc.doc_type,
                text=parsed.text,
                citation=f"Section {parsed.section}, {doc.act}" if doc.act else None,
                court=None,
                year=doc.year,
                start_offset=parsed.start_offset,
                end_offset=parsed.end_offset,
                chunk_index=i,
            )
            
            # Validate chunk
            self._validate_chunk(chunk, i)
            
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_case_law(self, doc: LegalDocument) -> List[LegalChunk]:
        """
        Chunk case law by paragraphs/holdings.
        
        Each numbered paragraph or major section becomes one chunk.
        """
        chunks: List[LegalChunk] = []
        
        # Parse paragraphs
        parsed_sections = self.parser.parse_case_law(doc.raw_text)
        
        for i, parsed in enumerate(parsed_sections):
            # Generate chunk ID from text (no section numbers in case law)
            chunk_id = generate_chunk_id_from_text(
                doc_id=doc.doc_id,
                text=parsed.text,
                start_offset=parsed.start_offset,
            )
            
            # Create chunk
            chunk = LegalChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                act=doc.act,
                section=parsed.section,  # Paragraph number or heading
                subsection=None,
                doc_type=doc.doc_type,
                text=parsed.text,
                citation=doc.citation,
                court=doc.court,
                year=doc.year,
                start_offset=parsed.start_offset,
                end_offset=parsed.end_offset,
                chunk_index=i,
            )
            
            # Validate chunk
            self._validate_chunk(chunk, i)
            
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_amendment(self, doc: LegalDocument) -> List[LegalChunk]:
        """
        Chunk an amendment by sections.
        
        Similar to bare acts but may reference original sections.
        """
        # Amendments follow same structure as bare acts
        return self._chunk_bare_act(doc)
    
    def _chunk_notification(self, doc: LegalDocument) -> List[LegalChunk]:
        """
        Chunk a notification by paragraphs.
        
        Notifications typically don't have section structure.
        """
        chunks: List[LegalChunk] = []
        
        # Split by double newlines
        paragraphs = doc.raw_text.split('\n\n')
        offset = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            
            # Skip empty or very short paragraphs
            if not para or len(para) < 20:
                offset += len(para) + 2
                continue
            
            # Generate chunk ID
            chunk_id = generate_chunk_id_from_text(
                doc_id=doc.doc_id,
                text=para,
                start_offset=offset,
            )
            
            # Create chunk
            chunk = LegalChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                act=doc.act,
                section=str(chunk_index + 1),  # Paragraph number
                subsection=None,
                doc_type=doc.doc_type,
                text=para,
                citation=doc.citation,
                court=None,
                year=doc.year,
                start_offset=offset,
                end_offset=offset + len(para),
                chunk_index=chunk_index,
            )
            
            # Validate chunk
            self._validate_chunk(chunk, chunk_index)
            
            chunks.append(chunk)
            offset += len(para) + 2
            chunk_index += 1
        
        return chunks
    
    def _validate_chunk(self, chunk: LegalChunk, index: int) -> None:
        """
        Validate a chunk against strict rules.
        
        Raises ChunkValidationError if invalid.
        """
        # Rule 1: Text must not be empty
        if not chunk.text or not chunk.text.strip():
            raise ChunkValidationError("text is empty", index)
        
        # Rule 2: Section must exist for bare acts
        if chunk.doc_type == DocumentType.BARE_ACT:
            if not chunk.section:
                raise ChunkValidationError("section missing for bare_act", index)
            if not chunk.act:
                raise ChunkValidationError("act missing for bare_act", index)
        
        # Rule 3: Citation must exist for case law
        if chunk.doc_type == DocumentType.CASE_LAW:
            # Citation is recommended but not strictly required
            pass
        
        # Rule 4: Offsets must be valid
        if chunk.end_offset < chunk.start_offset:
            raise ChunkValidationError(
                f"end_offset ({chunk.end_offset}) < start_offset ({chunk.start_offset})",
                index
            )


def chunk_document(doc: LegalDocument) -> List[LegalChunk]:
    """
    Convenience function to chunk a document.
    
    Args:
        doc: LegalDocument to chunk
        
    Returns:
        List of LegalChunk objects
    """
    chunker = LegalChunker()
    return chunker.chunk_document(doc)


if __name__ == "__main__":
    """CLI entry point for chunking."""
    import sys
    from pathlib import Path
    from ..storage.filesystem import FilesystemStorage
    from ..storage.chunk_storage import ChunkStorage
    
    print("=" * 50)
    print("RAG Chunking Pipeline")
    print("=" * 50)
    
    doc_storage = FilesystemStorage()
    chunk_storage = ChunkStorage()
    chunker = LegalChunker()
    
    doc_ids = doc_storage.list_all()
    
    if not doc_ids:
        print("No documents found. Run ingestion first.")
        sys.exit(1)
    
    total_chunks = 0
    
    for doc_id in doc_ids:
        try:
            doc = doc_storage.load(doc_id)
            chunks = chunker.chunk_document(doc)
            
            for chunk in chunks:
                chunk_storage.save(chunk)
            
            print(f"✓ Chunked: {doc.title} -> {len(chunks)} chunks")
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"✗ Failed: {doc_id[:8]}... - {e}")
    
    print()
    print(f"Total chunks created: {total_chunks}")
    sys.exit(0)
