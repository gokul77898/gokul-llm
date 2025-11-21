"""
Text Chunking Module

Provides intelligent text chunking with overlap for better context preservation.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunking with configurable size and overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        
        logger.info(f"TextChunker initialized (size={chunk_size}, overlap={overlap})")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < len(text):
                end = self._find_sentence_boundary(text, end)
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Move start position (with overlap)
            start = end - self.overlap
            
            # Prevent infinite loop
            if start <= chunks.__len__() * (self.chunk_size - self.overlap):
                start = end - self.overlap
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def chunk_text_with_metadata(
        self,
        text: str,
        base_metadata: dict = None
    ) -> List[Tuple[str, dict]]:
        """
        Chunk text and attach metadata to each chunk.
        
        Args:
            text: Input text
            base_metadata: Base metadata to include in each chunk
            
        Returns:
            List[Tuple[str, dict]]: List of (chunk_text, metadata) tuples
        """
        chunks = self.chunk_text(text)
        base_metadata = base_metadata or {}
        
        result = []
        for i, chunk in enumerate(chunks):
            metadata = base_metadata.copy()
            metadata['chunk_index'] = i
            metadata['total_chunks'] = len(chunks)
            metadata['char_count'] = len(chunk)
            metadata['word_count'] = len(chunk.split())
            
            result.append((chunk, metadata))
        
        return result
    
    def chunk_by_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """
        Chunk text by sentences (alternative method).
        
        Args:
            text: Input text
            max_sentences: Maximum sentences per chunk
            
        Returns:
            List[str]: List of chunks
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            # Check if we should finalize chunk
            if len(current_chunk) >= max_sentences or current_length >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep last sentence for overlap
                if self.overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
        
        # Add remaining
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    @staticmethod
    def _find_sentence_boundary(text: str, position: int) -> int:
        """Find nearest sentence boundary near position."""
        # Look for sentence endings within reasonable range
        search_range = 100
        start = max(0, position - search_range)
        end = min(len(text), position + search_range)
        
        segment = text[start:end]
        
        # Find sentence endings (. ! ?)
        sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', segment)]
        
        if sentence_ends:
            # Find closest to target position
            target_pos = position - start
            closest = min(sentence_ends, key=lambda x: abs(x - target_pos))
            return start + closest
        
        # No sentence boundary found, return original position
        return position
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# Convenience function
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunk text with overlap (convenience function).
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        List[str]: List of chunks
    """
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_text(text)
