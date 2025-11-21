"""
Document Chunker for RAG
Chunks documents into 300-400 token chunks with overlap
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    content: str
    chunk_id: int
    token_count: int
    source_doc: str
    page_number: int = 0
    metadata: Dict[str, Any] = None

class DocumentChunker:
    """
    Intelligent document chunker
    - Chunks: 300-400 tokens
    - Overlap: 50 tokens
    - Preserves sentence boundaries
    """
    
    def __init__(self, chunk_size: int = 350, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(self, content: str, source_doc: str = "unknown") -> List[DocumentChunk]:
        """
        Chunk document into optimal sizes
        
        Args:
            content: Document content
            source_doc: Source document name
            
        Returns:
            List of DocumentChunk objects
        """
        # Split into sentences first
        sentences = self._split_into_sentences(content)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_content = ' '.join(current_chunk)
                    chunks.append(DocumentChunk(
                        content=chunk_content,
                        chunk_id=chunk_id,
                        token_count=current_tokens,
                        source_doc=source_doc,
                        metadata={'sentence_count': len(current_chunk)}
                    ))
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                token_count=current_tokens,
                source_doc=source_doc,
                metadata={'sentence_count': len(current_chunk)}
            ))
        
        logger.info(f"Chunked document '{source_doc}' into {len(chunks)} chunks")
        return chunks
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences preserving structure"""
        import re
        
        # Split on period, question mark, exclamation
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, current_chunk: List[str]) -> List[str]:
        """Get sentences for overlap"""
        if not current_chunk:
            return []
        
        # Calculate how many sentences to include in overlap
        overlap_sentences = []
        overlap_tokens = 0
        
        for sentence in reversed(current_chunk):
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if overlap_tokens + sentence_tokens > self.overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens
        
        return overlap_sentences
    
    def chunk_multiple_documents(self, documents: List[Dict[str, str]]) -> List[DocumentChunk]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of dicts with 'content' and 'source' keys
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            
            if content:
                chunks = self.chunk_document(content, source)
                all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
