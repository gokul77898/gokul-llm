"""
Text Chunker for Legal Judgments

Chunks long judgment texts into smaller, overlapping segments suitable for
vector embedding and retrieval.

Usage:
    from src.ingest.chunker import chunk_text, chunk_judgment_dataframe
"""

import pandas as pd
from typing import List, Dict, Any
from uuid import uuid4


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or not isinstance(text, str):
        return []
    
    text = text.strip()
    
    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary (. ! ?)
        if end < len(text):
            # Look for last sentence boundary in the chunk
            last_period = max(
                chunk.rfind('. '),
                chunk.rfind('! '),
                chunk.rfind('? ')
            )
            
            # If found a sentence boundary and it's not too far back
            if last_period > chunk_size * 0.5:  # At least halfway through chunk
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        
        # Move start position (with overlap)
        start = end - overlap
        
        # Ensure we're making progress
        if start <= chunks[-1:][0].__len__() if chunks else 0:
            start = end
    
    return chunks


def chunk_judgment_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 1500,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Chunk all judgments in a DataFrame and prepare for ChromaDB ingestion.
    
    Args:
        df: DataFrame with judgment data
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List[Dict]: List of chunks with metadata, ready for ChromaDB
    """
    print(f"\n✂️  Chunking judgments (size={chunk_size}, overlap={overlap})...")
    
    chunks_data = []
    total_chunks = 0
    
    for idx, row in df.iterrows():
        # Extract fields
        text = str(row['judgment_text'])
        case_number = str(row.get('case_number', 'Unknown'))
        judges = str(row.get('judges', 'Unknown'))
        date = str(row.get('date_of_judgment', 'Unknown'))
        
        # Chunk the judgment text
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        # Create chunk data with metadata
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_data = {
                'id': str(uuid4()),
                'text': chunk_text,
                'metadata': {
                    'case_number': case_number,
                    'judges': judges,
                    'date': date,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks),
                    'source_row': idx
                }
            }
            chunks_data.append(chunk_data)
            total_chunks += 1
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(df)} judgments ({total_chunks} chunks)...")
    
    print(f"✅ Chunking complete!")
    print(f"   Total judgments: {len(df)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Avg chunks per judgment: {total_chunks / len(df):.1f}")
    
    return chunks_data


def get_chunk_stats(chunks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about chunks.
    
    Args:
        chunks_data: List of chunk dictionaries
        
    Returns:
        Dict with chunk statistics
    """
    chunk_lengths = [len(chunk['text']) for chunk in chunks_data]
    
    stats = {
        'total_chunks': len(chunks_data),
        'avg_length': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
        'max_length': max(chunk_lengths) if chunk_lengths else 0,
        'min_length': min(chunk_lengths) if chunk_lengths else 0,
        'total_chars': sum(chunk_lengths)
    }
    
    return stats


def print_chunk_stats(chunks_data: List[Dict[str, Any]]):
    """Print statistics about chunks."""
    stats = get_chunk_stats(chunks_data)
    
    print("\n📊 Chunk Statistics:")
    print(f"   Total chunks: {stats['total_chunks']:,}")
    print(f"   Avg length: {stats['avg_length']:.0f} chars")
    print(f"   Max length: {stats['max_length']:,} chars")
    print(f"   Min length: {stats['min_length']:,} chars")
    print(f"   Total chars: {stats['total_chars']:,}")


if __name__ == "__main__":
    """Test the chunker."""
    import sys
    
    # Test with sample text
    sample_text = """
    This is a legal judgment from the Supreme Court of India. It contains multiple
    paragraphs and sections. The judgment discusses various legal principles and
    precedents. This is the first part. This is the second part with more details.
    The court considered the arguments presented by both parties. The petitioner
    argued that the lower court's decision was incorrect. The respondent maintained
    that the judgment should be upheld. After careful consideration, the court
    decided to allow the appeal. The judgment is hereby set aside.
    """ * 10  # Make it longer
    
    print("Testing text chunker...")
    print(f"Sample text length: {len(sample_text)} chars")
    
    chunks = chunk_text(sample_text, chunk_size=300, overlap=50)
    
    print(f"\n✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"  {chunk[:100]}...")
    
    print("\n✅ Chunker test successful!")
    sys.exit(0)
