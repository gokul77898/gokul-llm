"""Ingestion Pipeline - Extract, Chunk, Embed, Store"""
import logging
import time
import os
from pathlib import Path
from typing import List, Optional
from .client import ChromaDBClient
from .extractor import TextExtractor
from .chunker import TextChunker
from .embeddings import EmbeddingModel
from .schema import DocumentChunk, IngestionStats
from .config import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

# SETUP MODE FLAG - Blocks all ingestion
SETUP_MODE = os.getenv("SETUP_MODE", "true").lower() == "true"

def ingest_file(
    file_path: str,
    collection_name: str = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> IngestionStats:
    """Ingest file into ChromaDB. Returns stats."""
    
    # BLOCK INGESTION IN SETUP MODE
    if SETUP_MODE:
        logger.error("❌ DATA INGESTION IS BLOCKED - SETUP MODE")
        raise RuntimeError(
            "Data ingestion disabled. System is in SETUP MODE. "
            "To enable ingestion, set environment variable: SETUP_MODE=false"
        )
    
    start_time = time.time()
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Ingesting: {path.name}")
    
    try:
        # 1. Extract text
        pages = TextExtractor.extract_text_with_pages(str(path))
        total_chars = sum(len(text) for _, text in pages)
        total_words = sum(len(text.split()) for _, text in pages)
        
        # 2. Chunk text
        chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        all_chunks = []
        for page_num, text in pages:
            chunks = chunker.chunk_text(text)
            for i, chunk_text in enumerate(chunks):
                doc_chunk = DocumentChunk.create(
                    text=chunk_text,
                    filename=path.name,
                    chunk_index=len(all_chunks),
                    page=page_num,
                    total_chunks=len(chunks)
                )
                all_chunks.append(doc_chunk)
        
        # 3. Embed chunks
        embedder = EmbeddingModel()
        texts = [c.text for c in all_chunks]
        embeddings = embedder.embed(texts)
        
        # 4. Store in ChromaDB
        client = ChromaDBClient()
        collection = client.get_or_create_collection(collection_name or DEFAULT_COLLECTION)
        
        ids = [c.id for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        elapsed = time.time() - start_time
        stats = IngestionStats(
            filename=path.name,
            chunks_created=len(all_chunks),
            total_chars=total_chars,
            total_words=total_words,
            pages=len(pages),
            ingestion_time=elapsed
        )
        
        logger.info(f"Ingestion complete: {stats.summary()}")
        return stats
        
    except Exception as e:
        logger.error(f"Ingestion failed for {path.name}: {e}")
        raise

def ingest_directory(
    dir_path: str,
    collection_name: str = None,
    recursive: bool = False
) -> List[IngestionStats]:
    """Ingest all supported files in directory."""
    
    # BLOCK INGESTION IN SETUP MODE
    if SETUP_MODE:
        logger.error("❌ DATA INGESTION IS BLOCKED - SETUP MODE")
        raise RuntimeError(
            "Data ingestion disabled. System is in SETUP MODE. "
            "To enable ingestion, set environment variable: SETUP_MODE=false"
        )
    
    path = Path(dir_path)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")
    
    pattern = "**/*" if recursive else "*"
    files = [f for f in path.glob(pattern) if f.is_file() and TextExtractor.is_supported(str(f))]
    
    logger.info(f"Found {len(files)} files to ingest")
    
    results = []
    for file in files:
        try:
            stats = ingest_file(str(file), collection_name)
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to ingest {file.name}: {e}")
    
    return results
