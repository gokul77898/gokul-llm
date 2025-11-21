"""
Indian Supreme Court Judgments Ingestion Pipeline

A complete data ingestion pipeline for AWS Indian Supreme Court judgments dataset.
Downloads parquet files, processes them, chunks the text, and ingests into ChromaDB.

Modules:
    - download: Download parquet files from AWS S3
    - parquet_loader: Load and process parquet files
    - chunker: Chunk long texts with overlap
    - chroma_ingest: Main ingestion pipeline
    - test_retrieval: Test retrieval functionality

Usage:
    # Download data
    python3 src/ingest/download.py --years 2018 2019 2020
    
    # Ingest into ChromaDB
    python3 src/ingest/chroma_ingest.py
    
    # Test retrieval
    python3 src/ingest/test_retrieval.py --query "murder case IPC 302"
"""

__version__ = "1.0.0"
__author__ = "MARK AI Team"

from .download import download_parquet, download_multiple_years
from .parquet_loader import load_parquet_file, load_all_parquets
from .chunker import chunk_text, chunk_judgment_dataframe

__all__ = [
    'download_parquet',
    'download_multiple_years',
    'load_parquet_file',
    'load_all_parquets',
    'chunk_text',
    'chunk_judgment_dataframe',
]
