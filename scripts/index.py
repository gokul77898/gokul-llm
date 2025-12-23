#!/usr/bin/env python3
"""
Phase-1 RAG: Vector Indexing Script

Indexes chunks from data/rag/chunks/ into ChromaDB at data/rag/chromadb/
Creates dense vector embeddings for semantic search.

Usage:
    python scripts/index.py
    python scripts/index.py --config configs/phase1_rag.yaml

This script is stateless and can be run independently.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Phase-1 RAG: Vector Indexing")
    parser.add_argument(
        "--config",
        default="configs/phase1_rag.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from scratch (delete existing)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        log(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Print header
    print("=" * 60)
    print("Phase-1 RAG: Vector Indexing")
    print("=" * 60)
    log(f"Config: {args.config}")
    log(f"Version: {config.get('version', 'unknown')}")
    
    # Resolve paths
    chunks_dir = PROJECT_ROOT / config['paths']['chunks_dir']
    chromadb_dir = PROJECT_ROOT / config['paths']['chromadb_dir']
    
    log(f"Chunks directory: {chunks_dir}")
    log(f"ChromaDB directory: {chromadb_dir}")
    
    # Get encoder config
    encoder_config = config.get('encoder', {})
    encoder_model = encoder_config.get('model_name', 'BAAI/bge-large-en-v1.5')
    
    # Get retrieval config
    retrieval_config = config.get('retrieval', {})
    collection_name = retrieval_config.get('collection_name', 'legal_chunks')
    
    log(f"Encoder model: {encoder_model}")
    log(f"Collection name: {collection_name}")
    
    # Ensure directories exist
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chromadb_dir.mkdir(parents=True, exist_ok=True)
    
    # Check ChromaDB availability
    if not CHROMADB_AVAILABLE:
        log("ERROR: ChromaDB not installed. Run: pip install chromadb")
        sys.exit(1)
    
    # Handle rebuild flag
    if args.rebuild:
        log("WARNING: Rebuild flag set - will recreate collection")
        import shutil
        if chromadb_dir.exists():
            shutil.rmtree(chromadb_dir)
            chromadb_dir.mkdir(parents=True, exist_ok=True)
            log("Deleted existing ChromaDB data")
    
    print()
    log("Initializing ChromaDB...")
    log(f"Loading embedding model: {encoder_model}")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(chromadb_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    
    # Create embedding function
    try:
        from chromadb.utils import embedding_functions
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=encoder_model
        )
        log("Embedding model loaded successfully")
    except Exception as e:
        log(f"WARNING: Failed to load embedding model: {e}")
        log("Using default embeddings")
        embedding_fn = None
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Index chunks
    print()
    log("Indexing chunks...")
    
    # Load chunks from filesystem
    chunk_files = [f for f in chunks_dir.glob("*.json") if f.name != "index.json"]
    log(f"Found {len(chunk_files)} chunk files")
    
    indexed_count = 0
    skipped_count = 0
    
    # Batch processing
    batch_ids = []
    batch_documents = []
    batch_metadatas = []
    batch_size = 100
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk = json.load(f)
            
            chunk_id = chunk.get('chunk_id', '')
            text = chunk.get('text', '')
            
            if not chunk_id or not text:
                continue
            
            # Check if already indexed
            existing = collection.get(ids=[chunk_id])
            if existing and existing['ids']:
                skipped_count += 1
                continue
            
            batch_ids.append(chunk_id)
            batch_documents.append(text)
            batch_metadatas.append({
                "chunk_id": chunk_id,
                "semantic_id": chunk.get('semantic_id') or "",
                "act": chunk.get('act') or "",
                "section": chunk.get('section') or "",
                "doc_type": chunk.get('doc_type', 'unknown'),
                "year": chunk.get('year') or 0,
                "citation": chunk.get('citation') or "",
                "court": chunk.get('court') or "",
                "doc_id": chunk.get('doc_id') or "",
            })
            
            # Batch add
            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )
                indexed_count += len(batch_ids)
                batch_ids = []
                batch_documents = []
                batch_metadatas = []
                
        except Exception as e:
            log(f"WARNING: Failed to process {chunk_file.name}: {e}")
    
    # Add remaining batch
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        indexed_count += len(batch_ids)
    
    # Get collection stats
    collection_count = collection.count()
    
    # Print summary
    print()
    print("=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)
    log(f"New chunks indexed: {indexed_count}")
    log(f"Total chunks in collection: {collection_count}")
    log(f"Collection name: {collection_name}")
    log(f"ChromaDB path: {chromadb_dir}")
    log(f"Encoder model: {encoder_model}")
    
    log(f"Skipped (already indexed): {skipped_count}")
    
    # Verify index with a test query
    print()
    log("Verifying index with test query...")
    
    try:
        test_results = collection.query(
            query_texts=["legal document"],
            n_results=1
        )
        if test_results and test_results['ids'] and test_results['ids'][0]:
            log(f"✓ Index verification passed - found {len(test_results['ids'][0])} result(s)")
        else:
            log("○ Index empty or no matching documents")
    except Exception as e:
        log(f"✗ Index verification failed: {e}")
    
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
