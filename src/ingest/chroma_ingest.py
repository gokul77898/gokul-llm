"""
ChromaDB Ingestion Pipeline for Indian Supreme Court Judgments

Main script to ingest parquet files into ChromaDB vector database.

Usage:
    python3 src/ingest/chroma_ingest.py --data-dir data/parquet
"""

import argparse
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.ingest.parquet_loader import load_all_parquets, print_stats
from src.ingest.chunker import chunk_judgment_dataframe, print_chunk_stats


# ChromaDB configuration
CHROMA_DB_PATH = "db_store/chroma"
COLLECTION_NAME = "supreme_court_judgments"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def initialize_chromadb(reset: bool = False) -> chromadb.Collection:
    """
    Initialize ChromaDB client and get/create collection.
    
    Args:
        reset: If True, delete existing collection and create new one
        
    Returns:
        chromadb.Collection: The collection for ingestion
    """
    print("\n🔧 Initializing ChromaDB...")
    print(f"   Path: {CHROMA_DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    
    # Initialize client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Create embedding function
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Handle reset if requested
    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"   🗑️  Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass  # Collection doesn't exist, that's fine
    
    # Get or create collection
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"description": "Indian Supreme Court Judgments from AWS Open Data"}
        )
        
        existing_count = collection.count()
        if existing_count > 0:
            print(f"   ⚠️  Collection exists with {existing_count} documents")
            if not reset:
                print(f"   💡 Use --reset flag to start fresh")
        else:
            print(f"   ✅ Created new collection")
        
        return collection
        
    except Exception as e:
        print(f"❌ Failed to initialize ChromaDB: {e}")
        raise


def ingest_chunks_to_chroma(
    chunks_data: list,
    collection: chromadb.Collection,
    batch_size: int = 100
) -> int:
    """
    Ingest chunks into ChromaDB collection in batches.
    
    Args:
        chunks_data: List of chunk dictionaries with id, text, metadata
        collection: ChromaDB collection
        batch_size: Number of chunks to ingest per batch
        
    Returns:
        int: Number of chunks successfully ingested
    """
    print(f"\n📥 Ingesting {len(chunks_data)} chunks into ChromaDB...")
    print(f"   Batch size: {batch_size}")
    
    total_ingested = 0
    failed = 0
    
    # Process in batches
    for i in range(0, len(chunks_data), batch_size):
        batch = chunks_data[i:i + batch_size]
        
        try:
            # Extract batch data
            ids = [chunk['id'] for chunk in batch]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            total_ingested += len(batch)
            
            # Progress update
            progress = (i + len(batch)) / len(chunks_data) * 100
            print(f"   Progress: {progress:.1f}% ({total_ingested}/{len(chunks_data)} chunks)")
            
        except Exception as e:
            print(f"⚠️  Batch {i//batch_size + 1} failed: {e}")
            failed += len(batch)
            continue
    
    print(f"\n✅ Ingestion complete!")
    print(f"   Successfully ingested: {total_ingested}")
    if failed > 0:
        print(f"   Failed: {failed}")
    
    return total_ingested


def verify_ingestion(collection: chromadb.Collection):
    """
    Verify that ingestion was successful.
    
    Args:
        collection: ChromaDB collection to verify
    """
    print(f"\n🔍 Verifying ingestion...")
    
    try:
        count = collection.count()
        print(f"   ✅ Collection contains {count:,} documents")
        
        # Sample a few documents
        if count > 0:
            sample = collection.peek(limit=3)
            print(f"\n📄 Sample documents:")
            for i, (doc_id, doc, meta) in enumerate(zip(
                sample['ids'],
                sample['documents'],
                sample['metadatas']
            )):
                print(f"\n   Document {i+1}:")
                print(f"      ID: {doc_id}")
                print(f"      Case: {meta.get('case_number', 'N/A')}")
                print(f"      Date: {meta.get('date', 'N/A')}")
                print(f"      Text: {doc[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest Indian Supreme Court judgments into ChromaDB"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/parquet',
        help='Directory containing parquet files (default: data/parquet)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1500,
        help='Chunk size in characters (default: 1500)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Overlap between chunks in characters (default: 200)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for ingestion (default: 100)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Delete existing collection and start fresh'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  INDIAN SUPREME COURT JUDGMENTS - CHROMADB INGESTION")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Step 1: Load parquet files
        print("\n[1/4] Loading parquet files...")
        df = load_all_parquets(args.data_dir)
        print_stats(df)
        
        # Step 2: Chunk judgments
        print("\n[2/4] Chunking judgments...")
        chunks_data = chunk_judgment_dataframe(
            df,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        print_chunk_stats(chunks_data)
        
        # Step 3: Initialize ChromaDB
        print("\n[3/4] Initializing ChromaDB...")
        collection = initialize_chromadb(reset=args.reset)
        
        # Step 4: Ingest chunks
        print("\n[4/4] Ingesting chunks...")
        ingested_count = ingest_chunks_to_chroma(
            chunks_data,
            collection,
            batch_size=args.batch_size
        )
        
        # Verify ingestion
        verify_ingestion(collection)
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("  INGESTION SUMMARY")
        print("=" * 70)
        print(f"✅ Total judgments processed: {len(df):,}")
        print(f"✅ Total chunks created: {len(chunks_data):,}")
        print(f"✅ Total chunks ingested: {ingested_count:,}")
        print(f"✅ Time elapsed: {elapsed_time:.1f} seconds")
        print(f"✅ Collection: {COLLECTION_NAME}")
        print(f"✅ Database: {CHROMA_DB_PATH}")
        print("=" * 70)
        
        print("\n🎉 Ingestion pipeline completed successfully!")
        print(f"\n💡 Test retrieval with: python3 src/ingest/test_retrieval.py")
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print(f"\n💡 Download parquet files first:")
        print(f"   python3 src/ingest/download.py --years 2018 2019 2020")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
