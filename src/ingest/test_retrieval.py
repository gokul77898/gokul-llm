"""
ChromaDB Retrieval Test for Indian Supreme Court Judgments

Test script to query the vector database and verify retrieval quality.

Usage:
    python3 src/ingest/test_retrieval.py
    python3 src/ingest/test_retrieval.py --query "murder case IPC 302"
"""

import argparse
import sys
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ChromaDB configuration (must match chroma_ingest.py)
CHROMA_DB_PATH = "db_store/chroma"
COLLECTION_NAME = "supreme_court_judgments"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def initialize_retrieval() -> chromadb.Collection:
    """
    Initialize ChromaDB client and get collection for retrieval.
    
    Returns:
        chromadb.Collection: The collection to query
    """
    print(f"🔧 Initializing ChromaDB retrieval...")
    print(f"   Path: {CHROMA_DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    
    try:
        # Initialize client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Create embedding function
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Get collection
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        count = collection.count()
        print(f"   ✅ Collection loaded with {count:,} documents")
        
        return collection
        
    except Exception as e:
        print(f"❌ Failed to initialize retrieval: {e}")
        raise


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Query the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection to query
        query_text: Query string
        top_k: Number of results to return
        
    Returns:
        Dict with query results
    """
    print(f"\n🔍 Querying: '{query_text}'")
    print(f"   Retrieving top {top_k} matches...")
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        raise


def print_results(results: Dict[str, Any], query: str):
    """
    Print query results in a formatted way.
    
    Args:
        results: Results from ChromaDB query
        query: Original query text
    """
    print("\n" + "=" * 70)
    print(f"  QUERY RESULTS")
    print("=" * 70)
    print(f"Query: '{query}'")
    print(f"Results: {len(results['ids'][0])}")
    print("=" * 70)
    
    for i, (doc_id, doc, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n📄 Result {i+1}:")
        print(f"   ID: {doc_id}")
        print(f"   Distance: {distance:.4f}")
        print(f"\n   📋 Metadata:")
        print(f"      Case Number: {metadata.get('case_number', 'N/A')}")
        print(f"      Judges: {metadata.get('judges', 'N/A')}")
        print(f"      Date: {metadata.get('date', 'N/A')}")
        print(f"      Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")
        print(f"\n   📝 Text Preview:")
        preview = doc[:300] if len(doc) > 300 else doc
        print(f"      {preview}...")
        print("-" * 70)


def run_test_queries(collection: chromadb.Collection):
    """
    Run a set of predefined test queries.
    
    Args:
        collection: ChromaDB collection to query
    """
    test_queries = [
        "murder case IPC 302",
        "Supreme Court constitutional law",
        "right to equality Article 14",
        "criminal procedure code",
        "civil appeal judgment"
    ]
    
    print("\n" + "=" * 70)
    print("  RUNNING TEST QUERIES")
    print("=" * 70)
    
    for query in test_queries:
        results = query_collection(collection, query, top_k=2)
        
        print(f"\n✅ Query: '{query}'")
        print(f"   Top result case: {results['metadatas'][0][0].get('case_number', 'N/A')}")
        print(f"   Distance: {results['distances'][0][0]:.4f}")


def main():
    """Main entry point for retrieval testing."""
    parser = argparse.ArgumentParser(
        description="Test retrieval from Indian Supreme Court judgments ChromaDB"
    )
    parser.add_argument(
        '--query',
        type=str,
        default='murder case IPC 302',
        help='Query text (default: "murder case IPC 302")'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of results to retrieve (default: 3)'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run multiple test queries'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize retrieval
        collection = initialize_retrieval()
        
        if collection.count() == 0:
            print("\n⚠️  Collection is empty!")
            print("💡 Run ingestion first: python3 src/ingest/chroma_ingest.py")
            sys.exit(1)
        
        if args.test_mode:
            # Run test queries
            run_test_queries(collection)
        else:
            # Run single query
            results = query_collection(collection, args.query, args.top_k)
            print_results(results, args.query)
        
        print("\n✅ Retrieval test completed successfully!")
        sys.exit(0)
        
    except ValueError as e:
        print(f"\n❌ Collection not found: {e}")
        print(f"💡 Run ingestion first: python3 src/ingest/chroma_ingest.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
