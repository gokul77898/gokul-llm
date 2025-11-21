"""
ChromaDB Verification Test Script
Tests basic functionality with mock data
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from db.chroma.client import ChromaDBClient
from db.chroma.embeddings import EmbeddingModel

def test_chroma_db():
    """Test ChromaDB with mock data."""
    
    print("=" * 60)
    print("CHROMADB VERIFICATION TEST")
    print("=" * 60)
    
    try:
        # Step 1: Initialize client
        print("\n[1] Initializing ChromaDB client...")
        client = ChromaDBClient()
        print("✅ Client initialized")
        
        # Step 2: Create mock_test collection
        print("\n[2] Creating 'mock_test' collection...")
        collection = client.get_or_create_collection("mock_test")
        print(f"✅ Collection created (ID: {collection.name})")
        print(f"   Initial count: {collection.count()}")
        
        # Step 3: Prepare mock documents
        print("\n[3] Preparing mock documents...")
        mock_docs = [
            {
                "id": "doc_1",
                "text": "India has 28 states.",
                "metadata": {"source": "mock", "topic": "geography"}
            },
            {
                "id": "doc_2",
                "text": "The Minimum Wages Act was enacted in 1948.",
                "metadata": {"source": "mock", "topic": "law"}
            },
            {
                "id": "doc_3",
                "text": "Quantum computing uses qubits.",
                "metadata": {"source": "mock", "topic": "technology"}
            }
        ]
        print(f"✅ Prepared {len(mock_docs)} documents")
        
        # Step 4: Generate embeddings
        print("\n[4] Generating embeddings...")
        embedder = EmbeddingModel()
        texts = [doc["text"] for doc in mock_docs]
        embeddings = embedder.embed(texts)
        print(f"✅ Generated embeddings (dimension: {len(embeddings[0])})")
        
        # Step 5: Insert documents
        print("\n[5] Inserting documents into collection...")
        collection.add(
            ids=[doc["id"] for doc in mock_docs],
            documents=texts,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in mock_docs]
        )
        print(f"✅ Documents inserted")
        print(f"   Collection count: {collection.count()}")
        
        # Step 6: Query A - "How many states does India have?"
        print("\n[6] Testing Query A: 'How many states does India have?'")
        query_a = "How many states does India have?"
        query_a_embedding = embedder.embed_single(query_a)
        
        results_a = collection.query(
            query_embeddings=[query_a_embedding],
            n_results=1
        )
        
        if results_a['documents'] and results_a['documents'][0]:
            retrieved_text_a = results_a['documents'][0][0]
            expected_text_a = "India has 28 states."
            
            print(f"   Query: '{query_a}'")
            print(f"   Retrieved: '{retrieved_text_a}'")
            print(f"   Expected: '{expected_text_a}'")
            
            if retrieved_text_a == expected_text_a:
                print("   ✅ CORRECT - Query A passed")
                query_a_pass = True
            else:
                print("   ❌ INCORRECT - Query A failed")
                query_a_pass = False
        else:
            print("   ❌ FAILED - No results returned")
            query_a_pass = False
        
        # Step 7: Query B - "When was the Minimum Wages Act enacted?"
        print("\n[7] Testing Query B: 'When was the Minimum Wages Act enacted?'")
        query_b = "When was the Minimum Wages Act enacted?"
        query_b_embedding = embedder.embed_single(query_b)
        
        results_b = collection.query(
            query_embeddings=[query_b_embedding],
            n_results=1
        )
        
        if results_b['documents'] and results_b['documents'][0]:
            retrieved_text_b = results_b['documents'][0][0]
            expected_text_b = "The Minimum Wages Act was enacted in 1948."
            
            print(f"   Query: '{query_b}'")
            print(f"   Retrieved: '{retrieved_text_b}'")
            print(f"   Expected: '{expected_text_b}'")
            
            if retrieved_text_b == expected_text_b:
                print("   ✅ CORRECT - Query B passed")
                query_b_pass = True
            else:
                print("   ❌ INCORRECT - Query B failed")
                query_b_pass = False
        else:
            print("   ❌ FAILED - No results returned")
            query_b_pass = False
        
        # Step 8: Delete mock_test collection
        print("\n[8] Deleting 'mock_test' collection...")
        delete_success = client.delete_collection("mock_test")
        
        if delete_success:
            print("✅ Collection deleted")
        else:
            print("❌ Failed to delete collection")
        
        # Step 9: Verify deletion
        print("\n[9] Verifying deletion...")
        all_collections = client.list_collections()
        
        if "mock_test" not in all_collections:
            print("✅ Verified - 'mock_test' collection does not exist")
            deletion_verified = True
        else:
            print("❌ ERROR - 'mock_test' collection still exists")
            deletion_verified = False
        
        # Final Result
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Query A (India states):        {'✅ PASS' if query_a_pass else '❌ FAIL'}")
        print(f"Query B (Minimum Wages Act):   {'✅ PASS' if query_b_pass else '❌ FAIL'}")
        print(f"Collection Deletion:           {'✅ PASS' if deletion_verified else '❌ FAIL'}")
        print("=" * 60)
        
        all_passed = query_a_pass and query_b_pass and deletion_verified
        
        if all_passed:
            print("\n🎉 CHROMA DB WORKING ✔️")
            print("\nAll tests passed successfully!")
            print("- Collection creation: ✅")
            print("- Document insertion: ✅")
            print("- Vector search: ✅")
            print("- Query accuracy: ✅")
            print("- Collection deletion: ✅")
            return 0
        else:
            print("\n⚠️ ERROR ❌ — Issue found")
            print("\nSome tests failed. Please review the logs above.")
            return 1
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ CRITICAL ERROR")
        print("=" * 60)
        print(f"Exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\n⚠️ ERROR ❌ — Issue found")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = test_chroma_db()
    sys.exit(exit_code)
