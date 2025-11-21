"""
Test PDF retrieval from ChromaDB
"""
from chromadb import PersistentClient

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_docs"

def test_retrieval():
    client = PersistentClient(CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    
    print(f"📊 Collection: {COLLECTION_NAME}")
    print(f"   Total documents: {collection.count()}")
    
    # Test query
    query = "What is the law about?"
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print(f"\n🔍 Query: '{query}'")
    print(f"   Results: {len(results['documents'][0])}")
    print("\n" + "="*70)
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n📄 Result {i+1}:")
        print(f"   Source: {meta.get('source', 'N/A')}")
        print(f"   Page: {meta.get('page', 'N/A')}")
        print(f"   Text: {doc[:200]}...")
        print("-"*70)
    
    print("\n✅ Retrieval test successful!")

if __name__ == "__main__":
    test_retrieval()
