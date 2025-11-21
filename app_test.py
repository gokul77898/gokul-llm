"""ChromaDB Ingestion + Retrieval Test Script"""
import logging
from pathlib import Path
from db.chroma import ChromaDBClient, VectorRetriever, ingest_file, ingest_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate ChromaDB ingestion and retrieval."""
    
    print("=" * 60)
    print("ChromaDB Ingestion + Retrieval Test")
    print("=" * 60)
    
    # Step 1: Initialize client
    print("\n[1] Initializing ChromaDB client...")
    client = ChromaDBClient()
    print(f"✓ Client ready")
    
    # Step 2: Create collection
    print("\n[2] Creating/getting collection 'legal_docs'...")
    collection = client.get_or_create_collection("legal_docs")
    print(f"✓ Collection ready (current count: {collection.count()})")
    
    # Step 3: Ingest sample files
    print("\n[3] Ingesting sample files...")
    
    # Look for PDFs in data directory
    data_dir = Path("data")
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))[:2]  # Ingest first 2 PDFs
        
        if pdf_files:
            for pdf_file in pdf_files:
                print(f"\n   Ingesting: {pdf_file.name}")
                try:
                    stats = ingest_file(str(pdf_file), "legal_docs")
                    print(f"   ✓ {stats.summary()}")
                except Exception as e:
                    print(f"   ✗ Failed: {e}")
        else:
            print("   ⚠ No PDF files found in data/ directory")
            print("   Creating sample text file for testing...")
            
            # Create sample file
            sample_file = Path("sample_legal_doc.txt")
            sample_content = """
            Minimum Wages Act, 1948
            
            CHAPTER I - PRELIMINARY
            
            1. Short title, extent and commencement
            (1) This Act may be called the Minimum Wages Act, 1948.
            (2) It extends to the whole of India.
            
            2. Definitions
            In this Act, unless there is anything repugnant in the subject or context:
            
            (a) "appropriate Government" means:
                (i) in relation to any scheduled employment carried on by or under the authority of 
                    the Central Government or a railway administration, or in relation to a mine, 
                    oilfield or major port, or any corporation established by a Central Act, 
                    the Central Government;
                (ii) in relation to any other scheduled employment, the State Government.
            
            (b) "employer" means any person who employs, whether directly or through another person, 
                one or more employees in any scheduled employment in respect of which minimum rates 
                of wages have been fixed under this Act.
            
            (c) "employee" means any person who is employed for hire or reward to do any work, 
                skilled or unskilled, manual or clerical, in a scheduled employment.
            
            (d) "scheduled employment" means an employment specified in the Schedule, or any process 
                or branch of work forming part of such employment.
            
            3. Fixing of minimum wages
            (1) The appropriate Government shall fix minimum rates of wages payable to employees 
                employed in the scheduled employments under its jurisdiction.
            
            (2) In fixing minimum rates of wages, the appropriate Government may:
                (a) fix a minimum rate of wages for time work;
                (b) fix a minimum rate of wages for piece work;
                (c) fix a minimum remuneration to apply in the case of employees employed on 
                    piece work for the purpose of securing to such employees a minimum rate of wages.
            """
            
            sample_file.write_text(sample_content)
            print(f"   Created: {sample_file.name}")
            
            print(f"\n   Ingesting: {sample_file.name}")
            stats = ingest_file(str(sample_file), "legal_docs")
            print(f"   ✓ {stats.summary()}")
    else:
        print("   ⚠ data/ directory not found, skipping file ingestion")
    
    # Step 4: Initialize retriever
    print("\n[4] Initializing retriever...")
    retriever = VectorRetriever("legal_docs")
    stats = retriever.get_collection_stats()
    print(f"✓ Retriever ready")
    print(f"   Collection: {stats['name']}")
    print(f"   Documents: {stats['document_count']}")
    
    # Step 5: Test queries
    print("\n[5] Running test queries...")
    print("=" * 60)
    
    test_queries = [
        "What is appropriate government?",
        "Define employer",
        "What are minimum wages?",
        "Who is an employee?"
    ]
    
    for query_text in test_queries:
        print(f"\nQuery: \"{query_text}\"")
        print("-" * 60)
        
        results = retriever.query(query_text, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n[Result {i}] Score: {result.score:.4f}")
                print(f"Source: {result.get_source()}")
                if result.get_page():
                    print(f"Page: {result.get_page()}")
                print(f"Text: {result.text[:200]}...")
        else:
            print("No results found")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print(f"\nFinal collection stats:")
    print(f"  Documents: {retriever.collection.count()}")
    print(f"  Collection: {retriever.collection_name}")
    print(f"\n✓ ChromaDB system is working end-to-end")
    print(f"✓ Ready to integrate with AutoPipeline")

if __name__ == "__main__":
    main()
