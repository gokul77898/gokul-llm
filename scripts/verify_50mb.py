#!/usr/bin/env python3
"""
Verification script for 50MB RAG pipeline validation.
Checks ingestion, chunking, indexing correctness.
NO training, NO LLMs, retrieval-only verification.
"""

import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.storage.filesystem import DocumentStorage
from src.rag.storage.chunk_storage import ChunkStorage

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def collect_stats():
    """Collect statistics from all pipeline stages."""
    print_section("PHASE D: VERIFICATION - COLLECTING STATISTICS")
    
    stats = {}
    
    # Raw files
    raw_dir = project_root / "data" / "rag" / "raw"
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.txt"))
        stats['raw_files'] = len(raw_files)
        total_size = sum(f.stat().st_size for f in raw_files)
        stats['raw_size_mb'] = total_size / (1024 * 1024)
    else:
        stats['raw_files'] = 0
        stats['raw_size_mb'] = 0.0
    
    # Canonical documents
    doc_storage = DocumentStorage(base_dir=project_root / "data" / "rag" / "documents")
    try:
        doc_stats = doc_storage.get_stats()
        stats['canonical_docs'] = doc_stats['total_documents']
        stats['canonical_versions'] = doc_stats['total_versions']
    except Exception as e:
        print(f"Warning: Could not get document stats: {e}")
        stats['canonical_docs'] = 0
        stats['canonical_versions'] = 0
    
    # Chunks
    chunk_storage = ChunkStorage(base_dir=project_root / "data" / "rag" / "chunks")
    try:
        chunk_stats = chunk_storage.get_stats()
        stats['total_chunks'] = chunk_stats['total_chunks']
        stats['unique_docs'] = chunk_stats['unique_documents']
    except Exception as e:
        print(f"Warning: Could not get chunk stats: {e}")
        stats['total_chunks'] = 0
        stats['unique_docs'] = 0
    
    # Indexed vectors
    chroma_dir = project_root / "data" / "rag" / "chroma"
    try:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name="legal_chunks")
        stats['indexed_vectors'] = collection.count()
    except Exception as e:
        print(f"Warning: Could not get vector count: {e}")
        stats['indexed_vectors'] = 0
    
    return stats

def print_stats_table(stats: dict):
    """Print statistics in table format."""
    print("\n" + "-" * 70)
    print("PIPELINE STATISTICS")
    print("-" * 70)
    print(f"{'Metric':<40} {'Count':<15} {'Status':<15}")
    print("-" * 70)
    
    # Raw dataset
    print(f"{'Raw text files':<40} {stats['raw_files']:<15} {'✓' if stats['raw_files'] > 0 else '✗':<15}")
    print(f"{'Dataset size (MB)':<40} {stats['raw_size_mb']:<15.2f} {'✓' if 40 <= stats['raw_size_mb'] <= 60 else '✗':<15}")
    
    # Ingestion
    print(f"{'Canonical documents':<40} {stats['canonical_docs']:<15} {'✓' if stats['canonical_docs'] > 0 else '✗':<15}")
    print(f"{'Document versions':<40} {stats['canonical_versions']:<15} {'✓' if stats['canonical_versions'] > 0 else '✗':<15}")
    
    # Chunking
    print(f"{'Total chunks':<40} {stats['total_chunks']:<15} {'✓' if stats['total_chunks'] > 0 else '✗':<15}")
    print(f"{'Unique documents chunked':<40} {stats['unique_docs']:<15} {'✓' if stats['unique_docs'] > 0 else '✗':<15}")
    
    # Indexing
    print(f"{'Indexed vectors':<40} {stats['indexed_vectors']:<15} {'✓' if stats['indexed_vectors'] > 0 else '✗':<15}")
    
    print("-" * 70)

def run_verification_queries():
    """Run retrieval-only queries to verify pipeline."""
    print_section("VERIFICATION QUERIES (RETRIEVAL ONLY)")
    
    queries = [
        "supreme court cheating offence",
        "criminal liability judgment",
        "mens rea supreme court",
        "procedural law judgment",
        "constitutional interpretation"
    ]
    
    chroma_dir = project_root / "data" / "rag" / "chroma"
    
    try:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name="legal_chunks")
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 70)
            
            results = collection.query(
                query_texts=[query],
                n_results=3,
                include=['metadatas', 'distances']
            )
            
            if results['metadatas'] and results['metadatas'][0]:
                for j, metadata in enumerate(results['metadatas'][0], 1):
                    semantic_id = metadata.get('semantic_id', 'MISSING')
                    court = metadata.get('court', 'MISSING')
                    year = metadata.get('year', 'MISSING')
                    distance = results['distances'][0][j] if results['distances'] else 'N/A'
                    
                    print(f"   Result {j}:")
                    print(f"     semantic_id: {semantic_id}")
                    print(f"     court: {court}")
                    print(f"     year: {year}")
                    print(f"     distance: {distance:.4f}" if isinstance(distance, float) else f"     distance: {distance}")
            else:
                print("   No results found")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n✗ ERROR running verification queries: {e}")
        return False
    
    return True

def verify_metadata_integrity():
    """Verify metadata consistency and correctness."""
    print_section("METADATA INTEGRITY VERIFICATION")
    
    issues = []
    
    chroma_dir = project_root / "data" / "rag" / "chroma"
    
    try:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name="legal_chunks")
        
        # Sample check: get 100 random chunks
        total_count = collection.count()
        sample_size = min(100, total_count)
        
        print(f"\nChecking {sample_size} sample chunks out of {total_count} total...")
        
        results = collection.get(
            limit=sample_size,
            include=['metadatas']
        )
        
        required_fields = ['semantic_id', 'court', 'year', 'doc_type']
        
        for i, metadata in enumerate(results['metadatas']):
            # Check required fields
            for field in required_fields:
                if field not in metadata:
                    issues.append(f"Chunk {i}: Missing field '{field}'")
            
            # Check semantic_id format
            if 'semantic_id' in metadata:
                semantic_id = metadata['semantic_id']
                if not semantic_id.startswith('SupremeCourt_'):
                    issues.append(f"Chunk {i}: Invalid semantic_id format: {semantic_id}")
            
            # Check court value
            if 'court' in metadata:
                if metadata['court'] != 'Supreme Court of India':
                    issues.append(f"Chunk {i}: Invalid court: {metadata['court']}")
            
            # Check year
            if 'year' in metadata:
                try:
                    year = int(metadata['year'])
                    if not (2010 <= year <= 2023):
                        issues.append(f"Chunk {i}: Year out of range: {year}")
                except (ValueError, TypeError):
                    issues.append(f"Chunk {i}: Invalid year format: {metadata['year']}")
        
        if issues:
            print("\n✗ METADATA ISSUES FOUND:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        else:
            print("\n✓ All metadata checks passed")
        
    except Exception as e:
        print(f"\n✗ ERROR during metadata verification: {e}")
        issues.append(str(e))
    
    return len(issues) == 0

def print_verification_checklist(stats: dict, queries_ok: bool, metadata_ok: bool):
    """Print final PASS/FAIL checklist."""
    print_section("VERIFICATION CHECKLIST")
    
    checks = [
        ("Dataset size (40-60 MB)", 40 <= stats['raw_size_mb'] <= 60),
        ("Raw files present", stats['raw_files'] > 0),
        ("Documents ingested", stats['canonical_docs'] > 0),
        ("Documents == Raw files", stats['canonical_docs'] == stats['raw_files']),
        ("Chunks created", stats['total_chunks'] > 0),
        ("Chunks > Documents", stats['total_chunks'] > stats['canonical_docs']),
        ("Vectors indexed", stats['indexed_vectors'] > 0),
        ("Vectors == Chunks", stats['indexed_vectors'] == stats['total_chunks']),
        ("Verification queries successful", queries_ok),
        ("Metadata integrity verified", metadata_ok),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Check':<50} {'Status':<20}")
    print("-" * 70)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<50} {status:<20}")
        if not passed:
            all_passed = False
    
    print("-" * 70)
    
    return all_passed

def main():
    """Main verification workflow."""
    print("\n" + "=" * 70)
    print("RAG PIPELINE VERIFICATION (50MB DATASET)")
    print("=" * 70)
    
    # Collect statistics
    stats = collect_stats()
    print_stats_table(stats)
    
    # Run verification queries
    queries_ok = run_verification_queries()
    
    # Verify metadata integrity
    metadata_ok = verify_metadata_integrity()
    
    # Print checklist
    all_passed = print_verification_checklist(stats, queries_ok, metadata_ok)
    
    # Final result
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nRAG INGESTION PIPELINE VALIDATED (50MB)")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nRAG INGESTION PIPELINE VALIDATION INCOMPLETE")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
