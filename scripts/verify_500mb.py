#!/usr/bin/env python3
"""
Verification Script for 500MB Supreme Court Dataset
Validates ingestion, chunking, indexing, and runs sample queries
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ERROR: ChromaDB not available")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("ERROR: sentence-transformers not available")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not available")
    sys.exit(1)


def load_config():
    """Load configuration."""
    config_path = PROJECT_ROOT / "configs" / "phase1_rag.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    if path.exists():
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def collect_stats(config: dict) -> Dict[str, Any]:
    """Collect pipeline statistics."""
    stats = {}
    
    # Raw documents
    raw_dir = PROJECT_ROOT / config['paths']['raw_dir']
    if raw_dir.exists():
        raw_files = list(raw_dir.glob('*.txt'))
        stats['raw_documents'] = len(raw_files)
        raw_size = sum(f.stat().st_size for f in raw_files)
        stats['raw_size_mb'] = raw_size / (1024 * 1024)
    else:
        stats['raw_documents'] = 0
        stats['raw_size_mb'] = 0.0
    
    # Canonical documents
    docs_dir = PROJECT_ROOT / config['paths']['documents_dir']
    if docs_dir.exists():
        doc_files = list(docs_dir.glob('*.json'))
        stats['canonical_documents'] = len(doc_files)
    else:
        stats['canonical_documents'] = 0
    
    # Chunks
    chunks_dir = PROJECT_ROOT / config['paths']['chunks_dir']
    if chunks_dir.exists():
        chunk_files = list(chunks_dir.glob('*.json'))
        stats['total_chunks'] = len(chunk_files)
    else:
        stats['total_chunks'] = 0
    
    # Indexed vectors
    chromadb_dir = PROJECT_ROOT / config['paths']['chromadb_dir']
    if chromadb_dir.exists():
        try:
            client = chromadb.PersistentClient(
                path=str(chromadb_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            collection_name = config['retrieval']['collection_name']
            collection = client.get_collection(name=collection_name)
            stats['indexed_vectors'] = collection.count()
        except Exception as e:
            print(f"Warning: Could not get vector count: {e}")
            stats['indexed_vectors'] = 0
    else:
        stats['indexed_vectors'] = 0
    
    # Disk usage
    total_disk = 0
    for path_key in ['documents_dir', 'chunks_dir', 'chromadb_dir']:
        path = PROJECT_ROOT / config['paths'][path_key]
        total_disk += get_directory_size(path)
    
    stats['total_disk_mb'] = total_disk / (1024 * 1024)
    
    return stats


def run_queries(config: dict) -> List[Dict[str, Any]]:
    """Run sample retrieval queries."""
    
    # Initialize ChromaDB
    chromadb_dir = PROJECT_ROOT / config['paths']['chromadb_dir']
    client = chromadb.PersistentClient(
        path=str(chromadb_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection_name = config['retrieval']['collection_name']
    collection = client.get_collection(name=collection_name)
    
    # Initialize encoder
    encoder_model = config['encoder']['model_name']
    print(f"Loading encoder: {encoder_model}...")
    model = SentenceTransformer(encoder_model)
    print()
    
    # Sample queries
    queries = [
        "supreme court cheating offence",
        "criminal liability judgment",
        "procedural law supreme court",
        "mens rea judgment",
        "constitutional interpretation supreme court"
    ]
    
    results = []
    
    for query in queries:
        print(f"Query: {query}")
        print("-" * 70)
        
        # Encode query
        query_embedding = model.encode([query])[0].tolist()
        
        # Retrieve
        top_k = 5
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        query_result = {
            'query': query,
            'chunks': []
        }
        
        if result['ids'] and result['ids'][0]:
            for i in range(len(result['ids'][0])):
                chunk_id = result['ids'][0][i]
                metadata = result['metadatas'][0][i] if result['metadatas'] else {}
                distance = result['distances'][0][i] if result['distances'] else 0.0
                
                chunk_info = {
                    'rank': i + 1,
                    'semantic_id': metadata.get('semantic_id', 'MISSING'),
                    'act': metadata.get('act', 'MISSING'),
                    'court': metadata.get('court', 'MISSING'),
                    'year': metadata.get('year', 'MISSING'),
                    'distance': distance
                }
                
                query_result['chunks'].append(chunk_info)
                
                # Print
                print(f"  [{i+1}] semantic_id: {chunk_info['semantic_id']}")
                print(f"      act: {chunk_info['act']}")
                print(f"      court: {chunk_info['court']}")
                print(f"      year: {chunk_info['year']}")
                print(f"      distance: {chunk_info['distance']:.4f}")
                print()
        else:
            print("  No results found")
        
        results.append(query_result)
    
    return results


def verify_metadata(results: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Verify metadata integrity."""
    checks = {
        'no_missing_semantic_id': True,
        'no_missing_act': True,
        'no_missing_court': True,
        'no_missing_year': True,
        'semantic_id_format': True
    }
    
    for result in results:
        for chunk in result['chunks']:
            if chunk['semantic_id'] == 'MISSING':
                checks['no_missing_semantic_id'] = False
            if chunk['act'] == 'MISSING':
                checks['no_missing_act'] = False
            if chunk['court'] == 'MISSING':
                checks['no_missing_court'] = False
            if chunk['year'] == 'MISSING':
                checks['no_missing_year'] = False
            
            # Check semantic_id format: SupremeCourt_<YEAR>_<INDEX>
            semantic_id = chunk['semantic_id']
            if semantic_id != 'MISSING':
                if not semantic_id.startswith('SupremeCourt_'):
                    checks['semantic_id_format'] = False
    
    return checks


def print_summary_table(stats: Dict[str, Any]):
    """Print summary table."""
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"Total raw docs:       {stats['raw_documents']:,}")
    print(f"Total canonical docs: {stats['canonical_documents']:,}")
    print(f"Total chunks:         {stats['total_chunks']:,}")
    print(f"Total vectors indexed:{stats['indexed_vectors']:,}")
    print(f"Raw data size:        {stats['raw_size_mb']:.2f} MB")
    print(f"Disk usage:           {stats['total_disk_mb']:.2f} MB")
    print("=" * 70)
    print()


def print_checklist(stats: Dict[str, Any], metadata_checks: Dict[str, bool]) -> bool:
    """Print PASS/FAIL checklist."""
    print("=" * 70)
    print("PASS/FAIL CHECKLIST")
    print("=" * 70)
    
    checks = []
    
    # Ingestion
    ingestion_pass = (
        stats['raw_documents'] > 0 and
        stats['canonical_documents'] > 0 and
        stats['canonical_documents'] == stats['raw_documents']
    )
    checks.append(('Ingestion', ingestion_pass))
    
    # Chunking
    chunking_pass = (
        stats['total_chunks'] > 0 and
        stats['total_chunks'] > stats['canonical_documents']
    )
    checks.append(('Chunking', chunking_pass))
    
    # Indexing
    indexing_pass = (
        stats['indexed_vectors'] > 0 and
        stats['indexed_vectors'] == stats['total_chunks']
    )
    checks.append(('Indexing', indexing_pass))
    
    # Metadata integrity
    metadata_pass = all(metadata_checks.values())
    checks.append(('Metadata integrity', metadata_pass))
    
    # Dataset size
    size_pass = 400 <= stats['raw_size_mb'] <= 550
    checks.append(('Dataset size (400-550 MB)', size_pass))
    
    # Print
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check_name:.<40} {status}")
    
    print("=" * 70)
    print()
    
    # Print errors if any
    if not all(passed for _, passed in checks):
        print("ERRORS FOUND:")
        for check_name, passed in checks:
            if not passed:
                print(f"  ✗ {check_name}")
        print()
    
    all_pass = all(passed for _, passed in checks)
    return all_pass


def main():
    print("=" * 70)
    print("PHASE D: VERIFICATION (500MB Dataset)")
    print("=" * 70)
    print()
    
    # Load config
    config = load_config()
    
    # Step 1: Collect stats
    print("Step 1: Collecting statistics...")
    stats = collect_stats(config)
    print("✓ Statistics collected")
    print()
    
    # Print summary
    print_summary_table(stats)
    
    # Step 2: Run queries
    print("Step 2: Running retrieval-only queries...")
    print()
    results = run_queries(config)
    print()
    print("✓ Queries complete")
    print()
    
    # Step 3: Verify metadata
    print("Step 3: Verifying metadata integrity...")
    metadata_checks = verify_metadata(results)
    
    print("Metadata checks:")
    for check, passed in metadata_checks.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {check}")
    print()
    
    # Step 4: Print checklist
    all_pass = print_checklist(stats, metadata_checks)
    
    # Final result
    if all_pass:
        print("=" * 70)
        print("RAG INGESTION PIPELINE VALIDATED FOR SCALE (500MB)")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("VALIDATION FAILED - See errors above")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
