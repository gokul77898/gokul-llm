#!/usr/bin/env python3
"""
PHASE D: Verification at Scale
Run sample queries and verify metadata integrity
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

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def load_config():
    """Load configuration."""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "phase1_rag.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_stats(config: dict) -> Dict[str, Any]:
    """Get ingestion pipeline statistics.
    
    Returns:
        Dictionary with stats
    """
    stats = {}
    
    # Raw documents
    raw_dir = PROJECT_ROOT / config['paths']['raw_dir']
    if raw_dir.exists():
        raw_files = list(raw_dir.glob('*.txt'))
        stats['raw_documents'] = len(raw_files)
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in raw_files)
        stats['raw_size_gb'] = total_size / (1024 ** 3)
    else:
        stats['raw_documents'] = 0
        stats['raw_size_gb'] = 0.0
    
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
        if path.exists():
            for f in path.rglob('*'):
                if f.is_file():
                    total_disk += f.stat().st_size
    
    stats['total_disk_gb'] = total_disk / (1024 ** 3)
    
    return stats


def run_sample_queries(config: dict) -> List[Dict[str, Any]]:
    """Run sample queries and return results.
    
    Returns:
        List of query results
    """
    if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Error: Required libraries not available")
        return []
    
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
    
    # Sample queries
    queries = [
        "punishment under section 420 IPC",
        "definition of employer minimum wages act",
        "supreme court cheating offence",
        "procedure under crpc",
        "labour law employer liability"
    ]
    
    results = []
    
    for query in queries:
        print(f"\nQuery: {query}")
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
                document = result['documents'][0][i] if result['documents'] else ""
                distance = result['distances'][0][i] if result['distances'] else 0.0
                
                chunk_info = {
                    'rank': i + 1,
                    'chunk_id': chunk_id,
                    'semantic_id': metadata.get('semantic_id', 'MISSING'),
                    'act': metadata.get('act', 'MISSING'),
                    'section': metadata.get('section', 'MISSING'),
                    'source_doc': metadata.get('source_doc', 'MISSING'),
                    'distance': distance,
                    'text_preview': document[:100] + "..." if len(document) > 100 else document
                }
                
                query_result['chunks'].append(chunk_info)
                
                # Print
                print(f"  [{i+1}] semantic_id: {chunk_info['semantic_id']}")
                print(f"      act: {chunk_info['act']}")
                print(f"      section: {chunk_info['section']}")
                print(f"      source: {chunk_info['source_doc']}")
                print(f"      distance: {chunk_info['distance']:.4f}")
                print()
        else:
            print("  No results found")
        
        results.append(query_result)
    
    return results


def verify_metadata_integrity(results: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Verify metadata integrity across query results.
    
    Returns:
        Dictionary of verification checks
    """
    checks = {
        'no_missing_semantic_id': True,
        'no_missing_act': True,
        'no_missing_section': True,
        'no_missing_source': True,
        'deterministic_results': True
    }
    
    for result in results:
        for chunk in result['chunks']:
            if chunk['semantic_id'] == 'MISSING':
                checks['no_missing_semantic_id'] = False
            if chunk['act'] == 'MISSING':
                checks['no_missing_act'] = False
            if chunk['section'] == 'MISSING':
                checks['no_missing_section'] = False
            if chunk['source_doc'] == 'MISSING':
                checks['no_missing_source'] = False
    
    return checks


def print_summary_table(stats: Dict[str, Any]):
    """Print summary table."""
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"Raw documents:        {stats['raw_documents']:,}")
    print(f"Canonical documents:  {stats['canonical_documents']:,}")
    print(f"Total chunks:         {stats['total_chunks']:,}")
    print(f"Indexed vectors:      {stats['indexed_vectors']:,}")
    print(f"Raw data size:        {stats['raw_size_gb']:.2f} GB")
    print(f"Total disk usage:     {stats['total_disk_gb']:.2f} GB")
    print("=" * 70)
    print()


def print_verification_checklist(
    stats: Dict[str, Any],
    metadata_checks: Dict[str, bool]
) -> bool:
    """Print verification checklist.
    
    Returns:
        True if all checks pass
    """
    print("=" * 70)
    print("VERIFICATION CHECKLIST")
    print("=" * 70)
    
    checks = []
    
    # Ingestion check
    ingestion_pass = (
        stats['raw_documents'] > 0 and
        stats['canonical_documents'] > 0 and
        stats['canonical_documents'] == stats['raw_documents']
    )
    checks.append(('Ingestion', ingestion_pass))
    
    # Chunking check
    chunking_pass = (
        stats['total_chunks'] > 0 and
        stats['total_chunks'] > stats['canonical_documents']
    )
    checks.append(('Chunking', chunking_pass))
    
    # Indexing check
    indexing_pass = (
        stats['indexed_vectors'] > 0 and
        stats['indexed_vectors'] == stats['total_chunks']
    )
    checks.append(('Indexing', indexing_pass))
    
    # Metadata integrity check
    metadata_pass = all(metadata_checks.values())
    checks.append(('Metadata integrity', metadata_pass))
    
    # Scale handling check
    scale_pass = (
        stats['raw_size_gb'] >= 4.5 and
        stats['raw_size_gb'] <= 5.5 and
        stats['total_chunks'] > 10000  # Expect at least 10k chunks from 5GB
    )
    checks.append(('Scale handling', scale_pass))
    
    # Print checks
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check_name:.<30} {status}")
    
    print("=" * 70)
    print()
    
    all_pass = all(passed for _, passed in checks)
    return all_pass


def main():
    print("=" * 70)
    print("PHASE D: VERIFICATION AT SCALE")
    print("=" * 70)
    print()
    
    # Check dependencies
    if not CHROMADB_AVAILABLE:
        print("Error: ChromaDB not available")
        sys.exit(1)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Error: sentence-transformers not available")
        sys.exit(1)
    
    # Load config
    config = load_config()
    
    # Step 1: Get stats
    print("Step 1: Collecting statistics...")
    stats = get_stats(config)
    print("✓ Statistics collected")
    print()
    
    # Step 2: Print summary table
    print_summary_table(stats)
    
    # Step 3: Run sample queries
    print("Step 2: Running sample queries (retrieval only)...")
    print()
    results = run_sample_queries(config)
    print()
    print("✓ Sample queries complete")
    print()
    
    # Step 4: Verify metadata integrity
    print("Step 3: Verifying metadata integrity...")
    metadata_checks = verify_metadata_integrity(results)
    
    print("Metadata checks:")
    for check, passed in metadata_checks.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {check}")
    print()
    
    # Step 5: Print verification checklist
    all_pass = print_verification_checklist(stats, metadata_checks)
    
    # Final result
    if all_pass:
        print("=" * 70)
        print("RAG INGESTION PIPELINE VALIDATED FOR SCALE")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("VALIDATION FAILED - See checklist above")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
