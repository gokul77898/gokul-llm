"""Quickstart example for RAG indexer"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.common import init_logger
from src.data import create_sample_data
from src.rag.indexer import FAISSIndexer
from src.rag.eval import recall_at_k


def main():
    print("="*70)
    print("RAG INDEXER QUICKSTART")
    print("="*70)
    
    # Create sample data
    print("\n1. Creating sample data...")
    create_sample_data()
    
    logger = init_logger("rag_quickstart")
    
    # Load documents
    print("2. Loading documents...")
    documents = []
    with open("data/documents.jsonl", 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Limit for quickstart
                break
            documents.append(json.loads(line))
    
    print(f"   Loaded {len(documents)} documents")
    
    # Create indexer
    print("3. Creating FAISS indexer...")
    indexer = FAISSIndexer(embedding_dim=384)
    
    # Build index
    print("4. Building index...")
    indexer.build_index(documents)
    print(f"   Index built with {indexer.index.ntotal} vectors")
    
    # Save index
    print("5. Saving index...")
    index_path = Path("checkpoints/rag") / "quickstart.index"
    metadata_path = Path("checkpoints/rag") / "quickstart_metadata.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    indexer.save(str(index_path), str(metadata_path))
    print(f"   Saved to {index_path}")
    
    # Test retrieval
    print("\n6. Testing retrieval...")
    queries = [
        "What is contract law?",
        "Criminal law information",
        "Constitutional government"
    ]
    
    for query in queries:
        results = indexer.search(query, k=3)
        print(f"\n   Query: '{query}'")
        for i, result in enumerate(results[:2], 1):
            doc_text = result['document']['text'][:50] + "..."
            print(f"      {i}. {doc_text} (score: {result['score']:.4f})")
    
    # Simple evaluation
    print("\n7. Evaluation metrics:")
    eval_results = []
    eval_data = [
        {"query": "contract law", "relevant": ["doc1"]},
        {"query": "criminal law", "relevant": ["doc2"]},
    ]
    
    for item in eval_data:
        results = indexer.search(item["query"], k=5)
        retrieved_ids = [r['document'].get('id', f"doc_{r['index']}") for r in results]
        recall = recall_at_k(retrieved_ids, item["relevant"], 3)
        eval_results.append(recall)
    
    avg_recall = sum(eval_results) / len(eval_results) if eval_results else 0
    print(f"   Average Recall@3: {avg_recall:.4f}")
    
    print("\n" + "="*70)
    print("✓ RAG indexer quickstart completed successfully!")
    print(f"   Index saved to: {index_path}")
    print("="*70)


if __name__ == "__main__":
    main()
