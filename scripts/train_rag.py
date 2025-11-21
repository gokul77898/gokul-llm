"""Training script for RAG system"""

import argparse
import yaml
from pathlib import Path

from src.rag.document_store import Document, FAISSStore
from src.rag.pipeline import create_legal_rag_pipeline, RAGTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_documents(num_docs: int = 100):
    """Create sample legal documents"""
    documents = []
    
    for i in range(num_docs):
        content = (
            f"Legal Document {i}: This document discusses contract law principles. "
            f"The plaintiff filed a case under Section {i % 20}. "
            f"The court held that parties must comply with statutory requirements. "
            f"This is an important precedent for future cases."
        )
        
        metadata = {
            'doc_id': f"doc_{i}",
            'category': ['contract', 'tort', 'criminal'][i % 3],
            'date': f"2024-01-{(i % 28) + 1:02d}",
            'source': f"Court_{i % 5}"
        }
        
        documents.append(Document(content=content, metadata=metadata))
    
    return documents


def main(args):
    """Main function"""
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create or load documents
    print("Loading documents...")
    if args.data_path:
        print(f"Loading documents from {args.data_path}")
        # TODO: Implement real document loading
        documents = create_sample_documents(500)
    else:
        print("Using sample documents...")
        documents = create_sample_documents(500)
    
    print(f"Loaded {len(documents)} documents")
    
    # Create RAG pipeline
    print("Creating RAG pipeline...")
    pipeline = create_legal_rag_pipeline(
        documents=documents,
        embedding_model=config['retriever']['embedding_model'],
        generation_model=config['generator']['model_name'],
        use_citations=config['generator'].get('use_citations', True),
        use_chain_of_thought=config['generator'].get('use_chain_of_thought', False)
    )
    
    # Save pipeline
    save_dir = config['training']['checkpoint_dir']
    print(f"Saving pipeline to {save_dir}...")
    pipeline.save(save_dir)
    
    # Create sample queries for evaluation
    sample_queries = [
        "What are the requirements for a valid contract?",
        "Explain tort law principles.",
        "What is the statute of limitations?",
        "How does criminal law define intent?",
        "What are the remedies for breach of contract?"
    ]
    
    # Test pipeline
    print("\nTesting RAG pipeline with sample queries...")
    for query in sample_queries[:3]:
        print(f"\nQuery: {query}")
        result = pipeline.query(query, top_k=3)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Retrieved {result['num_docs_retrieved']} documents")
    
    # Optional: Train with labeled data
    if args.train:
        print("\nTraining RAG pipeline...")
        trainer = RAGTrainer(pipeline=pipeline)
        
        # Create sample training data
        train_queries = sample_queries * 10  # Repeat for training
        train_answers = [
            "A valid contract requires offer, acceptance, and consideration."
        ] * len(train_queries)
        
        # Train retriever
        print("Training retriever...")
        trainer.train_retriever(
            queries=train_queries[:20],
            relevant_doc_ids=[['doc_0', 'doc_1'] for _ in range(20)],
            num_epochs=config['training'].get('num_epochs', 3)
        )
        
        # Evaluate
        print("Evaluating...")
        metrics = trainer.evaluate(
            eval_queries=sample_queries,
            eval_answers=train_answers[:len(sample_queries)]
        )
        
        print(f"\nEvaluation metrics: {metrics}")
    
    print("\nRAG system setup completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and train RAG system")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rag_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to document corpus'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the RAG components'
    )
    
    args = parser.parse_args()
    main(args)
