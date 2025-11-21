"""
Quick Start Example - Legal AI System

This script demonstrates basic usage of all four components:
1. Mamba Architecture
2. Transfer Learning
3. RAG System
4. Reinforcement Learning
"""

import torch
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.mamba.model import MambaModel
from src.mamba.tokenizer import DocumentTokenizer
from src.transfer.model import LegalTransferModel, LegalTaskType
from src.transfer.tokenizer import LegalTokenizer
from src.rag.document_store import Document, FAISSStore
from src.rag.retriever import LegalRetriever
from src.rag.generator import RAGGenerator
from src.rag.pipeline import RAGPipeline
from src.utils.data_loader import LegalDataLoader


def example_mamba():
    """Example: Mamba model for long document classification"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Mamba Architecture for Long Documents")
    print("="*60)
    
    # Create tokenizer and build vocabulary
    print("\n1. Creating tokenizer...")
    tokenizer = DocumentTokenizer(vocab_size=5000, max_length=256)
    
    sample_texts = [
        "This is a legal contract between two parties.",
        "The plaintiff filed a lawsuit against the defendant.",
        "Criminal law defines crimes and their punishments."
    ]
    tokenizer.build_vocab(sample_texts)
    
    # Create model
    print("2. Creating Mamba model...")
    model = MambaModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=8,
        num_classes=3
    )
    
    print(f"   Model parameters: {model.get_num_parameters():,}")
    
    # Tokenize and forward pass
    print("3. Running inference...")
    text = "This is a test legal document about contracts."
    encoded = tokenizer.encode(text, return_tensors=True)
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            chunk_boundaries=encoded.chunk_boundaries,
            task="classification"
        )
    
    predictions = torch.argmax(outputs['logits'], dim=-1)
    print(f"   Input: {text}")
    print(f"   Predicted class: {predictions.item()}")
    print("   ✓ Mamba model works!")


def example_transfer():
    """Example: Transfer learning for legal classification"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Transfer Learning for Legal Data")
    print("="*60)
    
    # Create tokenizer
    print("\n1. Creating legal tokenizer...")
    tokenizer = LegalTokenizer(base_model="bert-base-uncased")
    
    # Create model
    print("2. Creating transfer model...")
    model = LegalTransferModel(
        model_name="bert-base-uncased",
        task=LegalTaskType.CLASSIFICATION,
        num_labels=3
    )
    
    model.resize_token_embeddings(tokenizer.vocab_size)
    print(f"   Model parameters: {model.get_num_parameters():,}")
    
    # Inference
    print("3. Running inference...")
    text = "The case Smith v. Jones, 123 U.S. 456, was filed on 12/25/2023."
    
    # Preprocess and tokenize
    preprocessed = tokenizer.preprocess_legal_text(text)
    encoded = tokenizer.encode(preprocessed, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    predictions = torch.argmax(outputs['logits'], dim=-1)
    print(f"   Input: {text}")
    print(f"   Preprocessed: {preprocessed}")
    print(f"   Predicted class: {predictions.item()}")
    print("   ✓ Transfer model works!")


def example_rag():
    """Example: RAG system for legal Q&A"""
    print("\n" + "="*60)
    print("EXAMPLE 3: RAG System for Legal Q&A")
    print("="*60)
    
    # Create sample documents
    print("\n1. Creating document store...")
    documents = [
        Document(
            "A contract is a legally binding agreement between two or more parties.",
            metadata={'category': 'contract_law'}
        ),
        Document(
            "Tort law deals with civil wrongs and provides remedies for damages.",
            metadata={'category': 'tort_law'}
        ),
        Document(
            "Criminal law defines crimes and establishes punishments for offenses.",
            metadata={'category': 'criminal_law'}
        )
    ]
    
    # Create document store
    store = FAISSStore(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    store.add_documents(documents)
    
    # Create retriever
    print("2. Creating retriever...")
    retriever = LegalRetriever(document_store=store, top_k=2)
    
    # Create generator (using small model for demo)
    print("3. Creating generator...")
    generator = RAGGenerator(model_name="gpt2")
    
    # Create pipeline
    print("4. Creating RAG pipeline...")
    pipeline = RAGPipeline(store, retriever, generator)
    
    # Query
    print("5. Running query...")
    query = "What is a contract?"
    result = pipeline.query(query, top_k=2)
    
    print(f"\n   Query: {query}")
    print(f"   Retrieved {result['num_docs_retrieved']} documents")
    print(f"   Answer: {result['answer'][:200]}...")
    print("   ✓ RAG system works!")


def example_rl():
    """Example: RL environment setup"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Reinforcement Learning Setup")
    print("="*60)
    
    from src.rl.environment import LegalTaskEnvironment, LegalTaskType
    from src.rl.rewards import RewardCalculator
    
    # Create base model
    print("\n1. Creating base model for RL environment...")
    model = MambaModel(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        num_heads=4
    )
    
    tokenizer = DocumentTokenizer(vocab_size=1000)
    
    # Create environment
    print("2. Creating RL environment...")
    env = LegalTaskEnvironment(
        model=model,
        tokenizer=tokenizer,
        task_type=LegalTaskType.SUMMARIZATION
    )
    
    # Create reward calculator
    print("3. Creating reward calculator...")
    reward_calc = RewardCalculator()
    
    # Test environment
    print("4. Testing environment...")
    observation, info = env.reset()
    
    print(f"   Observation shape: {observation.shape}")
    print(f"   Action space: {env.action_space}")
    
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"   Reward: {reward:.4f}")
    print(f"   Terminated: {terminated}")
    print("   ✓ RL environment works!")


def example_data_loading():
    """Example: Data loading utilities"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Data Loading Utilities")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    data = LegalDataLoader.create_sample_data(num_samples=50, num_classes=3)
    
    print(f"   Created {len(data)} sample documents")
    print(f"   Sample: {data[0]['text'][:100]}...")
    
    # Save to file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(data, f)
        temp_file = f.name
    
    # Load data
    print("2. Loading data...")
    loader = LegalDataLoader(data_dir=os.path.dirname(temp_file))
    documents = loader.load_json(os.path.basename(temp_file))
    
    print(f"   Loaded {len(documents)} documents")
    
    # Split data
    print("3. Splitting data...")
    train, val, test = loader.split_data(documents, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Clean up
    os.unlink(temp_file)
    
    print("   ✓ Data loading works!")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("LEGAL AI SYSTEM - QUICK START EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates all four core components:")
    print("  1. Mamba Architecture (Long Document Processing)")
    print("  2. Transfer Learning (Legal-Specific Fine-tuning)")
    print("  3. RAG System (Retrieval-Augmented Generation)")
    print("  4. Reinforcement Learning (Policy Optimization)")
    print("  5. Data Loading (Utilities)")
    
    try:
        example_mamba()
    except Exception as e:
        print(f"   ✗ Error in Mamba example: {e}")
    
    try:
        example_transfer()
    except Exception as e:
        print(f"   ✗ Error in Transfer example: {e}")
    
    try:
        example_rag()
    except Exception as e:
        print(f"   ✗ Error in RAG example: {e}")
    
    try:
        example_rl()
    except Exception as e:
        print(f"   ✗ Error in RL example: {e}")
    
    try:
        example_data_loading()
    except Exception as e:
        print(f"   ✗ Error in Data Loading example: {e}")
    
    print("\n" + "="*60)
    print("QUICK START COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run training scripts in scripts/ directory")
    print("  2. Customize configs in configs/ directory")
    print("  3. Run tests with: pytest tests/ -v")
    print("  4. Check README.md for detailed documentation")
    print("\n")


if __name__ == "__main__":
    main()
