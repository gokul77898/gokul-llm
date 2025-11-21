"""
SFT Data Preparation from ChromaDB

Extracts QA pairs from ChromaDB collection and creates supervised fine-tuning dataset.

Usage:
    python -m src.training.data_prep --collection pdf_docs --out-dir data/ --top-k 3
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from chromadb import PersistentClient


class SFTDataGenerator:
    """Generate supervised fine-tuning data from ChromaDB documents."""
    
    QUESTION_TEMPLATES = [
        "What does this section say about {topic}?",
        "According to this document, what is {topic}?",
        "How is {topic} described in this text?",
        "What information is provided about {topic}?",
        "Can you explain {topic} based on this document?",
    ]
    
    INSTRUCTION = (
        "Answer concisely using the provided context. "
        "If the answer isn't present, respond 'NO_ANSWER_IN_DOCUMENTS'."
    )
    
    def __init__(self, chroma_path: str = "chroma_db", collection_name: str = "pdf_docs"):
        """Initialize data generator."""
        self.client = PersistentClient(chroma_path)
        self.collection = self.client.get_collection(collection_name)
        print(f"✅ Connected to ChromaDB collection: {collection_name}")
        print(f"   Total documents: {self.collection.count()}")
    
    def extract_noun_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """
        Extract potential topics/entities from text using simple heuristics.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of noun phrases/topics
        """
        # Simple heuristic: capitalized words/phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Common legal/technical terms
        legal_terms = re.findall(
            r'\b(?:section|article|clause|provision|act|law|regulation|statute|court|'
            r'punishment|offense|rights|liability|contract|agreement)\b',
            text,
            re.IGNORECASE
        )
        
        # Combine and deduplicate
        topics = list(set(capitalized + legal_terms))
        
        # Filter out very common words
        stopwords = {'The', 'This', 'That', 'Section', 'Article'}
        topics = [t for t in topics if t not in stopwords]
        
        return topics[:max_phrases]
    
    def generate_qa_pair(self, chunk: str, metadata: Dict) -> List[Dict[str, str]]:
        """
        Generate Q/A pairs from a chunk.
        
        Args:
            chunk: Text chunk
            metadata: Chunk metadata
            
        Returns:
            List of QA pairs
        """
        qa_pairs = []
        
        # Extract topics
        topics = self.extract_noun_phrases(chunk, max_phrases=3)
        
        if not topics:
            # Fallback: generic question
            topics = ["the main point"]
        
        for topic in topics:
            # Random question template
            template = random.choice(self.QUESTION_TEMPLATES)
            question = template.format(topic=topic)
            
            # Extract relevant answer (use first ~200 chars containing topic)
            answer = self._extract_answer(chunk, topic)
            
            qa_pair = {
                "instruction": self.INSTRUCTION,
                "input": f"Context: {chunk[:500]}...\n\nQuestion: {question}",
                "output": answer,
                "metadata": {
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", "unknown"),
                    "topic": topic
                }
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _extract_answer(self, text: str, topic: str) -> str:
        """Extract answer span containing the topic."""
        # Find sentences containing topic
        sentences = re.split(r'[.!?]+', text)
        
        for sent in sentences:
            if topic.lower() in sent.lower():
                return sent.strip()
        
        # Fallback: first sentence
        return sentences[0].strip() if sentences else "NO_ANSWER_IN_DOCUMENTS"
    
    def generate_dataset(
        self,
        top_k: int = 3,
        max_samples: int = 1000,
        val_split: float = 0.05
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate training and validation datasets.
        
        Args:
            top_k: Number of chunks to sample per document
            max_samples: Maximum number of samples to generate
            val_split: Validation set proportion
            
        Returns:
            Tuple of (train_data, val_data)
        """
        print(f"\n🔧 Generating dataset...")
        print(f"   Top-K chunks per doc: {top_k}")
        print(f"   Max samples: {max_samples}")
        print(f"   Val split: {val_split:.1%}")
        
        all_data = []
        
        # Get all documents
        total_docs = self.collection.count()
        batch_size = 100
        
        for offset in range(0, min(total_docs, max_samples * 2), batch_size):
            results = self.collection.get(
                limit=min(batch_size, total_docs - offset),
                offset=offset,
                include=['documents', 'metadatas']
            )
            
            for doc, meta in zip(results['documents'], results['metadatas']):
                # Generate QA pairs for this chunk
                qa_pairs = self.generate_qa_pair(doc, meta)
                all_data.extend(qa_pairs)
                
                if len(all_data) >= max_samples:
                    break
            
            if len(all_data) >= max_samples:
                break
            
            print(f"   Progress: {min(offset + batch_size, total_docs)}/{total_docs} docs processed...")
        
        # Shuffle and split
        random.shuffle(all_data)
        all_data = all_data[:max_samples]
        
        val_size = int(len(all_data) * val_split)
        val_data = all_data[:val_size]
        train_data = all_data[val_size:]
        
        print(f"\n✅ Dataset generated:")
        print(f"   Total samples: {len(all_data)}")
        print(f"   Training: {len(train_data)}")
        print(f"   Validation: {len(val_data)}")
        
        return train_data, val_data
    
    def save_jsonl(self, data: List[Dict], filepath: str):
        """Save dataset to JSONL format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                # Remove metadata for training file
                clean_item = {
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": item["output"]
                }
                f.write(json.dumps(clean_item, ensure_ascii=False) + '\n')
        
        print(f"   ✅ Saved: {filepath} ({len(data)} samples)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset from ChromaDB"
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='pdf_docs',
        help='ChromaDB collection name'
    )
    parser.add_argument(
        '--chroma-path',
        type=str,
        default='chroma_db',
        help='Path to ChromaDB'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/',
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Chunks per document to sample'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples to generate'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.05,
        help='Validation split ratio'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  SFT DATA PREPARATION")
    print("="*70)
    
    try:
        # Initialize generator
        generator = SFTDataGenerator(args.chroma_path, args.collection)
        
        # Generate dataset
        train_data, val_data = generator.generate_dataset(
            top_k=args.top_k,
            max_samples=args.max_samples,
            val_split=args.val_split
        )
        
        # Save to JSONL
        out_dir = Path(args.out_dir)
        generator.save_jsonl(train_data, out_dir / 'train_sft.jsonl')
        generator.save_jsonl(val_data, out_dir / 'val_sft.jsonl')
        
        print("\n" + "="*70)
        print("  SUCCESS")
        print("="*70)
        print(f"✅ Training data: {out_dir / 'train_sft.jsonl'}")
        print(f"✅ Validation data: {out_dir / 'val_sft.jsonl'}")
        print("\n💡 Next step: Run LoRA trainer")
        print("   python -m src.training.lora_trainer --config configs/lora_sft.yaml --dry-run")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
