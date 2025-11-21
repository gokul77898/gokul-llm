"""Data loading utilities for legal documents"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class LegalDocument:
    """Legal document container"""
    text: str
    label: Optional[int] = None
    metadata: Optional[Dict] = None


class LegalDataLoader:
    """
    Data loader for legal documents.
    
    Supports various formats:
    - JSON/JSONL
    - CSV
    - Plain text
    - PDF (requires pdfplumber)
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
    
    def load_json(
        self,
        filename: str,
        text_field: str = 'text',
        label_field: str = 'label'
    ) -> List[LegalDocument]:
        """
        Load data from JSON file.
        
        Args:
            filename: JSON filename
            text_field: Field name for text
            label_field: Field name for label
            
        Returns:
            List of LegalDocument objects
        """
        filepath = self.data_dir / filename
        documents = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for item in data:
                doc = LegalDocument(
                    text=item[text_field],
                    label=item.get(label_field),
                    metadata={k: v for k, v in item.items() 
                             if k not in [text_field, label_field]}
                )
                documents.append(doc)
        
        return documents
    
    def load_jsonl(
        self,
        filename: str,
        text_field: str = 'text',
        label_field: str = 'label'
    ) -> List[LegalDocument]:
        """Load data from JSONL file"""
        filepath = self.data_dir / filename
        documents = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                doc = LegalDocument(
                    text=item[text_field],
                    label=item.get(label_field),
                    metadata={k: v for k, v in item.items()
                             if k not in [text_field, label_field]}
                )
                documents.append(doc)
        
        return documents
    
    def load_csv(
        self,
        filename: str,
        text_column: str = 'text',
        label_column: str = 'label',
        delimiter: str = ','
    ) -> List[LegalDocument]:
        """Load data from CSV file"""
        filepath = self.data_dir / filename
        documents = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                doc = LegalDocument(
                    text=row[text_column],
                    label=int(row[label_column]) if label_column in row else None,
                    metadata={k: v for k, v in row.items()
                             if k not in [text_column, label_column]}
                )
                documents.append(doc)
        
        return documents
    
    def load_text_files(
        self,
        pattern: str = "*.txt",
        label_from_filename: bool = False
    ) -> List[LegalDocument]:
        """
        Load data from text files.
        
        Args:
            pattern: File pattern to match
            label_from_filename: Extract label from filename
            
        Returns:
            List of LegalDocument objects
        """
        documents = []
        
        for filepath in self.data_dir.glob(pattern):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            label = None
            if label_from_filename:
                # Extract numeric label from filename
                match = re.search(r'_(\d+)\.', filepath.name)
                if match:
                    label = int(match.group(1))
            
            doc = LegalDocument(
                text=text,
                label=label,
                metadata={'filename': filepath.name}
            )
            documents.append(doc)
        
        return documents
    
    def split_data(
        self,
        documents: List[LegalDocument],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[List[LegalDocument], List[LegalDocument], List[LegalDocument]]:
        """
        Split data into train/val/test sets.
        
        Args:
            documents: List of documents
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            shuffle: Whether to shuffle before splitting
            seed: Random seed
            
        Returns:
            Tuple of (train, val, test) document lists
        """
        import random
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        docs = documents.copy()
        
        if shuffle:
            random.seed(seed)
            random.shuffle(docs)
        
        n = len(docs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_docs = docs[:train_end]
        val_docs = docs[train_end:val_end]
        test_docs = docs[val_end:]
        
        return train_docs, val_docs, test_docs
    
    def preprocess_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False
    ) -> str:
        """
        Preprocess legal text.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_numbers: Remove numbers
            
        Returns:
            Preprocessed text
        """
        if lowercase:
            text = text.lower()
        
        if remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)
        
        if remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def create_sample_data(
        num_samples: int = 100,
        num_classes: int = 3,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Create sample legal data for testing.
        
        Args:
            num_samples: Number of samples
            num_classes: Number of classes
            save_path: Optional path to save data
            
        Returns:
            List of data dictionaries
        """
        import random
        
        categories = ['Contract Law', 'Tort Law', 'Criminal Law', 'Property Law', 'Constitutional Law']
        
        data = []
        for i in range(num_samples):
            text = f"Legal document {i}: This case involves {categories[i % len(categories)]}. "
            text += f"The court held that Section {i % 20} applies. "
            text += f"Plaintiff and defendant must comply with statutory requirements."
            
            item = {
                'text': text,
                'label': i % num_classes,
                'doc_id': f'doc_{i}',
                'category': categories[i % len(categories)],
                'date': f'2024-01-{(i % 28) + 1:02d}'
            }
            data.append(item)
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Sample data saved to {save_path}")
        
        return data
