"""Legal-specific tokenizer with special token handling"""

from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict, Optional, Union
import re


class LegalTokenizer:
    """
    Tokenizer specialized for legal documents.
    
    Features:
    - Legal-specific special tokens (case citations, statutes, etc.)
    - Named entity preservation
    - Date and reference formatting
    """
    
    # Legal-specific special tokens
    LEGAL_SPECIAL_TOKENS = [
        "[CASE_REF]",      # Case citation reference
        "[STATUTE]",       # Statute reference
        "[SECTION]",       # Section reference
        "[ARTICLE]",       # Article reference
        "[PLAINTIFF]",     # Plaintiff entity
        "[DEFENDANT]",     # Defendant entity
        "[COURT]",         # Court entity
        "[DATE]",          # Date entity
        "[CITATION]",      # Legal citation
        "[QUOTE]",         # Legal quotation
    ]
    
    def __init__(
        self,
        base_model: str = "bert-base-uncased",
        max_length: int = 512,
        add_legal_tokens: bool = True
    ):
        """
        Args:
            base_model: Base pretrained model to use
            max_length: Maximum sequence length
            add_legal_tokens: Whether to add legal-specific tokens
        """
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.max_length = max_length
        
        # Add legal-specific tokens
        if add_legal_tokens:
            self.base_tokenizer.add_special_tokens({
                'additional_special_tokens': self.LEGAL_SPECIAL_TOKENS
            })
        
        # Legal patterns for entity recognition
        self.patterns = {
            'case_citation': re.compile(
                r'\d+\s+[A-Z]\.?[A-Z]?\.?[A-Z]?\.?\s+\d+|'  # e.g., "123 U.S. 456"
                r'\[\d{4}\]\s+\w+\s+\d+'                      # e.g., "[2023] UKSC 12"
            ),
            'statute': re.compile(
                r'\d+\s+U\.S\.C\.\s+§\s*\d+|'  # U.S. Code
                r'Section\s+\d+|'                # Section reference
                r'§\s*\d+'                        # Section symbol
            ),
            'date': re.compile(
                r'\b\d{1,2}/\d{1,2}/\d{4}\b|'   # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b|'        # YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b'
            ),
        }
    
    def preprocess_legal_text(self, text: str) -> str:
        """
        Preprocess legal text to standardize entities.
        
        Args:
            text: Raw legal text
            
        Returns:
            Preprocessed text with standardized entities
        """
        # Replace case citations
        text = self.patterns['case_citation'].sub('[CASE_REF]', text)
        
        # Replace statutes
        text = self.patterns['statute'].sub('[STATUTE]', text)
        
        # Replace dates
        text = self.patterns['date'].sub('[DATE]', text)
        
        return text
    
    def encode(
        self,
        text: Union[str, List[str]],
        preprocess: bool = True,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt"
    ) -> Dict:
        """
        Encode text into token IDs.
        
        Args:
            text: Text or list of texts to encode
            preprocess: Whether to preprocess legal entities
            add_special_tokens: Add special tokens
            padding: Padding strategy
            truncation: Whether to truncate
            max_length: Maximum length (uses self.max_length if None)
            return_tensors: Return type ("pt" for PyTorch)
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if max_length is None:
            max_length = self.max_length
        
        # Preprocess if requested
        if preprocess:
            if isinstance(text, str):
                text = self.preprocess_legal_text(text)
            else:
                text = [self.preprocess_legal_text(t) for t in text]
        
        # Tokenize using base tokenizer
        encoded = self.base_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def tokenize(self, text: str):
        """
        Compatibility tokenizer wrapper.
        Uses encode() to get IDs and maps them back to token strings using id_to_token.
        Works even before any training data is fed.
        """
        # Encode text to IDs
        ids = self.encode(text)

        # Convert IDs back to tokens
        tokens = []
        for idx in ids['input_ids'][0]:
            tok = self.base_tokenizer.convert_ids_to_tokens([idx])[0]
            if tok is None:
                tok = "<unk>"
            tokens.append(tok)

        return tokens
    
    def batch_decode(
        self,
        token_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Batch decode token IDs.
        
        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        return self.base_tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities from text.
        
        Args:
            text: Legal text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'case_citations': self.patterns['case_citation'].findall(text),
            'statutes': self.patterns['statute'].findall(text),
            'dates': self.patterns['date'].findall(text),
        }
        
        return entities
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory"""
        self.base_tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'LegalTokenizer':
        """Load tokenizer from directory"""
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = AutoTokenizer.from_pretrained(load_directory)
        tokenizer.max_length = 512
        return tokenizer
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.base_tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID"""
        return self.base_tokenizer.pad_token_id
    
    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID"""
        return self.base_tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID"""
        return self.base_tokenizer.sep_token_id
