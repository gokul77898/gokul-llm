"""Custom Document Tokenizer for Long Legal Documents"""

import torch
from typing import List, Dict, Tuple, Optional, Union
import re
from dataclasses import dataclass


@dataclass
class TokenizedDocument:
    """Container for tokenized document data"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chunk_boundaries: torch.Tensor
    special_tokens_mask: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None


class DocumentTokenizer:
    """
    Custom tokenizer for chunking and processing long legal documents.
    
    Features:
    - Sliding window chunking for long documents
    - Special token handling for document structure
    - Chunk boundary tracking
    - Padding and masking
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    CHUNK_START_TOKEN = "[CHUNK_START]"
    CHUNK_END_TOKEN = "[CHUNK_END]"
    
    def __init__(
        self,
        vocab_size: int = 30000,
        max_length: int = 512,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        lowercase: bool = True
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            lowercase: Whether to lowercase text
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lowercase = lowercase
        
        # Initialize vocabulary
        self.special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN,
            self.SEP_TOKEN, self.CHUNK_START_TOKEN, self.CHUNK_END_TOKEN
        ]
        
        self.token2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
        # Special token IDs
        self.pad_token_id = self.token2id[self.PAD_TOKEN]
        self.unk_token_id = self.token2id[self.UNK_TOKEN]
        self.cls_token_id = self.token2id[self.CLS_TOKEN]
        self.sep_token_id = self.token2id[self.SEP_TOKEN]
        self.chunk_start_id = self.token2id[self.CHUNK_START_TOKEN]
        self.chunk_end_id = self.token2id[self.CHUNK_END_TOKEN]
        
        self.next_id = len(self.special_tokens)
        
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from a corpus of texts"""
        from collections import Counter
        
        token_freq = Counter()
        
        for text in texts:
            tokens = self._basic_tokenize(text)
            token_freq.update(tokens)
        
        # Add tokens to vocabulary based on frequency
        for token, freq in token_freq.most_common(self.vocab_size - len(self.special_tokens)):
            if freq >= min_freq and token not in self.token2id:
                self.token2id[token] = self.next_id
                self.id2token[self.next_id] = token
                self.next_id += 1
                
                if self.next_id >= self.vocab_size:
                    break
        
        print(f"Built vocabulary with {len(self.token2id)} tokens")
        
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization: split on whitespace and punctuation"""
        if self.lowercase:
            text = text.lower()
        
        # Split on whitespace and keep punctuation separate
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> Union[TokenizedDocument, Dict]:
        """
        Encode text into token IDs with chunking for long documents.
        
        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens
            return_tensors: Whether to return PyTorch tensors
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            
        Returns:
            TokenizedDocument or dict with token information
        """
        tokens = self._basic_tokenize(text)
        
        # Convert tokens to IDs
        token_ids = [
            self.token2id.get(token, self.unk_token_id)
            for token in tokens
        ]
        
        # Create chunks with sliding window
        chunks, chunk_boundaries = self._create_chunks(token_ids)
        
        # Add special tokens if requested
        if add_special_tokens:
            chunks, chunk_boundaries = self._add_special_tokens(chunks, chunk_boundaries)
        
        # Flatten chunks into single sequence
        input_ids = []
        for chunk in chunks:
            input_ids.extend(chunk)
        
        # Truncate if needed
        if truncation and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            # Adjust chunk boundaries
            chunk_boundaries = self._adjust_boundaries(chunk_boundaries, self.max_length)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if padding and len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        if return_tensors:
            return TokenizedDocument(
                input_ids=torch.tensor([input_ids], dtype=torch.long),
                attention_mask=torch.tensor([attention_mask], dtype=torch.long),
                chunk_boundaries=torch.tensor([chunk_boundaries], dtype=torch.long),
                metadata={'num_chunks': len(chunks), 'original_length': len(tokens)}
            )
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'chunk_boundaries': chunk_boundaries,
                'metadata': {'num_chunks': len(chunks), 'original_length': len(tokens)}
            }
    
    def _create_chunks(
        self,
        token_ids: List[int]
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """Create overlapping chunks from token IDs"""
        chunks = []
        boundaries = []
        
        if len(token_ids) <= self.chunk_size:
            chunks.append(token_ids)
            boundaries.append((0, len(token_ids)))
            return chunks, boundaries
        
        start = 0
        current_pos = 0
        
        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids))
            chunk = token_ids[start:end]
            chunks.append(chunk)
            boundaries.append((current_pos, current_pos + len(chunk)))
            
            current_pos += len(chunk)
            start += (self.chunk_size - self.chunk_overlap)
            
            if end == len(token_ids):
                break
        
        return chunks, boundaries
    
    def _add_special_tokens(
        self,
        chunks: List[List[int]],
        boundaries: List[Tuple[int, int]]
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """Add special tokens to chunks"""
        new_chunks = []
        new_boundaries = []
        offset = 0
        
        for i, chunk in enumerate(chunks):
            new_chunk = [self.chunk_start_id] + chunk + [self.chunk_end_id]
            new_chunks.append(new_chunk)
            
            start, end = boundaries[i]
            new_boundaries.append((start + offset, end + offset + 2))
            offset += 2
        
        # Add CLS at start and SEP at end
        if new_chunks:
            new_chunks[0] = [self.cls_token_id] + new_chunks[0]
            new_chunks[-1] = new_chunks[-1] + [self.sep_token_id]
            
            # Adjust first and last boundaries
            new_boundaries = [(s + 1, e + 1) for s, e in new_boundaries]
            new_boundaries[-1] = (new_boundaries[-1][0], new_boundaries[-1][1] + 1)
        
        return new_chunks, new_boundaries
    
    def _adjust_boundaries(
        self,
        boundaries: List[Tuple[int, int]],
        max_length: int
    ) -> List[Tuple[int, int]]:
        """Adjust chunk boundaries after truncation"""
        adjusted = []
        for start, end in boundaries:
            if start >= max_length:
                break
            adjusted.append((start, min(end, max_length)))
        return adjusted
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue
            token = self.id2token.get(token_id, self.UNK_TOKEN)
            if token not in self.special_tokens:
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: bool = True
    ) -> TokenizedDocument:
        """Encode a batch of texts"""
        encoded_list = [
            self.encode(text, add_special_tokens, False, padding, truncation)
            for text in texts
        ]
        
        if return_tensors:
            # Find max length in batch
            max_len = max(len(enc['input_ids']) for enc in encoded_list)
            
            # Pad all to same length
            batch_input_ids = []
            batch_attention_mask = []
            batch_chunk_boundaries = []
            
            for enc in encoded_list:
                input_ids = enc['input_ids']
                attention_mask = enc['attention_mask']
                
                if len(input_ids) < max_len:
                    padding_length = max_len - len(input_ids)
                    input_ids.extend([self.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_chunk_boundaries.append(enc['chunk_boundaries'])
            
            # Pad chunk boundaries to same number of chunks
            max_chunks = max(len(cb) for cb in batch_chunk_boundaries)
            for cb in batch_chunk_boundaries:
                while len(cb) < max_chunks:
                    cb.append((0, 0))
            
            return TokenizedDocument(
                input_ids=torch.tensor(batch_input_ids, dtype=torch.long),
                attention_mask=torch.tensor(batch_attention_mask, dtype=torch.long),
                chunk_boundaries=torch.tensor(batch_chunk_boundaries, dtype=torch.long)
            )
        
        return encoded_list
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        import json
        with open(path, 'w') as f:
            json.dump({
                'token2id': self.token2id,
                'config': {
                    'vocab_size': self.vocab_size,
                    'max_length': self.max_length,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'lowercase': self.lowercase
                }
            }, f, indent=2)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.token2id = {k: int(v) for k, v in data['token2id'].items()}
            self.id2token = {int(v): k for k, v in self.token2id.items()}
            
            config = data['config']
            self.vocab_size = config['vocab_size']
            self.max_length = config['max_length']
            self.chunk_size = config['chunk_size']
            self.chunk_overlap = config['chunk_overlap']
            self.lowercase = config['lowercase']
