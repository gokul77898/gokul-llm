"""Unit tests for Mamba Architecture"""

import pytest
import torch
import numpy as np
from src.mamba.model import MambaModel, PositionalEncoding
from src.mamba.tokenizer import DocumentTokenizer
from src.mamba.attention import HierarchicalAttention, MultiHeadAttention


class TestDocumentTokenizer:
    """Test suite for DocumentTokenizer"""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        tokenizer = DocumentTokenizer(vocab_size=1000, max_length=512)
        assert tokenizer.vocab_size == 1000
        assert tokenizer.max_length == 512
        assert len(tokenizer.special_tokens) == 6
    
    def test_basic_tokenization(self):
        """Test basic text tokenization"""
        tokenizer = DocumentTokenizer(vocab_size=1000)
        tokenizer.build_vocab(["This is a test document for legal processing."])
        
        text = "This is a test."
        encoded = tokenizer.encode(text, return_tensors=True)
        
        assert encoded.input_ids.shape[0] == 1
        assert encoded.attention_mask.shape[0] == 1
        assert encoded.chunk_boundaries.shape[0] == 1
    
    def test_chunk_creation(self):
        """Test document chunking"""
        tokenizer = DocumentTokenizer(
            vocab_size=1000,
            chunk_size=10,
            chunk_overlap=2
        )
        
        # Long text that should be chunked
        text = " ".join(["word"] * 50)
        encoded = tokenizer.encode(text, return_tensors=False)
        
        assert len(encoded['chunk_boundaries']) > 1
    
    def test_encode_decode(self):
        """Test encoding and decoding"""
        tokenizer = DocumentTokenizer(vocab_size=1000)
        # Build vocab with the text we'll encode
        text = "This is a test"
        tokenizer.build_vocab([text, "This is a test document"])
        
        encoded = tokenizer.encode(text, return_tensors=True)
        decoded = tokenizer.decode(encoded.input_ids[0])
        
        assert isinstance(decoded, str)
        assert len(decoded) > 0  # Should contain at least some tokens
    
    def test_batch_encoding(self):
        """Test batch encoding"""
        tokenizer = DocumentTokenizer(vocab_size=1000)
        tokenizer.build_vocab(["Test document one.", "Test document two."])
        
        texts = ["Test one", "Test two"]
        encoded = tokenizer.batch_encode(texts, return_tensors=True)
        
        assert encoded.input_ids.shape[0] == 2
        assert encoded.attention_mask.shape[0] == 2


class TestHierarchicalAttention:
    """Test suite for HierarchicalAttention"""
    
    def test_attention_initialization(self):
        """Test attention module initialization"""
        attention = HierarchicalAttention(d_model=512, num_heads=8)
        assert attention.d_model == 512
        assert attention.num_heads == 8
    
    def test_attention_forward(self):
        """Test forward pass"""
        batch_size, seq_len, d_model = 2, 10, 512
        attention = HierarchicalAttention(d_model=d_model, num_heads=8)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention_info = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert 'token_attention' in attention_info
    
    def test_attention_with_chunks(self):
        """Test attention with chunk boundaries"""
        batch_size, seq_len, d_model = 2, 20, 512
        attention = HierarchicalAttention(d_model=d_model, num_heads=8)
        
        x = torch.randn(batch_size, seq_len, d_model)
        chunk_boundaries = torch.tensor([[[0, 10], [10, 20]], [[0, 10], [10, 20]]])
        
        output, attention_info = attention(x, chunk_boundaries)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert 'chunk_attention' in attention_info


class TestMambaModel:
    """Test suite for MambaModel"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = MambaModel(
            vocab_size=1000,
            d_model=256,
            num_layers=4,
            num_heads=8,
            num_classes=5
        )
        
        assert model.vocab_size == 1000
        assert model.d_model == 256
        assert model.num_layers == 4
    
    def test_classification_forward(self):
        """Test classification forward pass"""
        vocab_size = 1000
        num_classes = 5
        model = MambaModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=2,
            num_heads=4,
            num_classes=num_classes
        )
        
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, num_classes, (batch_size,))
        
        outputs = model(input_ids=input_ids, labels=labels, task="classification")
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (batch_size, num_classes)
    
    def test_generation_forward(self):
        """Test generation forward pass"""
        vocab_size = 1000
        model = MambaModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=2,
            num_heads=4
        )
        
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids=input_ids, labels=labels, task="generation")
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
    
    def test_model_generate(self):
        """Test text generation"""
        vocab_size = 1000
        model = MambaModel(vocab_size=vocab_size, d_model=128, num_layers=2, num_heads=4)
        model.eval()
        
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        generated = model.generate(input_ids, max_length=20)
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 30  # input + generated
    
    def test_model_parameters(self):
        """Test parameter counting"""
        model = MambaModel(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            num_heads=4
        )
        
        total_params = model.get_num_parameters()
        trainable_params = model.get_num_parameters(only_trainable=True)
        
        assert total_params > 0
        assert trainable_params == total_params


class TestPositionalEncoding:
    """Test suite for PositionalEncoding"""
    
    def test_absolute_encoding(self):
        """Test absolute positional encoding"""
        d_model = 512
        max_len = 100
        pos_enc = PositionalEncoding(d_model, max_len, encoding_type="absolute")
        
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_learned_encoding(self):
        """Test learned positional encoding"""
        d_model = 512
        max_len = 100
        pos_enc = PositionalEncoding(d_model, max_len, encoding_type="learned")
        
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == (batch_size, seq_len, d_model)


@pytest.fixture
def sample_tokenizer():
    """Fixture for sample tokenizer"""
    tokenizer = DocumentTokenizer(vocab_size=1000, max_length=512)
    tokenizer.build_vocab([
        "This is a sample legal document.",
        "Another legal text for testing."
    ])
    return tokenizer


@pytest.fixture
def sample_model():
    """Fixture for sample model"""
    return MambaModel(
        vocab_size=1000,
        d_model=256,
        num_layers=2,
        num_heads=4,
        num_classes=5
    )


def test_end_to_end_mamba(sample_tokenizer, sample_model):
    """Test end-to-end Mamba pipeline"""
    text = "This is a test legal document for classification."
    
    # Tokenize
    encoded = sample_tokenizer.encode(text, return_tensors=True)
    
    # Forward pass
    outputs = sample_model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        chunk_boundaries=encoded.chunk_boundaries,
        task="classification"
    )
    
    assert 'logits' in outputs
    assert outputs['logits'].shape[1] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
