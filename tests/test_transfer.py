"""Unit tests for Transfer Architecture"""

import pytest
import torch
from src.transfer.model import LegalTransferModel, LegalTaskType
from src.transfer.tokenizer import LegalTokenizer


class TestLegalTokenizer:
    """Test suite for LegalTokenizer"""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        tokenizer = LegalTokenizer(base_model="bert-base-uncased")
        assert tokenizer.vocab_size > 0
        assert tokenizer.max_length == 512
    
    def test_legal_text_preprocessing(self):
        """Test legal text preprocessing"""
        tokenizer = LegalTokenizer()
        
        text = "In Smith v. Jones, 123 U.S. 456 (2020), the court held..."
        preprocessed = tokenizer.preprocess_legal_text(text)
        
        # Should replace case citation
        assert "[CASE_REF]" in preprocessed
    
    def test_encoding(self):
        """Test text encoding"""
        tokenizer = LegalTokenizer()
        
        text = "This is a legal document."
        encoded = tokenizer.encode(text)
        
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].shape[0] == 1
    
    def test_entity_extraction(self):
        """Test legal entity extraction"""
        tokenizer = LegalTokenizer()
        
        text = "The case was filed on 12/25/2023 under Section 123."
        entities = tokenizer.extract_legal_entities(text)
        
        assert 'dates' in entities
        assert 'statutes' in entities
        assert len(entities['dates']) > 0


class TestLegalTransferModel:
    """Test suite for LegalTransferModel"""
    
    def test_classification_model(self):
        """Test classification model"""
        model = LegalTransferModel(
            model_name="bert-base-uncased",
            task=LegalTaskType.CLASSIFICATION,
            num_labels=3
        )
        
        assert model.task == LegalTaskType.CLASSIFICATION
        assert model.num_labels == 3
    
    def test_ner_model(self):
        """Test NER model"""
        model = LegalTransferModel(
            model_name="bert-base-uncased",
            task=LegalTaskType.NER,
            num_labels=5
        )
        
        assert model.task == LegalTaskType.NER
    
    def test_classification_forward(self):
        """Test classification forward pass"""
        model = LegalTransferModel(
            model_name="bert-base-uncased",
            task=LegalTaskType.CLASSIFICATION,
            num_labels=3
        )
        
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 3, (batch_size,))
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (batch_size, 3)
    
    def test_ner_forward(self):
        """Test NER forward pass"""
        model = LegalTransferModel(
            model_name="bert-base-uncased",
            task=LegalTaskType.NER,
            num_labels=5
        )
        
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 5, (batch_size, seq_len))
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, 5)
    
    def test_qa_forward(self):
        """Test QA forward pass"""
        model = LegalTransferModel(
            model_name="bert-base-uncased",
            task=LegalTaskType.QA
        )
        
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        start_positions = torch.randint(0, seq_len, (batch_size,))
        end_positions = torch.randint(0, seq_len, (batch_size,))
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        
        assert 'start_logits' in outputs
        assert 'end_logits' in outputs
        assert 'loss' in outputs
    
    def test_model_freezing(self):
        """Test base model freezing"""
        model = LegalTransferModel(
            model_name="bert-base-uncased",
            task=LegalTaskType.CLASSIFICATION,
            num_labels=3,
            freeze_base=True
        )
        
        # Check that base model parameters are frozen
        for param in model.base_model.parameters():
            assert not param.requires_grad
    
    def test_parameter_count(self):
        """Test parameter counting"""
        model = LegalTransferModel(
            model_name="bert-base-uncased",
            task=LegalTaskType.CLASSIFICATION,
            num_labels=3
        )
        
        total_params = model.get_num_parameters()
        trainable_params = model.get_num_parameters(only_trainable=True)
        
        assert total_params > 0
        assert trainable_params > 0


@pytest.fixture
def sample_legal_tokenizer():
    """Fixture for legal tokenizer"""
    return LegalTokenizer(base_model="bert-base-uncased")


@pytest.fixture
def sample_legal_model():
    """Fixture for legal model"""
    return LegalTransferModel(
        model_name="bert-base-uncased",
        task=LegalTaskType.CLASSIFICATION,
        num_labels=3
    )


def test_end_to_end_transfer(sample_legal_tokenizer, sample_legal_model):
    """Test end-to-end transfer learning pipeline"""
    text = "This is a legal document about contract law."
    
    # Tokenize
    encoded = sample_legal_tokenizer.encode(text, return_tensors="pt")
    
    # Forward pass
    outputs = sample_legal_model(
        input_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask']
    )
    
    assert 'logits' in outputs
    assert outputs['logits'].shape[1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
