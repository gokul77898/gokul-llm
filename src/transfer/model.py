"""Transfer Learning Model for Legal Data"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoConfig
)
from typing import Optional, Dict, Tuple, List
from enum import Enum


class LegalTaskType(Enum):
    """Types of legal tasks"""
    CLASSIFICATION = "classification"
    NER = "ner"  # Named Entity Recognition
    SUMMARIZATION = "summarization"
    GENERATION = "generation"
    QA = "qa"  # Question Answering


class LegalTransferModel(nn.Module):
    """
    Transfer learning model for legal tasks.
    
    Supports:
    - Document classification
    - Named Entity Recognition (NER)
    - Text summarization
    - Text generation
    - Question answering
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        task: LegalTaskType = LegalTaskType.CLASSIFICATION,
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
        hidden_size: Optional[int] = None
    ):
        """
        Args:
            model_name: Pretrained model name from HuggingFace
            task: Type of legal task
            num_labels: Number of labels (for classification/NER)
            dropout: Dropout probability
            freeze_base: Whether to freeze base model weights
            hidden_size: Hidden size for custom heads (if None, uses model's hidden size)
        """
        super().__init__()
        
        self.model_name = model_name
        self.task = task
        self.num_labels = num_labels
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)
        
        if hidden_size is None:
            self.hidden_size = self.config.hidden_size
        else:
            self.hidden_size = hidden_size
        
        # Load base model based on task
        if task == LegalTaskType.CLASSIFICATION:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
        elif task == LegalTaskType.NER:
            self.base_model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
        elif task in [LegalTaskType.GENERATION, LegalTaskType.SUMMARIZATION]:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
        else:
            # Load base encoder for custom tasks
            self.base_model = AutoModel.from_pretrained(
                model_name,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
            
            # Custom task-specific heads
            if task == LegalTaskType.QA:
                self.qa_outputs = nn.Linear(self.hidden_size, 2)  # start/end positions
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Additional layers for legal-specific features
        self.legal_feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task learning heads
        self.auxiliary_classifier = nn.Linear(self.hidden_size, 5)  # e.g., document type
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            labels: Labels for the task
            start_positions: Start positions for QA
            end_positions: End positions for QA
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attention weights
            return_dict: Whether to return as dictionary
            
        Returns:
            Dictionary containing model outputs
        """
        outputs = {}
        
        # Forward through base model
        if self.task == LegalTaskType.CLASSIFICATION:
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True
            )
            
            outputs['loss'] = base_outputs.loss
            outputs['logits'] = base_outputs.logits
            
            if output_hidden_states:
                outputs['hidden_states'] = base_outputs.hidden_states
            if output_attentions:
                outputs['attentions'] = base_outputs.attentions
        
        elif self.task == LegalTaskType.NER:
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True
            )
            
            outputs['loss'] = base_outputs.loss
            outputs['logits'] = base_outputs.logits
            
            if output_hidden_states:
                outputs['hidden_states'] = base_outputs.hidden_states
            if output_attentions:
                outputs['attentions'] = base_outputs.attentions
        
        elif self.task in [LegalTaskType.GENERATION, LegalTaskType.SUMMARIZATION]:
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True
            )
            
            outputs['loss'] = base_outputs.loss
            outputs['logits'] = base_outputs.logits
            
            if output_hidden_states:
                outputs['hidden_states'] = base_outputs.hidden_states
            if output_attentions:
                outputs['attentions'] = base_outputs.attentions
        
        elif self.task == LegalTaskType.QA:
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True
            )
            
            sequence_output = base_outputs.last_hidden_state
            
            # Extract legal features
            legal_features = self.legal_feature_extractor(sequence_output)
            
            # QA head
            logits = self.qa_outputs(legal_features)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            
            outputs['start_logits'] = start_logits
            outputs['end_logits'] = end_logits
            
            # Calculate loss if positions provided
            if start_positions is not None and end_positions is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                outputs['loss'] = (start_loss + end_loss) / 2
            
            if output_hidden_states:
                outputs['hidden_states'] = base_outputs.hidden_states
            if output_attentions:
                outputs['attentions'] = base_outputs.attentions
        
        return outputs if return_dict else tuple(outputs.values())
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text for generation/summarization tasks.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length to generate
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs
        """
        if self.task not in [LegalTaskType.GENERATION, LegalTaskType.SUMMARIZATION]:
            raise ValueError(f"Generation not supported for task: {self.task}")
        
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings (when adding new tokens)"""
        self.base_model.resize_token_embeddings(new_num_tokens)
    
    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """Get number of parameters"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        self.base_model.save_pretrained(save_directory)
        
        # Save additional components
        torch.save({
            'legal_feature_extractor': self.legal_feature_extractor.state_dict(),
            'auxiliary_classifier': self.auxiliary_classifier.state_dict(),
            'task': self.task.value,
            'num_labels': self.num_labels,
            'model_name': self.model_name
        }, f"{save_directory}/additional_components.pt")
    
    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        **kwargs
    ) -> 'LegalTransferModel':
        """Load model from directory"""
        # Load additional components
        additional = torch.load(
            f"{load_directory}/additional_components.pt",
            map_location='cpu'
        )
        
        # Create model instance
        model = cls(
            model_name=load_directory,
            task=LegalTaskType(additional['task']),
            num_labels=additional['num_labels'],
            **kwargs
        )
        
        # Load additional component weights
        model.legal_feature_extractor.load_state_dict(
            additional['legal_feature_extractor']
        )
        model.auxiliary_classifier.load_state_dict(
            additional['auxiliary_classifier']
        )
        
        return model


class EnsembleLegalModel(nn.Module):
    """Ensemble of multiple legal models for improved performance"""
    
    def __init__(self, models: List[LegalTransferModel], weights: Optional[List[float]] = None):
        """
        Args:
            models: List of LegalTransferModel instances
            weights: Optional weights for each model (normalized automatically)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.weights = weights
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        outputs_list = [model(**kwargs) for model in self.models]
        
        # Weighted average of logits
        logits = sum(
            w * out['logits'] for w, out in zip(self.weights, outputs_list)
        )
        
        # Average loss if available
        if 'loss' in outputs_list[0]:
            loss = sum(
                w * out['loss'] for w, out in zip(self.weights, outputs_list)
            )
        else:
            loss = None
        
        return {
            'logits': logits,
            'loss': loss
        }
