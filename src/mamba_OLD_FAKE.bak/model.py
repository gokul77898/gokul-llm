"""Mamba Model: Transformer with Hierarchical Attention for Long Documents"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from .attention import HierarchicalAttention, PositionwiseFeedForward


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Supports both absolute and relative positional encodings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        encoding_type: str = "absolute"
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type
        
        if encoding_type == "absolute":
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            self.register_buffer('pe', pe)
        elif encoding_type == "learned":
            self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        if self.encoding_type == "absolute":
            x = x + self.pe[:, :x.size(1)]
        elif self.encoding_type == "learned":
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.pe(positions)
        
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer with Hierarchical Attention"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_hierarchical: bool = True
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        self.use_hierarchical = use_hierarchical
        
        if use_hierarchical:
            self.attention = HierarchicalAttention(d_model, num_heads, dropout=dropout)
        else:
            from .attention import MultiHeadAttention
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = PositionwiseFeedForward(d_model, dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        chunk_boundaries: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            chunk_boundaries: [batch_size, num_chunks, 2]
            mask: [batch_size, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_info: dict with attention weights
        """
        if self.use_hierarchical:
            output, attention_info = self.attention(x, chunk_boundaries, mask)
            return output, attention_info
        else:
            # Standard transformer layer
            attn_out, attn_weights = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))
            return x, {'attention': attn_weights}


class MambaModel(nn.Module):
    """
    Mamba: Transformer-based model with Hierarchical Attention for Long Document Processing.
    
    Features:
    - Custom hierarchical attention mechanism
    - Sliding window processing for long documents
    - Efficient memory management with padding/masking
    - Configurable positional encodings
    - Document classification and generation heads
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        use_hierarchical: bool = True,
        positional_encoding: str = "absolute"
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Dimension of feedforward network (default: 4 * d_model)
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            num_classes: Number of output classes for classification (None for generation)
            use_hierarchical: Whether to use hierarchical attention
            positional_encoding: Type of positional encoding ("absolute", "learned")
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_hierarchical = use_hierarchical
        
        if d_ff is None:
            d_ff = d_model * 4
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(
            d_model, max_seq_length, dropout, positional_encoding
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, d_ff, dropout, use_hierarchical
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        if num_classes is not None:
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )
        else:
            self.classifier = None
        
        # Language modeling head (for generation)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_boundaries: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        task: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            chunk_boundaries: [batch_size, num_chunks, 2]
            labels: [batch_size] for classification or [batch_size, seq_len] for LM
            output_attentions: Whether to return attention weights
            task: "classification" or "generation"
            
        Returns:
            Dictionary containing:
                - logits: Model predictions
                - loss: Loss value (if labels provided)
                - attentions: Attention weights (if output_attentions=True)
                - hidden_states: Final hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        
        # Embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.positional_encoding(x)
        
        # Prepare mask for attention (convert to proper shape)
        if attention_mask is not None:
            # Expand mask for multi-head attention
            # [batch, 1, 1, seq_len] for broadcasting
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.to(dtype=x.dtype)
        else:
            extended_mask = None
        
        # Pass through encoder layers
        all_attentions = [] if output_attentions else None
        
        for layer in self.encoder_layers:
            x, attention_info = layer(x, chunk_boundaries, extended_mask)
            if output_attentions:
                all_attentions.append(attention_info)
        
        x = self.final_norm(x)
        hidden_states = x
        
        # Task-specific heads
        loss = None
        
        if task == "classification":
            # Use [CLS] token (first token) for classification
            cls_repr = x[:, 0]  # [batch_size, d_model]
            
            if self.classifier is not None:
                logits = self.classifier(cls_repr)  # [batch_size, num_classes]
                
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
            else:
                logits = cls_repr
        
        elif task == "generation":
            # Language modeling head
            logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
            
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        
        else:
            raise ValueError(f"Unknown task: {task}")
        
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
        if loss is not None:
            output['loss'] = loss
        
        if output_attentions:
            output['attentions'] = all_attentions
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: [batch_size, seq_len] - prompt tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            attention_mask: Attention mask for input
            
        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(
                    generated,
                    attention_mask=attention_mask,
                    task="generation"
                )
                
                # Get logits for next token
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
                    ], dim=1)
                
                # Check for end of sequence (assuming EOS token is 3)
                if (next_token == 3).all():
                    break
        
        return generated
    
    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """Get number of parameters in the model"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
