"""Hierarchical Attention Mechanism for Long Documents"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Custom Multi-Head Attention implementation"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or broadcastable
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query)  # [batch, seq_len, d_model]
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: [batch, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # Shape: [batch, num_heads, seq_len, d_k]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        # Shape: [batch, seq_len, d_model]
        
        output = self.out_linear(context)
        
        return output, attention_weights


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention Mechanism for processing long documents.
    
    Processes documents at multiple levels:
    1. Token-level attention within chunks
    2. Chunk-level attention across document sections
    3. Document-level attention for global context
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_levels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        
        # Create attention layers for each level
        self.token_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.chunk_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.document_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization for each level
        self.token_norm = nn.LayerNorm(d_model)
        self.chunk_norm = nn.LayerNorm(d_model)
        self.document_norm = nn.LayerNorm(d_model)
        
        # Feedforward networks
        self.token_ffn = PositionwiseFeedForward(d_model, dropout)
        self.chunk_ffn = PositionwiseFeedForward(d_model, dropout)
        self.document_ffn = PositionwiseFeedForward(d_model, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        chunk_boundaries: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            chunk_boundaries: [batch_size, num_chunks, 2] - start and end indices
            mask: [batch_size, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_info: dict with attention weights at each level
        """
        attention_info = {}
        
        # Level 1: Token-level attention
        token_attn_out, token_weights = self.token_attention(x, x, x, mask)
        x = self.token_norm(x + self.dropout(token_attn_out))
        x = x + self.dropout(self.token_ffn(x))
        attention_info['token_attention'] = token_weights
        
        # Level 2: Chunk-level attention
        if chunk_boundaries is not None:
            chunk_repr = self._aggregate_chunks(x, chunk_boundaries)
            chunk_attn_out, chunk_weights = self.chunk_attention(
                chunk_repr, chunk_repr, chunk_repr
            )
            chunk_repr = self.chunk_norm(chunk_repr + self.dropout(chunk_attn_out))
            chunk_repr = chunk_repr + self.dropout(self.chunk_ffn(chunk_repr))
            attention_info['chunk_attention'] = chunk_weights
            
            # Broadcast chunk information back to tokens
            x = self._broadcast_chunk_info(x, chunk_repr, chunk_boundaries)
        
        # Level 3: Document-level attention (global pooling)
        doc_repr = x.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        doc_attn_out, doc_weights = self.document_attention(
            doc_repr, x, x, mask
        )
        doc_repr = self.document_norm(doc_repr + self.dropout(doc_attn_out))
        doc_repr = doc_repr + self.dropout(self.document_ffn(doc_repr))
        attention_info['document_attention'] = doc_weights
        
        # Broadcast document-level information
        x = x + doc_repr.expand_as(x)
        
        return x, attention_info
    
    def _aggregate_chunks(
        self,
        x: torch.Tensor,
        chunk_boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate token representations into chunk representations"""
        batch_size, seq_len, d_model = x.shape
        num_chunks = chunk_boundaries.size(1)
        
        chunk_repr = torch.zeros(
            batch_size, num_chunks, d_model,
            device=x.device, dtype=x.dtype
        )
        
        for b in range(batch_size):
            for c in range(num_chunks):
                start, end = chunk_boundaries[b, c]
                if end > start:
                    chunk_repr[b, c] = x[b, start:end].mean(dim=0)
        
        return chunk_repr
    
    def _broadcast_chunk_info(
        self,
        x: torch.Tensor,
        chunk_repr: torch.Tensor,
        chunk_boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast chunk-level information back to token level"""
        batch_size, seq_len, d_model = x.shape
        num_chunks = chunk_boundaries.size(1)
        
        output = x.clone()
        
        for b in range(batch_size):
            for c in range(num_chunks):
                start, end = chunk_boundaries[b, c]
                if end > start:
                    output[b, start:end] += chunk_repr[b, c].unsqueeze(0)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
