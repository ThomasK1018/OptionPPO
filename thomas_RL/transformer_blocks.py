import torch
import torch.nn as nn
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the Positional Encoder.

        :param d_model: The dimension of the model (embedding size).
        :param max_len: The maximum sequence length.
        """
        super(PositionalEncoder, self).__init__()

        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Compute sin and cos encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add positional encodings to input embeddings.

        :param x: Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, d_model)
        :return: Positional encoded tensor of shape (batch_size, seq_len, d_model)
        """
        if x.dim() == 2:  # (batch_size, seq_len)
            batch_size, seq_len = x.size()
            # Get the positional encoding for the sequence length
            pe = self.pe[:, :seq_len]
            # Expand the input to (batch_size, seq_len, d_model) and add the positional encoding
            x_expanded = x.unsqueeze(-1).expand(batch_size, seq_len, self.pe.size(2))  # Expand across the third axis
            return x_expanded + pe.expand(batch_size, -1, -1)

        elif x.dim() == 3:  # (batch_size, seq_len, d_model)
            batch_size, seq_len, _ = x.size()
            pe = self.pe[:, :seq_len]
            return x + pe.expand(batch_size, -1, -1)

        else:
            raise ValueError("Input tensor must be 2D or 3D.")

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)

        # Feedforward network
        ff_output = self.feedforward(x)
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.norm2(x)

        return x

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        attn_scores = self.attention_weights(x).squeeze(-1)  # Shape: (batch_size, sequence_length)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: (batch_size, sequence_length)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, embed_dim)
        return pooled

