import pandas as pd
import numpy as np
import os
import torch
import torch.distributions as D
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

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