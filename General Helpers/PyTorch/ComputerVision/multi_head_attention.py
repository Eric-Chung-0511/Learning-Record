import torch
import torch.nn as nn
import torch.nn.functional as F
from einsum_utils import Einsum

class MultiHeadAttention(nn.Module):
  
    def __init__(self, in_dim, num_heads=8):
        """
        Multi-Head Attention mechanism with Einstein summation for scaled dot-product attention.
        Args:
            in_dim (int): Input feature dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, in_dim // num_heads)  # Dimension of each attention head
        self.scaled_dim = self.head_dim * num_heads

        # Linear layers for Query, Key, and Value
        self.fc_in = nn.Linear(in_dim, self.scaled_dim)
        self.query = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.key = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.value = nn.Linear(self.scaled_dim, self.scaled_dim)
        self.fc_out = nn.Linear(self.scaled_dim, in_dim)  # Output projection

    def forward(self, x):
        """
        Applies multi-head attention to the input tensor.
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_dim).
        Returns:
            Tensor: Output tensor after attention.
        """
        N = x.shape[0]
        x = self.fc_in(x)

        # Compute Query, Key, and Value
        q = self.query(x).view(N, self.num_heads, self.head_dim)
        k = self.key(x).view(N, self.num_heads, self.head_dim)
        v = self.value(x).view(N, self.num_heads, self.head_dim)

        # Apply attention using Einsum
        attention = Einsum.compute_attention(q, k, v, self.head_dim)
        out = attention.reshape(N, self.scaled_dim)  # Concatenate all heads
        out = self.fc_out(out)
        return out


# Usage Example
# attention = MultiHeadAttention(in_dim=2048, num_heads=8)
# features = torch.rand(16, 2048)
# attended_features = attention(features)
