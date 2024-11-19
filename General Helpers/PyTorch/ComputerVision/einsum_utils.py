import torch
import torch.nn.functional as F

class Einsum:
    @staticmethod
    def compute_attention(q, k, v, head_dim):
        """
        Computes scaled dot-product attention using Einstein summation.
        Args:
            q (Tensor): Query tensor of shape (N, num_heads, head_dim).
            k (Tensor): Key tensor of shape (N, num_heads, head_dim).
            v (Tensor): Value tensor of shape (N, num_heads, head_dim).
            head_dim (int): Dimension of each attention head.
        Returns:
            Tensor: Attention-weighted values.
        """
        # Compute attention scores
        energy = torch.einsum("nqd,nkd->nqk", [q, k])
        attention = F.softmax(energy / (head_dim ** 0.5), dim=2) # 縮放,防止梯度爆炸

        # Compute weighted sum
        return torch.einsum("nqk,nvd->nqd", [attention, v])


# Usage Example
# q = torch.rand(2, 8, 64)
# k = torch.rand(2, 8, 64)
# v = torch.rand(2, 8, 64)
# output = Einsum.compute_attention(q, k, v, head_dim=64)
