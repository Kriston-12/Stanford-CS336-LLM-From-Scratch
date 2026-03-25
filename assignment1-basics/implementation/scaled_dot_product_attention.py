import torch
import torch.nn as nn
from einops import einsum
from implementation.softmax import Softmax 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=-1)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        self.attention = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") / (q.shape[-1] ** 0.5)
        if mask is not None:
            self.attention = self.attention.masked_fill_(mask == 0, float("-inf"))
        self.attention = self.softmax(self.attention)
        # v has a shape of (... values dv), values = keys
        return einsum(self.attention, v, "... queries keys, ... keys d_v -> ... queries d_v")