import torch
import torch.nn as nn
from einops import einsum

class ScaledDotProductAttention(nn.Module):
    def __init__(self, mask: torch.Tensor = None):
        super().__init__()
        self.mask = mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.attention = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") / (q.shape[-1] ** 0.5)
        if self.mask is not None:
            self.attention = self.attention.masked_fill_(self.mask == 0, float("-inf"))
        self.attention = torch.softmax(self.attention, dim=-1)
        # v has a shape of (... values dv), values = keys
        return einsum(self.attention, v, "... queries keys, ... keys d_v -> ... queries d_v")