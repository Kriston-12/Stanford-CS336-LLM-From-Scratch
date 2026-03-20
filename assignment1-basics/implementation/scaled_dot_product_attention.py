import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, mask: torch.Tensor = None):
        super().__init__()
        self.mask = mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.attention = q @ k.T / (q.shape[-1] ** 0.5)
        if self.mask is not None:
            self.attention = self.attention.masked_fill(self.mask == 0, float("-inf"))
        self.attention = torch.softmax(self.attention, dim=-1)
        return self.attention @ v