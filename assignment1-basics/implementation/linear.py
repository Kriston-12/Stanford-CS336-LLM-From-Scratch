from torch import nn, Tensor
import torch


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
