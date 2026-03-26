from torch import nn
import torch


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = 2 / (in_features + out_features)
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3 * std ** 0.5, b = 3 * std ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
