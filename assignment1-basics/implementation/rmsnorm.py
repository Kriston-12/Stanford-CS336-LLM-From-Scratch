from torch import nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        self.eps = eps

    # input: (batch_size, seq_len, d_model) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.g * x / rms

def test():
    a = torch.arange(1, 10).reshape(3, 3).float()
    b = torch.stack([a, a], dim=0)
    print(b)
    print(torch.mean(b ** 2, dim=-1, keepdim=True))

# test()