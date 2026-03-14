import torch

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()

        self.w1 = torch.nn.Parameter(torch.empty((d_model, 3/8 * d_model), device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty((d_model, 3/8 * d_model), device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty((3/8 * d_model, d_model), device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2 @ (torch.sigmoid(self.w1 @ x) * (self.w3 @ x))
    