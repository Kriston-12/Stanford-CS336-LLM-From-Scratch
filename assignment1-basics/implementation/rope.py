import torch
from torch import nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"

        half = d_k // 2 # d_k could be 1024, half = 512
        # exponent is (2k - 2) / d_k, where k starts from 1, so the first exponent is 0, the second is 2 / d_k, the third is 4/d_k, ..., the last is (d_k - 2)/d_k
        # last_index of torch.arange(half) is half - 1, so the last exponent is (2 * (half - 1)) / d_k = (d_k - 2) / d_k
        inv_freq = theta ** (-2 * torch.arange(half, device=device) / d_k) # (d_k/2,)
        pos = torch.arange(max_seq_len, device=device) # (max_seq_len,)
        angles = pos[:, None] * inv_freq[None, :] # (max_seq_len, d_k/2)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        *prefix, seq_len, d_k = x.shape

        # (seq_len, half) -> broadcast to (..., seq_len, half)
        cos = self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype)
        # add leading singleton dims to match prefix broadcast
        for _ in prefix:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # Group pairs: (..., seq_len, half, 2) without copying (view if contiguous, else a cheap view/reshape)
        x2 = rearrange(x, "... s (h two) -> ... s h two", two=2)  # two=2

        # Rotate each 2D vector [a,b] by angle: [a*cos - b*sin, a*sin + b*cos]
        a = x2[..., 0]  # (..., s, h)
        b = x2[..., 1]  # (..., s, h)

        y2 = torch.empty_like(x2)
        y2[..., 0] = a * cos - b * sin
        y2[..., 1] = a * sin + b * cos

        # Back to (..., seq_len, d_k)
        y = rearrange(y2, "... s h two -> ... s (h two)")
        return y
