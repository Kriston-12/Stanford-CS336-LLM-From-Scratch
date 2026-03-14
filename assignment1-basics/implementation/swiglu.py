import torch

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))

    # input: (batch_size, seq_len, d_model)
    # W2 @ (SiLU(W1 @ x) * (W3 @ x))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = x @ self.w1.T
        return (torch.sigmoid(w1x) * (w1x) * (x @ self.w3.T)) @ self.w2.T

# def test():
#     a = torch.arange(1, 10).reshape(3, 3).float()
#     b = a.T
#     c = a.transpose(0, 1)
#     print(id(a), id(b), id(c)) # ids are different, but all three objects share the same underlying data
#     print(a.is_contiguous(), b.is_contiguous(), c.is_contiguous())  # a is contiguous, but b and c are not
#     print(a.storage().data_ptr(), b.storage().data_ptr(), c.storage().data_ptr())  # all point to the same underlying data

# test()