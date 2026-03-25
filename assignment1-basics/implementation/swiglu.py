import torch
from implementation.linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(out_features=d_ff, in_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(out_features=d_ff, in_features=d_model, device=device, dtype=dtype)
        self.w2 = Linear(out_features=d_model, in_features=d_ff, device=device, dtype=dtype)

    # input: (batch_size, seq_len, d_model)
    # W2 @ (SiLU(W1 @ x) * (W3 @ x))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x) # (batch_size, seq_len, d_ff)
        w3x = self.w3(x) # (batch_size, seq_len, d_ff)
        return self.w2(torch.sigmoid(w1x) * w1x * w3x) # (batch_size, seq_len, d_model)
# def test():
#     a = torch.arange(1, 10).reshape(3, 3).float()
#     b = a.T
#     c = a.transpose(0, 1)
#     print(id(a), id(b), id(c)) # ids are different, but all three objects share the same underlying data
#     print(a.is_contiguous(), b.is_contiguous(), c.is_contiguous())  # a is contiguous, but b and c are not
#     print(a.storage().data_ptr(), b.storage().data_ptr(), c.storage().data_ptr())  # all point to the same underlying data

# test()