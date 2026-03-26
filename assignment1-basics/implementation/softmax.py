import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x_max = torch.max(x, dim=self.dim, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        x_sum = torch.sum(x_exp, dim=self.dim, keepdim=True)
        return x_exp / x_sum