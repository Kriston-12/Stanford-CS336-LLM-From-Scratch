import torch
import torch.nn as nn
from torch import Tensor
from implementation.softmax import Softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super.__init__()
        self.softmax = Softmax(dim=-1)
    
    def forward(self, x: Tensor):
        logits = self.softmax(x)
        ls = x.shape[:-1]
        total_elements_to_average = 1
        for d in ls:
            total_elements_to_average *= d
        return torch.sum(torch.log(logits)) / total_elements_to_average
