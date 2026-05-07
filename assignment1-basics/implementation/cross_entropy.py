import torch
import torch.nn as nn
from torch import Tensor

class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    '''Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.'''
    def forward(self, x: Tensor, targets: Tensor):
        log_probs = x.log_softmax(dim=-1)
        row_idx = torch.arange(x.shape[0], device=x.device)
        return (-log_probs[row_idx, targets]).mean()
