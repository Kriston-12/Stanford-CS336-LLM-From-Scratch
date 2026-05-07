import torch
from typing import Iterable
"""Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
"""
class GradientClipper:
    def __init__(self, max_l2_norm: float):
        self.max_l2_norm = max_l2_norm
    
    def clip_gradients(self, parameters: Iterable[torch.nn.Parameter]):
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                # total_norm += p.grad.data.norm(2) ** 2
                total_norm += p.grad.data.pow(2).sum() # pow(2) is element-wise square, sum() is sum of all elements, this is equivalent to norm(2) ** 2 but more efficient since we don't need to compute the square root.
        total_norm = total_norm ** 0.5
        if total_norm > self.max_l2_norm:
            clip_coef = self.max_l2_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef) # mul_ means multiply by clip_coef in place

def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> torch.Tensor:
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.pow(2).sum()
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm