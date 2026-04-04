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
                total_norm += p.grad.data.norm(2) ** 2
        total_norm = total_norm ** 0.5
        if total_norm > self.max_l2_norm:
            clip_coef = self.max_l2_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef) # mul_ means modify in place

