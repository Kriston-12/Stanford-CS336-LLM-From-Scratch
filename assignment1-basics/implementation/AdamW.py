import torch
from torch.optim import Optimizer
from typing import Optional, Callable

class AdamW(Optimizer):
    def __init__(
        self, 
        parameters,
        lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):  
        init_dict = {
            'lr': lr,
            'weight_decay': weight_decay,
            'betas': betas,
            'eps': eps,
        }
        super().__init__(parameters, init_dict)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta0, beta1 = group['betas']
            eps = group['eps']
            # beta, weight_decay, eps, lr are group-shared hyperparameters, so we can get them from the group dict.
            # moment are parameter-specific states, so we need to save them in the state dict with parameter tensors as keys.   
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]['t'] = self.state[p].get('t', 0) + 1
                self.state[p]['first_moment'] = \
                    beta0 * self.state[p].get('first_moment', torch.zeros_like(p)) + (1 - beta0) * grad
                self.state[p]['second_moment'] = \
                    beta1 * self.state[p].get('second_moment', torch.zeros_like(p)) + (1 - beta1) * grad.pow(2)

                lr_t = lr * (1 - beta1 ** (self.state[p]['t'])) ** 0.5 / (1 - beta0 ** (self.state[p]['t']))
                p.data -= lr_t * self.state[p]['first_moment'] / (self.state[p]['second_moment'].sqrt() + eps)
                p.data = p.data - lr * weight_decay * p.data
        return loss


