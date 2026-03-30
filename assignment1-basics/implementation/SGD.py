import torch
import torch.optim.optimizer as optim
from typing import Optional, Callable
import math

class SGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad  # Update wegiht tensor in-place.
                state['t'] = t + 1  # Increment iteration number and save it back to the state.
        return loss

def test(lr: float):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    loss = (weights ** 2).sum()
    print(loss.cpu().item())
    loss.backward()
    opt.step()

if __name__ == "__main__":
    TRAINING_STEPS = 10
    LR_LIST = [0.1, 0.01, 0.001]
    for lr in LR_LIST:
        for step in range(TRAINING_STEPS):
            test(lr=lr)
        print("Complete training with lr =", lr)