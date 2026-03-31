from torch.optim import Optimizer, ParamsT
from typing import Optional, Callable
class AdamW(Optimizer):
    def __init__(
        self, 
        parameters,
        lr,
    ):  
        super().__init__(parameters, lr)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.first_moment: float = 0
        self.second_moment: float = 0
        self.epsilon = 1e-8
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                self.first_moment = self.beta1 * self.first_moment + (1 - self.beta1) * grad
                self.second_moment = self.beta2 * self.second_moment + (1 - self.beta2) * grad.pow(2)
                self.lr = lr * (1 - self.beta2 ** (self.state[p].get('t', 0))) ** 0.5 / (1 - self.beta1 ** (self.state[p].get('t', 0)))
                p.data -= self.lr * self.first_moment / (self.second_moment.sqrt() + self.epsilon)
                self.state[p]['t'] = self.state[p].get('t', 0) + 1
        return loss


