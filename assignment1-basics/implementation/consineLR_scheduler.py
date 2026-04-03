import math

"""
Given the parameters of a cosine learning rate decay schedule (with linear
warmup) and an iteration number, return the learning rate at the given
iteration under the specified schedule.

Args:
    it (int): Iteration number to get learning rate for.
    max_learning_rate (float): alpha_max, the maximum learning rate for
        cosine learning rate schedule (with warmup).
    min_learning_rate (float): alpha_min, the minimum / final learning rate for
        the cosine learning rate schedule (with warmup).
    warmup_iters (int): T_w, the number of iterations to linearly warm-up
        the learning rate.
    cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

Returns:
    Learning rate at the given iteration under the specified schedule.
"""
class CosineLRScheduler:
    def __init__(self, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
        self.max_lr = max_learning_rate
        self.min_lr = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
    
    def get_learning_rate(self, t: int) -> float:
        if t < self.warmup_iters: # 开始lr 要够大，迅速上升
            return self.max_lr * t / self.warmup_iters
        elif t <= self.cosine_cycle_iters: # 中途cos cycle. 从0.5(1 + cos0) 到 0.5(1 + cospi) -- 从1到0
            cosine_decay = 0.5 * (1 + math.cos((t - self.warmup_iters) * math.pi \
                                                        / (self.cosine_cycle_iters - self.warmup_iters)))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay 
        else:
            return self.min_lr
             