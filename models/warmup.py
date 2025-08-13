import numpy as np
import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, fix_first_lr: float=1e-8):
        self.warmup = num_warmup_steps
        self.max_num_epoch = num_training_steps
        self.fix_first_lr = fix_first_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(base_lr * lr_factor, self.fix_first_lr) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epoch))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
