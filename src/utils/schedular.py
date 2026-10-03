
import math                                         

class WarmupCosineSchedule(object):
 
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, last_epoch=-1, final_lr=0.):
        self.optimizer = optimizer                      
        self.start_lr = start_lr                        # lr at the very beginning (small)
        self.ref_lr = ref_lr                            # top lr, reached after warmup
        self.final_lr = final_lr                        # lr at the very end
        self.warmup_steps = warmup_steps                # how many steps to warm up
        self.T_max = T_max - warmup_steps               # steps left after warmup
        self._step = 0.                                 # step counter
 
    def step(self):                                     # called once every training step
        self._step += 1
        if self._step < self.warmup_steps:              # still warming up?
            progress = float(self._step) / float(max(1, self.warmup_steps))  # 0 → 1 during warmup
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)  # straight line up
        else:
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))  # 0 → 1 after warmup
            new_lr = max(self.final_lr, self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))  # smooth curve down
        for group in self.optimizer.param_groups:       # give the new lr to the optimizer
            group['lr'] = new_lr
        return new_lr
 
class CosineWDSchedule(object):
 
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.):
        self.optimizer = optimizer
        self.ref_wd = ref_wd                            # wd at the start
        self.final_wd = final_wd                        # wd at the end
        self.T_max = T_max                              # total number of steps
        self._step = 0.
 
    def step(self):                                     # called once every training step
        self._step += 1
        progress = self._step / self.T_max              # 0 → 1 over training
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))  # smooth curve
        if self.final_wd <= self.ref_wd:                # going down?
            new_wd = max(self.final_wd, new_wd)         # don't go below the end value
        else:                                           # going up?
            new_wd = min(self.final_wd, new_wd)         # don't go above the end value
        for group in self.optimizer.param_groups:       # give the new wd to the optimizer
            if ('WD_exclude' not in group) or not group['WD_exclude']:  # skip "no weight decay" groups (biases etc.)
                group['weight_decay'] = new_wd
        return new_wd
 