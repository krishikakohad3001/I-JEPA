import torch

def gpu_timer(closure, log_timings = True):
    log_timings = log_timings and torch.cuda.is_available()
    elapsed_time=-1,
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)    # start flag
        end = torch.cuda.Event(enable_timing=True)      # end flag
        start.record()                                  # place start flag
 
    result = closure()                                  # run the function
 
    if log_timings:
        end.record()                                    # place end flag
        torch.cuda.synchronize()                        # wait for GPU to finish,as sometime python can finish ahead
        elapsed_time = start.elapsed_time(end)          # time in ms
 
    return result, elapsed_time                         # (function's output, time)
 
class CSVLogger(object):
 
    def __init__(self, fname, *argv):                 
        self.fname = fname
        self.types = []                                 # stores formats like '%d'
        # print headers
        with open(self.fname, '+a') as f:               # open file in append mode
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])                 # save format
                if i < len(argv):
                    print(v[1], end=',', file=f)        # column name + comma
                else:
                    print(v[1], end='\n', file=f)       # last name + new line
 
    def log(self, *argv):                               # argv = values for one row
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):  # pair format with value
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)   # e.g. '%.5f' % 0.873 -> 0.87300
 
class AverageMeter(object):
    """computes and stores the average and current value"""
 
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0                                   
        self.avg = 0                                 
        self.max = float('-inf')                       
        self.min = float('inf')                         
        self.sum = 0                                    
        self.count = 0                                  
 
    def update(self, val, n=1):                         # n = weight
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass                                        # skip if not comparable
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def grad_logger(named_params):                          # list of (name, weight) pairs
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):  # only main weight grids
            grad_norm = float(torch.norm(p.grad.data))  # size of gradient as one number
            stats.update(grad_norm)
            if 'qkv' in n:                              # attention layer
                stats.last_layer = grad_norm            # keeps the last one
                if stats.first_layer is None:
                    stats.first_layer = grad_norm       # keeps the first one
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.       # avoid crash if none found
    return stats
 