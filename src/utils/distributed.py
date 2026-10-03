import os
import torch
import torch.distributed as dist
from logging import getLogger
logger = getLogger()

def init_distributed(port=40112, rank_and_world_size=(None, None)):
    if dist.is_available() and dist.is_initialized(): # if already initialised
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost' 
    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']

        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0

            return world_size, rank

    try :
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend = 'nccl',
            world_size = world_size,
            rank = rank
        )
    except Exception as e:
        world_size, rank = 1,0
        logger.info(f'distributed training not available {e}')

    return world_size, rank

# not needed  for JEPA
# class AllGather(torch.autograd.Function):
# class AllReduceSum(torch.autograd.Function):

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size()>1)
        ):
            x = x.contiguous()/ dist.get_world_size()
            dist.all_reduce(x)

        return x
    @ staticmethod

    def backward(ctx, grads):
        return grads

