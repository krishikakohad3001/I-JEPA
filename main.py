import argparse              # reads options you type in the terminal
import multiprocessing as mp # starts separate processes (one per GPU)
import pprint                # prints dictionaries neatly
import yaml                  # reads the .yaml config file

from src.utils.distributed import init_distributed  # sets up multi-GPU communication
from src.train import main as app_main  # the training function

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

# Runs once inside each process. rank = 0, 1, 2,  (one per GPU)
def process_main(rank, fname, world_size, devices):
    import os
    # Make this process see only its own GPU ('cuda:1' = '1').
    # Inside the process, that GPU is now called cuda:0.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
 
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    # Only rank 0 prints progress, so logs aren't repeated per GPU
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
 
    logger.info(f'called-params {fname}')
 
    #load script params (YAML file into Python dict)
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
 
    # Connect all processes so they can share gradients during training
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params)   # hand over to src/train.py
 
 
if __name__ == '__main__':
    args = parser.parse_args()
 
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')   # required for CUDA in child processes
 
    # Start one process per GPU; they all run in parallel
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()