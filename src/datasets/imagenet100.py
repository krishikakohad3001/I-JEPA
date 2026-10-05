import os
import subprocess
import time
import numpy as np
from logging import getLogger
import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()

def make_stl10(
    transform,
    batch_size,
    collator = None,
    pin_mem = True,
    num_workers = 8,
    world_size = 1,
    rank = 0,
    root_path = './data',
    split='unlabeled', # 'unlabeled for pretraining'
    download = True,
    drop_last = True):
    dataset = torchvision.datasets.STL10(
        root = root_path,
        split = split,
        transform=transform,
        download=download)
    logger.info(f'STL-10 {split} dataset created : {len(dataset)} images')
    dist_sampler = torch.utils.data.DistributedSampler(
        dataset = dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size= batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('STL-10 data loader created')
    return dataset, data_loader, dist_sampler

# ----- ImageNet-100 ----
def make_imagenet100(
        transform,
        batch_size,
        collator = None,
        pin_mem=True,
        num_workers=8,
        world_size=1,
        rank=0,
        root_path=None,
        image_folder=None,
        training=True,
        copy_data=False,
        drop_last=True,
        subset_file=None):
    
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False)

    # if subset_file is not None:
    #     dataset = ImageNetSubbset(dataset, subset_file)

    logger.info('ImageNet dataset created')
    dist_sampler =torch.utils.data.distributed.DistributedSampler(
        dataset= dataset,
        num_replicas= world_size,
        rank = rank
    ) # if you have 2 GPUs, it splits your images so each GPU gets different ones, instead of both GPUs training on the same images.

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn = collator,
        sampler = dist_sampler, 
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')
    return dataset, data_loader, dist_sampler

class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
            self,
            root,
            image_folder = 'imagenet100/...',
            tar_file = 'image_100',
            transform = None,
            train = True,
            job_id = None,
            local_rank = None,
            copy_data = True,
            index_targets = False):
        
        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info("copying data locally ")
            data_path = copy_imgnt_locally(
                root = root,
                suffix = suffix,
                image_folder = image_folder,
                tar_file = tar_file,
                job_id = job_id,
                local_rank = local_rank
            )

        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super().__init__(root=data_path, transform=transform)
        logger.info("Initialized ImageNet")


        if index_targets:
            self.targets = []
            # self.samples a list of (image_path , class_label) [("dog1.jpg",0), ('cat1.jpg', 1) etc.]
            # for sample in self.samples :
            #     self.targets.append(sample[1]) # labels 

            self.targets = np.array([s[1] for s in self.samples])
            self.samples = np.array(self.samples, dtype=object)
            
            mint = None
            self.target_indices =[]

        #  Say you have 5 images and 2 classes (0 = dog, 1 = cat)

            # self.targets = [0, 1, 0, 1, 1]
            #     ^  ^  ^  ^  ^
            # image position: 0  1  2  3  4
            # for t = 0 (dogs)
            # self.targets == 0	[True, False, True, False, False]
            # np.argwhere(...)	[[0], [2]] (positions of the True values, as a column)
            # np.squeeze(...)	    [0, 2] (extra dimension removed)
            # .tolist()	        [0, 2] (a normal Python list)

            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == t)).tolist() # self.targets ==t gives True / False for each image , np.argwhere gives positions of the true values 
                self.target_indices.append(indices) # for all classes target_indices = [[0, 2], [1, 3, 4]]
                mint = len(indices) if mint is None else min(mint, len(indices))  # finding class with min number of images only that many will be used for training
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}') 

# ImageNet Subset code ...
def copy_imgnt_locally(
        root,
        suffix,
        image_folder = "",
        tar_file = "",
        job_id = None,
        local_rank = None):
    
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            os.makedirs(target, exist_ok=True)
            commands = [['tar', '-xf', source_file, '-C', target]]

            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')

            with open(tmp_sgnl_file, 'w+') as f:
                print('Done copying locally.', file = f)

        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path



    