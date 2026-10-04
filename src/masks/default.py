
# fallback for code paths that need no masking /9e.g. evalution) 
# I-JEPA pretraining uses multiblock.py

from logging import getLogger
import torch
#_GLOBAL_SEED = 0
#logger = getLogger() # A logger prints messages with a level (info / warning / error) and can send them to a file. Here it is imported out of habit; DefaultCollator never logs anything.


class DefaultCollator(object): # object means it inherits from pythons base class 
    def __call__(self, batch): # let's you call instance of that class like a regular function 

        collated_batch = torch.utils.data.default_collate(batch) # default_collate : it stacks the B image tensors into one (B, 3,224, 224) tensor and turns the B labels into a (B,) tensor
        return collated_batch, None, None # (merged batch, encoder mask, predictor mask)