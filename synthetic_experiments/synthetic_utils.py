import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import time

def save_checkpoint(state, save, path, filename):
    """Save checkpoint if a new best is achieved"""
    if save:
        print ("=> Saving a new model")
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(state, filename)
        print('=> saved model to {}'.format(filename))
    else:
        print ("=> Validation Accuracy did not improve")

def save_hyperparams(hyperparams, path, fname="hyperparams"):
    """Save hyper parameter details to file"""
    if not os.path.exists(path):
        os.makedirs(path)
    fname = fname + ".txt"
    with open(os.path.join(path, fname), 'w') as fo:
        for k, v in hyperparams.items():
            fo.write('{} :\t {}\n'.format(k, v))
    print('Saved hyper parameter details to {}'.format(os.path.join(path, fname)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
