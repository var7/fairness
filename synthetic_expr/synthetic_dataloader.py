import os
import torch
import pandas as pd

import numpy as np
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import BatchSampler

class GenderDataset(Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.colors_to_class = {'red': 1, 'green': 0}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        color = os.path.splitext(os.path.basename(self.dataset.samples[idx][0]))[0].split('_')[1]
        img, shape = self.dataset[idx]
        
        return img, self.colors_to_class[color]
        

class ShapeGenderDataset(Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.colors_to_class = {'red': 1, 'green': 0}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        color = os.path.splitext(os.path.basename(self.dataset.samples[idx][0]))[0].split('_')[1]
        img, shape = self.dataset[idx]
        
        return img, shape, self.colors_to_class[color]