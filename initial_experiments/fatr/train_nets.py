############## sys imports #############
import os
import sys
import time
import copy
import argparse
import datetime
############## basic stats imports #############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import RandomState
############## pytorch imports #############
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
############## parser args #############

parser = argparse.ArgumentParser()

parser.add_argument("-j", "--job-number", dest="job_number",
                    help="job number to store weights")
parser.add_argument("-e", "--epochs", dest="epochs",
                    default=20, type=int, help="Number of epochs")
parser.add_argument("-r", "--resume-training", dest="resume",
                    action="store_true", help="Resume training")
parser.add_argument("-rw", "--weigths", dest="resume_weights",
                    help="Path to weights file")
# parser.set_defaults(resume=False)
parser.add_argument("-nc", "--no-cuda", dest='use_cuda',
                    action='store_false', help="Do not use cuda")
# parser.set_defaults(use_cuda=True)
parser.add_argument("-d", "--data-path", dest="data_path",
                    help="path to data files")
parser.add_argument("-bs", "--batch-size", default=8,
                    dest="batch_size", type=int, help="batch size")
parser.add_argument("-lr", "--learning-rate", default=1e-2,
                    dest="learning_rate", type=float, help="learning rate")
parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true', help='use multiple gpus')
parser.add_argument('--print-freq', dest='print_freq', default=100, type=int, help='frequency for printing stats')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--debug', action='store_true', help='flag for debugging. If true only 1 mini batch is run')

############## DATA stuff #############
DATA_PATH = './uci_adult/'
TRAIN_PATH = os.path.join(DATA_PATH, 'adult.names')
TEST_PATH = os.path.join(DATA_PATH, 'adult.test')

train_df = pd.read_csv(TRAIN_PATH, header=None, delimiter=r"\s+",)
