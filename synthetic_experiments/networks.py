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

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # 96 * 96 * 3
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 6 * 46 * 46
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16 * 21 * 21
        self.fc1 = nn.Linear(21 * 21 * 16, 1000)
        self.fc2 = nn.Linear(1000, 128)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class LeNetBN(nn.Module):

    def __init__(self):
        super(LeNetBN, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # 96 * 96 * 3
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        # 6 * 46 * 46
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # 16 * 21 * 21
        self.fc1 = nn.Linear(21 * 21 * 16, 1000)
        self.fc2 = nn.Linear(1000, 128)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.leaky_relu(self.bn2((self.conv2(x)))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class LeNetBN_N(nn.Module):

    def __init__(self, output_size=16):
        super(LeNetBN_N, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # 96 * 96 * 3
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        # 6 * 46 * 46
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # 16 * 21 * 21
        self.fc1 = nn.Linear(21 * 21 * 16, 1000)
        self.fc2 = nn.Linear(1000, output_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.leaky_relu(self.bn2((self.conv2(x)))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class ClassNet(nn.Module):

    def __init__(self, input_size=128):
        super(ClassNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.out_acc = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.out_acc(self.fc2(x))
        return x

class ClassNet_N(nn.Module):

    def __init__(self, input_size=16):
        super(ClassNet_N, self).__init__()

        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.out_acc = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.out_acc(self.fc2(x))
        return x
    
class ClassNet_nosig(nn.Module):

    def __init__(self, input_size=128):
        super(ClassNet_nosig, self).__init__()

        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 1)
#         self.out_acc = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
#         x = self.out_acc(self.fc2(x))
        x = (self.fc2(x))
        return x
    
class ClassNet_nosigN(nn.Module):

    def __init__(self, input_size=16):
        super(ClassNet_nosigN, self).__init__()

        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 1)
#         self.out_acc = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
#         x = self.out_acc(self.fc2(x))
        x = (self.fc2(x))
        return x