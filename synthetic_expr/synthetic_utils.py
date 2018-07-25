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
def train_epoch(model, X, Y, opt, criterion, batch_size=64):
    model.train()
    losses = []
    acc = []
    for beg_i in range(0, X.shape[0], batch_size):
        x_batch = X[beg_i:beg_i + batch_size]
        y_batch = Y[beg_i:beg_i + batch_size]

        x_batch = torch.from_numpy(x_batch).to(device).float()
        y_batch = torch.from_numpy(y_batch).to(device).float()

        opt.zero_grad()
        # (1) Forward
        y_hat = model(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()

        losses.append(loss.item())

        preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)

        acc.append(accuracy)
    return losses, acc

def train_encoder_classifier_epoch(encoder, classifier, X, Y, encoder_opt, classifier_opt, criterion, device, batch_size=64):
    encoder.train()
    classifier.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    for beg_i in range(0, X.shape[0], batch_size):
        x_batch = X[beg_i:beg_i + batch_size]
        y_batch = Y[beg_i:beg_i + batch_size]

        x_batch = torch.from_numpy(x_batch).to(device).float()
        y_batch = torch.from_numpy(y_batch).to(device).float()

        x_batch = x_batch.permute((0, 3, 1, 2))

        classifier_opt.zero_grad()
        encoder_opt.zero_grad()

        z = encoder(x_batch)
        # (1) Forward

        y_hat = classifier(z)

        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        classifier_opt.step()
        encoder_opt.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # (3) Compute gradients
        losses.update(loss.item(), x_batch.size(0))

        preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)
        # print(y_batch)
        # print(preds)
        # print(accuracy)

        acc.update(accuracy, x_batch.size(0))
    return losses.avg, acc.avg

def validate_epoch(model, X, Y, criterion, batch_size=64):
    model.eval()
    losses = []
    acc = []
    for beg_i in range(0, X.shape[0], batch_size):
        with torch.no_grad():
            x_batch = X.iloc[beg_i:beg_i + batch_size].values
            y_batch = Y[beg_i:beg_i + batch_size]
            x_batch = torch.from_numpy(x_batch).to(device).float()
            y_batch = torch.from_numpy(y_batch).to(device).float()

            y_hat = model(x_batch)
            # (2) Compute diff
            loss = criterion(y_hat, y_batch)
            losses.append(loss.item())

            preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
            accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)

        acc.append(accuracy)
    return losses, acc

def validate_encoder_classifier_epoch(encoder, classifier, X, Y, criterion, device, batch_size=64):
    encoder.eval()
    classifier.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    with torch.no_grad():
        end = time.time()
        for beg_i in range(0, X.shape[0], batch_size):
            x_batch = X[beg_i:beg_i + batch_size]
            y_batch = Y[beg_i:beg_i + batch_size]

            x_batch = torch.from_numpy(x_batch).to(device).float()
            y_batch = torch.from_numpy(y_batch).to(device).float()

            x_batch = x_batch.permute((0, 3, 1, 2))

            z = encoder(x_batch)
            # (1) Forward

            y_hat = classifier(z)

            # (2) Compute diff
            loss = criterion(y_hat, y_batch)

            batch_time.update(time.time() - end)
            end = time.time()

            # (3) Compute gradients
            losses.update(loss.item(), x_batch.size(0))

            preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
            accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)

            acc.update(accuracy, x_batch.size(0))

    return losses.avg, acc.avg

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(21 * 21 * 16, 1000)
        self.fc2 = nn.Linear(1000, 128)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.out_acc = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out_acc(self.fc2(x))
        return x


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
