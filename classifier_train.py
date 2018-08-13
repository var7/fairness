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
############## pytorch imports #############
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader

############## custom imports #############
from dataloader import FaceScrubDataset, TripletFaceScrub, SiameseFaceScrub
from dataloader import FaceScrubBalancedBatchSampler

from networks import *
from losses import OnlineTripletLoss
from openface.loadOpenFace import prepareOpenFace
from utils import save_checkpoint, save_hyperparams, AverageMeter, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector

DATA_PATH = '/home/s1791387/facescrub-data/new_data_max/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train_full_with_ids.txt')
VALID_PATH = os.path.join(DATA_PATH, 'val_full_with_ids.txt')
TEST_PATH = os.path.join(DATA_PATH, 'test_full_with_ids.txt')
WEIGHTS_PATH = '/home/s1791387/facescrub-data/new_data_max/openface_model_weigths/job_semi_std_cos3_Jul_25_1000hrs/weights_75.pth'

batch_size = 64
input_size = 96
output_dim = 128
learning_rate = 1e2
num_epochs = 1
start_epoch = 0

triplet_margin = 1.  # margin
triplet_p = 2  # norm degree for distance calculation

resume_training = True
workers = 8
use_cuda = True

cuda = False
pin_memory = False
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cuda = True
    cudnn.benchmark = True
    pin_memory = True
else:
    device = torch.device("cpu")

print('Device set: {}'.format(device))
print('Training set path: {}'.format(TRAIN_PATH))
print('Training set Path exists: {}'.format(os.path.isfile(TRAIN_PATH)))

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


train_df = FaceScrubDataset(
    txt_file=TRAIN_PATH, root_dir=DATA_PATH, transform=data_transforms['val'])

val_df = FaceScrubDataset(
    txt_file=VALID_PATH, root_dir=DATA_PATH, transform=data_transforms['val'])

siamese_train_df = SiameseFaceScrub(train_df, train=True)
print('Train data converted to siamese form. Length: {}'.format(len(siamese_train_df)))

siamese_val_df=SiameseFaceScrub(val_df, train=False)
print('Validation data converted to siamese form. Length: {}'.format(
    len(siamese_val_df)))

train_loader=torch.utils.data.DataLoader(
        siamese_train_df, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=workers)
print('Train loader created. Length of train loader: {}'.format(
        len(train_loader)))
    
val_loader=torch.utils.data.DataLoader(
        siamese_val_df, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=workers)
print('Val loader created. Length of train loader: {}'.format(
        len(val_loader)))


openface = prepareOpenFace(useCuda=cuda)
params = sum(p.numel() for p in openface.parameters() if p.requires_grad)
print('Number of params in network {}'.format(params))

en_optimizer=optim.Adam(openface.parameters(), lr=learning_rate)

T_max = num_epochs
eta_min = 0.01
en_scheduler = lr_scheduler.CosineAnnealingLR(en_optimizer, T_max=T_max, eta_min=eta_min)

classifier = ClassNet(input_size=output_dim, training=True)
cl_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

T_max = num_epochs
eta_min = 0.01
cl_scheduler = lr_scheduler.CosineAnnealingLR(cl_optimizer, T_max=T_max, eta_min=eta_min)
cl_criterion = nn.BCEWithLogitsLoss()

if resume_training:
    resume_weights=WEIGHTS_PATH
    if cuda:
        checkpoint=torch.load(resume_weights)
    else:
        # Load GPU model on CPU
        checkpoint=torch.load(resume_weights,
                                map_location=lambda storage,
                                loc: storage)

    start_epoch=checkpoint['epoch']
    openface.load_state_dict(checkpoint['state_dict'])
    en_optimizer.load_state_dict(checkpoint['optimizer'])
    best_loss = checkpoint['best_loss']
    # scheduler.load_state_dict(checkpoint['scheduler'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
        resume_weights, checkpoint['epoch']))
#     for epoch in range(0, start_epoch):
#         en_scheduler.step()

if cuda:
    openface.cuda()
    classifier.cuda()
    print('Sent model to gpu {}'.format(
        next(openface.parameters()).is_cuda))
        
def train(train_loader, classifier, encoder, criterion, en_optimizer, cl_optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    print_freq=1
    # switch to train mode
    classifier.train()
    encoder.train()

    end = time.time()
    for batch_idx, ([imgs1,imgs2], [labels1, labels2], target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        target = target.to(device).float()
#         print(target.shape, target)
        embed1, _ = encoder(imgs1)
        embed2, _ = encoder(imgs2)
        pair_embed = torch.cat((embed1, embed2), dim=1)
#         print(pair_embed.shape)
        pred_target = classifier(pair_embed)
        pred_target.squeeze_()
#         print(pred_target.squeeze_())
#         print(pred_target.shape)
        loss = cl_criterion(pred_target, target)
#         print(loss)
        losses.update(loss.item(), imgs1[0].size(0))

        en_optimizer.zero_grad()
        cl_optimizer.zero_grad()

        loss.backward()
        en_optimizer.step()
        cl_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        if batch_idx == 20:
            break
    return losses.avg

def validate(val_loader, classifier, encoder, criterion, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    print_freq=1
    # switch to evaluate mode
    classifier.eval()
    encoder.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, ([imgs1,imgs2], [labels1, labels2], target) in enumerate(val_loader):
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            target = target.to(device).float()
    #         print(target.shape, target)
            embed1, _ = openface(imgs1)
            embed2, _ = openface(imgs2)
            pair_embed = torch.cat((embed1, embed2), dim=1)
    #         print(pair_embed.shape)
            pred_target = classifier(pair_embed)
            pred_target.squeeze_()
    #         print(pred_target.squeeze_())
    #         print(pred_target.shape)
            loss = cl_criterion(pred_target, target)
    #         print(loss)
            losses.update(loss.item(), imgs1[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       batch_idx, len(val_loader), batch_time=batch_time, loss=losses))
            if batch_idx == 20:
                break
    return losses.avg
    
print('-'*10)
print('Beginning Training')
train_losses = []
val_losses = []
epoch_time = AverageMeter()
ep_end = time.time()
for epoch in range(start_epoch, start_epoch + num_epochs):

    en_scheduler.step()
    cl_scheduler.step()

    # train
    train_loss = train(train_loader, classifier, openface, cl_criterion, en_optimizer, cl_optimizer, epoch, device)
    train_losses.append(train_loss)
    # validate
    print('-'*10)
    val_loss = validate(val_loader, classifier, openface, cl_criterion, epoch, device)

    print('Avg validation loss: {}'.format(val_loss))
    val_losses.append(val_loss)

    state = {
        'epoch': epoch,
        'state_dict': openface.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_loss': best_loss
        # 'scheduler': scheduler.state_dict()
    }
    print('-' * 20)
    epoch_time.update(time.time() - ep_end)
    ep_end = time.time()
    print('Epoch {}/{}\t'
          'Time {epoch_time.val:.3f} sec ({epoch_time.avg:.3f} sec)'.format(epoch, start_epoch + num_epochs - 1, epoch_time=epoch_time))
    print('-'*20)

print('Finished training')
