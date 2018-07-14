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
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms, utils, models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
############## custom imports #############
from dataloader import FaceScrubDataset, TripletFaceScrub, SiameseFaceScrub
from dataloader import FaceScrubBalancedBatchSampler

from networks import *

from utils import save_checkpoint, save_hyperparams

############## add parser arguments #############
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

args = parser.parse_args()
print(args)

if args.data_path is not None:
    if os.path.exists(args.data_path):
        DATA_PATH = args.data_path
    else:
        print('Data path: {} does not exist'.format(args.data_path))
else:
    DATA_PATH = '/home/s1791387/facescrub-data/new_data_max/'
    print('No data path provided. Setting to cluster default: {}'.format(DATA_PATH))

TRAIN_PATH = os.path.join(DATA_PATH, 'train_full_with_ids.txt')
VALID_PATH = os.path.join(DATA_PATH, 'val_full_with_ids.txt')
TEST_PATH = os.path.join(DATA_PATH, 'test_full_with_ids.txt')

CURR_DATE = "{}_{}_{}00hrs".format(time.strftime(
    "%b"), time.strftime("%d"), time.strftime("%H"))
JOB_NUMBER = "{}_{}".format(
    args.job_number, CURR_DATE) if args.job_number is not None else CURR_DATE
WEIGHTS_PATH = os.path.join(
    DATA_PATH, 'model_weigths', 'job_{}'.format(JOB_NUMBER))
############## hyper parameters #############
batch_size = args.batch_size
input_size = 299
output_dim = 128
learning_rate = args.learning_rate
num_epochs = args.epochs
print_every = 100
start_epoch = 0

triplet_margin = 1.  # margin
triplet_p = 2  # norm degree for distance calculation

resume_training = args.resume
############## data loading #############
cuda = False
if args.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cuda = True
    cudnn.benchmark = True
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
    txt_file=TRAIN_PATH, root_dir=DATA_PATH, transform=data_transforms['train'])

val_df = FaceScrubDataset(
    txt_file=VALID_PATH, root_dir=DATA_PATH, transform=data_transforms['val'])

print('Train data loaded from {}. Length: {}'.format(TRAIN_PATH, len(train_df)))
print('Validation data loaded from {}. Length: {}'.format(VALID_PATH, len(val_df)))

triplet_train_df = TripletFaceScrub(train_df, train=True)
print('Train data converted to triplet form. Length: {}'.format(len(triplet_train_df)))

triplet_val_df=TripletFaceScrub(val_df, train=False)
print('Validation data converted to triplet form. Length: {}'.format(
    len(triplet_val_df)))

train_tripletloader=torch.utils.data.DataLoader(
    triplet_train_df, batch_size=batch_size, shuffle=True, num_workers=4)
print('Train loader created. Length of train loader: {}'.format(
    len(train_tripletloader)))

val_tripletloader=torch.utils.data.DataLoader(
    triplet_val_df, batch_size=batch_size, shuffle=True, num_workers=4)
print('Val triplet loader created. Length of val load: {}'.format(
    len(val_tripletloader)))

############## set up models #############
inception=models.inception_v3(pretrained=True)
inception.aux_logits=False
num_ftrs=inception.fc.in_features
inception.fc=nn.Linear(num_ftrs, output_dim)

tripletinception=TripletNet(inception)

params=list(tripletinception.parameters())
print('Number of params in triplet inception: {}'.format(len(params)))
############## set up for training #############
print('Triplet margin: {}. Norm degree: {}.'.format(triplet_margin, triplet_p))
criterion=nn.TripletMarginLoss(margin=triplet_margin, p=triplet_p)

optimizer=optim.Adam(tripletinception.parameters(), lr=learning_rate)
scheduler=lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
best_loss = 1e2

############## Load saved weights #############
if resume_training:
    resume_weights=args.resume_weights
    if args.multi_gpu:
        if torch.cuda.device_count() > 1:
            print('Loading onto GPU')
            tripletinception = nn.DataParallel(tripletinception).cuda()
    if cuda:
        checkpoint=torch.load(resume_weights)
    else:
        # Load GPU model on CPU
        checkpoint=torch.load(resume_weights,
                                map_location=lambda storage,
                                loc: storage)

    start_epoch=checkpoint['epoch']
    tripletinception.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_loss = checkpoint['best_loss']
    # scheduler.load_state_dict(checkpoint['scheduler'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
        resume_weights, checkpoint['epoch']))
    for epoch in range(0, start_epoch):
        scheduler.step()
############## Send model to GPU ############
if cuda:
    tripletinception.cuda()
    print('Sent model to gpu {}'.format(
        next(tripletinception.parameters()).is_cuda))
    if args.multi_gpu:
        if torch.cuda.device_count() > 1:
          print("Using {} GPUS".format(torch.cuda.device_count()))
          tripletinception = nn.DataParallel(tripletinception).cuda()
############## Save Hyper params to file ############
hyperparams={
    'JOB_NUMBER': JOB_NUMBER,
    'batch_size': batch_size,
    'input_size': input_size,
    'output_dim': output_dim,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs,
    'print_every': print_every,
    'start_epoch': start_epoch,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'triplet_margin': triplet_margin,
    'triplet_p': triplet_p,
    'criterion': criterion
}
save_hyperparams(hyperparams=hyperparams, path=WEIGHTS_PATH)
############## Training #############

for epoch in range(start_epoch, start_epoch + num_epochs):
    scheduler.step()

    tripletinception.train()
    print('Epoch {}/{}'.format(epoch, start_epoch + num_epochs - 1))
    print('-' * 10)

    running_loss=0.0
    losses=[]

    start_time=time.time()
    for batch_idx, (imgs, labels) in enumerate(train_tripletloader):
        batch_start=time.time()
        optimizer.zero_grad()
        
        for batch_idx, (imgs, labels) in enumerate(val_tripletloader):
            with torch.cuda.device(0):
                imgs=[img.cuda(async=True) for img in imgs]

        embed_anchor, embed_pos, embed_neg=tripletinception(
            imgs[0], imgs[1], imgs[2])

        loss = criterion(embed_anchor, embed_pos, embed_neg)

        running_loss += loss.item()
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch_idx % print_every == 0:
            batch_end=(time.time() - batch_start)
            print('batch number: {} loss: {} time: {} sec'.format(
                batch_idx, running_loss/(batch_idx+1), batch_end))

    val_loss = 0
    val_losses = []
    with torch.no_grad():
        tripletinception.eval()
        val_loss = 0
        for batch_idx, (imgs, labels) in enumerate(val_tripletloader):
            with torch.cuda.device(0):
                imgs=[img.cuda(async=True) for img in imgs]

            embed_anchor, embed_pos, embed_neg=tripletinception(
                imgs[0], imgs[1], imgs[2])

            loss = criterion(embed_anchor, embed_pos, embed_neg)

            val_loss += loss.item()
            val_losses.append(loss.item())

        val_loss = val_loss / (batch_idx + 1)

    print('Validation loss: {}'.format(val_loss))

    state={
        'epoch': epoch,
        'state_dict': tripletinception.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': losses,
        'val_losses': val_losses,
        'best_loss': best_loss
        # 'scheduler': scheduler.state_dict()
    }
    if best_loss > val_loss:
        best_loss = val_loss
        MODEL_NAME=os.path.join(WEIGHTS_PATH, 'weights_{}.pth'.format(epoch))
        save_checkpoint(state, True, WEIGHTS_PATH, MODEL_NAME)

    elapsed=(time.time() - start_time)/60
    print('Elapsed time for epoch {}: {} minutes'.format(epoch, elapsed))

print('Finished training')
