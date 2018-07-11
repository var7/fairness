import os
import sys
import time
import copy
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

use_cuda = True
cuda = False
# if gpu is to be used
if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cuda = True
else:
    device = torch.device("cpu")

print(device)


# ## Data loader and samplers

from dataloader import FaceScrubDataset, TripletFaceScrub, SiameseFaceScrub
from dataloader import FaceScrubBalancedBatchSampler

DATA_PATH = '/home/s1791387/facescrub-data/new_data_max/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train_full_with_ids.txt')
VALID_PATH = os.path.join(DATA_PATH, 'val_full_with_ids.txt')
TEST_PATH = os.path.join(DATA_PATH, 'test_full_with_ids.txt')
MODEL_PATH = os.path.join(DATA_PATH, 'model_weights/')
print(TRAIN_PATH)
print(os.path.isfile(TRAIN_PATH))

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


train_df = FaceScrubDataset(
    txt_file=TRAIN_PATH, root_dir=DATA_PATH, transform=data_transforms['train'])

print('Train data loaded from {}. Length: {}'.format(TRAIN_PATH, len(train_df)))
triplet_train_df = TripletFaceScrub(train_df, train=True)
print('Train data converted to triplet data')


def triplet_images_show(imgs, labels):
    imgs_ = [img.numpy().transpose((1, 2, 0)) for img in imgs]
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(imgs_[0])
    plt.title('Anchor: {}'.format(labels[0]['name']))
    plt.subplot(132)
    plt.imshow(imgs_[1])
    plt.title('Positive: {}'.format(labels[1]['name']))
    plt.subplot(133)
    plt.imshow(imgs_[2])
    plt.title('Negative: {}'.format(labels[2]['name']))


triplet_dataloader = torch.utils.data.DataLoader(
    triplet_train_df, batch_size=4, shuffle=True, num_workers=1)


from losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss


from networks import *

inception = models.inception_v3(pretrained=True)
inception.aux_logits = False
num_ftrs = inception.fc.in_features
inception.fc = torch.nn.Linear(num_ftrs, 128)


tripletinception = TripletNet(inception)

params = list(tripletinception.parameters())
print('Number of params: {}'.format(len(params)))

tripletinception.train()

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

margin = 1.
criterion = nn.TripletMarginLoss(margin=margin, p=2)

lr = 1e-2
optimizer = optim.Adam(tripletinception.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

n_epochs = 20

triplet_dataloader = torch.utils.data.DataLoader(
    triplet_train_df, batch_size=4, shuffle=True, num_workers=4)
print('length of dataset {}'.format(len(triplet_train_df)))
tripletinception.train()

if cuda:
    tripletinception.cuda()
    print('sent model to gpu {}'.format(
        next(tripletinception.parameters()).is_cuda))
print('length of dataloader: {}'.format(len(triplet_dataloader)))

for epoch in range(n_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(tripletinception.state_dict())

    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    running_loss = 0.0

    for i, (imgs, labels) in enumerate(triplet_dataloader):
        optimizer.zero_grad()
        imgs = [img.to(device) for img in imgs]
        embed_anchor, embed_pos, embed_neg = tripletinception(
            imgs[0], imgs[1], imgs[2])

        loss = criterion(embed_anchor, embed_pos, embed_neg)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # best_model_wts = copy.deepcopy(tripletinception.state_dict())

        if i % 500 == 0:
            print('epoch number: {} batch number: {} loss: {}'.format(epoch, i, running_loss/500))
    
    state = {
        'epoch': epoch,
        'state_dict': tripletinception.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    job_number = sys.argv[1]
    model_path = '/home/s1791387/model_weigths/job_{}/'.format(job_number)
    model_name = model_path+'weights_{}.pth'.format(epoch)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, model_name)
    print('saved model to {}'.format(model_name))
print('finished training')
