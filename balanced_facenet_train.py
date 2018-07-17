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

from utils import *

############## add parser arguments #############
parser = argparse.ArgumentParser()

parser.add_argument("-j", "--job-number", dest="job_number",
                    help="job number to store weights")
parser.add_argument("-e", "--epochs", dest="epochs",
                    default=20, type=int, help="Number of epochs")
parser.add_argument("--num-classes", default=3, type=int, help="Number of classes to sample from for the balanced sampler")
parser.add_argument("--num-samples", default=10, type=int, help="Number of samples per class for the balanced sampler")
parser.add_argument("-r", "--resume-training", dest="resume",
                    action="store_true", help="Resume training")
parser.add_argument("-rw", "--weigths", dest="resume_weights",
                    help="Path to weights file")
# parser.set_defaults(resume=False)
parser.add_argument("-nc", "--no-cuda", dest='use_cuda',
                    action='store_false', help="Do not use gpu")
# parser.set_defaults(use_cuda=True)
parser.add_argument("-d", "--data-path", dest="data_path",
                    help="path to data files")
parser.add_argument("-bs", "--batch-size", default=8,
                    dest="batch_size", type=int, help="batch size")
parser.add_argument("-lr", "--learning-rate", default=1e-2,
                    dest="learning_rate", type=float, help="learning rate")
parser.add_argument('--multi-gpu', dest='multi_gpu',
                    action='store_true', help='use multiple gpus')
parser.add_argument('--print-freq', dest='print_freq',
                    default=100, type=int, help='frequency for printing stats')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--debug', action='store_true',
                    help='flag for debugging. If true only 1 mini batch is run')
group = parser.add_mutually_exclusive_group()
group.add_argument('--semi-hard', action='store_true', help='will do online selection with semi-hard negatives')
group.add_argument('--hardest', action='store_true', help='will do online triplet selection with hardest negatives')

best_loss = 1e2


def main():
    global args, best_loss
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
        DATA_PATH, 'balanced_model_weigths', 'job_{}'.format(JOB_NUMBER))
    ############## hyper parameters #############
    batch_size = args.batch_size
    input_size = 299
    output_dim = 128
    learning_rate = args.learning_rate
    num_epochs = args.epochs
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

    train_batch_sampler = FaceScrubBalancedBatchSampler(
        train_df, n_classes=args.num_classes, n_samples=args.num_samples)
    val_batch_sampler = FaceScrubBalancedBatchSampler(
        val_df, n_classes=args.num_classes, n_samples=args.num_samples)

    print('Train data loaded from {}. Length: {}'.format(
        TRAIN_PATH, len(train_df)))
    print('Validation data loaded from {}. Length: {}'.format(
        VALID_PATH, len(val_df)))

    online_train_loader = torch.utils.data.DataLoader(
        train_df, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=args.workers)

    print('Online train loader created. Length: {}'.format(
        len(online_train_loader)))

    online_val_loader = torch.utils.data.DataLoader(
        val_df, batch_sampler=val_batch_sampler, pin_memory=True, num_workers=args.workers)
    print('Online val loader created. Length: {}'.format(
        len(online_val_loader)))

    ############## set up models #############
    inception = models.inception_v3(pretrained=True)
    inception.aux_logits = False
    num_ftrs = inception.fc.in_features
    inception.fc = nn.Linear(num_ftrs, output_dim)

    # tripletinception=TripletNet(inception)

    params = sum(p.numel() for p in inception.parameters() if p.requires_grad)
    print('Number of params in triplet inception: {}'.format(params))

    ############## set up for training #############
    print('Triplet margin: {}. Norm degree: {}.'.format(triplet_margin, triplet_p))
    # loss_fn = OnlineTripletLoss(triplet_margin, RandomNegativeTripletSelector(margin))

    # criterion=nn.TripletMarginLoss(margin=triplet_margin, p=triplet_p)
    if args.semi_hard:
        negative_selection_fn_name = "semi-hard"
        negative_selection_fn = SemihardNegativeTripletSelector(margin=triplet_margin)
    elif args.hardest:
        negative_selection_fn_name = "hardest"
        negative_selection_fn = HardestNegativeTripletSelector(margin=triplet_margin)
    else:
        negative_selection_fn_name = "random negative"
        negative_selection_fn = RandomNegativeTripletSelector(margin=triplet_margin)
    print('Triplet selection method set to {}'.format(negative_selection_fn_name))

    criterion = OnlineTripletLoss(negative_selection_fn, margin=triplet_margin)
    optimizer = optim.Adam(inception.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    ############## Load saved weights #############
    if resume_training:
        resume_weights = args.resume_weights
        if args.multi_gpu:
            if torch.cuda.device_count() > 1:
                print('Loading onto GPU')
                inception = nn.DataParallel(inception).cuda()
        if cuda:
            checkpoint = torch.load(resume_weights)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(resume_weights,
                                    map_location=lambda storage,
                                    loc: storage)

        start_epoch = checkpoint['epoch']
        inception.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
        # scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
            resume_weights, checkpoint['epoch']))
        for epoch in range(0, start_epoch):
            scheduler.step()
    ############## Send model to GPU ############
    if cuda:
        inception.cuda()
        print('Sent model to gpu {}'.format(
            next(inception.parameters()).is_cuda))
        if args.multi_gpu:
            if torch.cuda.device_count() > 1:
                print("Using {} GPUS".format(torch.cuda.device_count()))
                inception = nn.DataParallel(inception).cuda()
    ############## Save Hyper params to file ############
    hyperparams = {
        'JOB_NUMBER': JOB_NUMBER,
        'num_classes': args.num_classes,
        'num_samples': args.num_samples,
        'input_size': input_size,
        'output_dim': output_dim,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'print_every': args.print_freq,
        'start_epoch': start_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'triplet_margin': triplet_margin,
        'triplet_p': triplet_p,
        'criterion': criterion,
        'negative_selection_fn': negative_selection_fn_name
    }
    save_hyperparams(hyperparams=hyperparams, path=WEIGHTS_PATH)
    ############## Training #############
    print('-'*10)
    print('Beginning Training')
    train_losses = []
    val_losses = []
    epoch_time = AverageMeter()
    ep_end = time.time()
    for epoch in range(start_epoch, start_epoch + num_epochs):

        scheduler.step()

        # train
        train_loss = train(online_train_loader, inception, criterion, optimizer, epoch, device)
        train_losses.append(train_loss)
        # validate
        print('-'*10)
        val_loss = validate(online_val_loader, inception, criterion, device)

        print('Avg validation loss: {}'.format(val_loss))
        val_losses.append(val_loss)

        state = {
            'epoch': epoch,
            'state_dict': inception.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': best_loss
            # 'scheduler': scheduler.state_dict()
        }
        if best_loss > val_loss:
            best_loss = val_loss
            MODEL_NAME = os.path.join(
                WEIGHTS_PATH, 'weights_{}.pth'.format(epoch))
            save_checkpoint(state, True, WEIGHTS_PATH, MODEL_NAME)
        print('-' * 20)
        epoch_time.update(time.time() - ep_end)
        ep_end = time.time()
        print('Epoch {}/{}\t'
              'Time {epoch_time.val:.3f} sec ({epoch_time.avg:.3f} sec)'.format(epoch, start_epoch + num_epochs - 1, epoch_time=epoch_time))
        print('-'*20)

    print('Finished training')


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.multi_gpu:
            with torch.cuda.device(0):
                imgs = imgs.cuda(async=True)
        else:
            imgs = imgs.to(device)

        targets = labels['person_id']
        targets.to(device)

        embeddings = model(imgs)
        loss = criterion(embeddings, targets)

        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
        if args.debug:
            break

    return losses.avg


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (imgs, labels) in enumerate(val_loader):
            if args.multi_gpu:
                with torch.cuda.device(0):
                    imgs = imgs.cuda(async=True)
            else:
                imgs = imgs.to(device)

            targets = labels['person_id']
            targets.to(device)

            embeddings = model(imgs)

            loss = criterion(embeddings, targets)

            losses.update(loss.item(), imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses))
            if args.debug:
                break

    return losses.avg


if __name__ == '__main__':
    main()
