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
from synthetic_utils import *
from sklearn.model_selection import train_test_split
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
parser.add_argument("-bs", "--batch-size", default=64,
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
parser.add_argument('--cosine', action='store_true',
                    help='use cosine annealing instead of step annealing for learning rate annealing')
parser.add_argument('--lr-restart', action='store_true',
                    help='reset the learning rate')
group = parser.add_mutually_exclusive_group()
group.add_argument('--semi-hard', action='store_true', help='will do online selection with semi-hard negatives')
group.add_argument('--hardest', action='store_true', help='will do online triplet selection with hardest negatives')

best_acc = 0

def main():
    global args, best_acc
    args = parser.parse_args()
    print(args)

    if args.data_path is not None:
        if os.path.exists(args.data_path):
            DATA_PATH = args.data_path
        else:
            print('Data path: {} does not exist'.format(args.data_path))
    else:
        DATA_PATH = './generated_data.pkl'
        print('No data path provided. Setting to default: {}'.format(DATA_PATH))

    CURR_DATE = "{}_{}_{}00hrs".format(time.strftime(
        "%b"), time.strftime("%d"), time.strftime("%H"))
    JOB_NUMBER = "{}_{}".format(
        args.job_number, CURR_DATE) if args.job_number is not None else CURR_DATE
    WEIGHTS_PATH = os.path.join(
        DATA_PATH, 'synthetic_weights', 'job_{}'.format(JOB_NUMBER))
    ############## hyper parameters #############
    batch_size = args.batch_size
    input_size = 96
    output_dim = 128
    num_epochs = args.epochs
    start_epoch = 0
    resume_training = args.resume
    rs = np.random.RandomState(1791387)
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

    with open(DATA_PATH, 'rb') as f:
        imgs, shapes, colors = pickle.load(f)

    print('Imgs shape: {}, shapes shape: {}, colors shape: {}'.format(imgs.shape, shapes.shape, colors.shape))
    imgs = imgs/255.
    imgs_train, imgs_test, shapes_train, shapes_test, colors_train, colors_test = train_test_split(
    imgs, shapes, colors, test_size=0.25, stratify=shapes, random_state=rs)
    ############## set up models #############
    encoder = LeNet()
    classifier = ClassNet()

    criterion = nn.BCEWithLogitsLoss()
    opt_cls = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    opt_enc = optim.Adam(encoder.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))#
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
        encoder.load_state_dict(checkpoint['encoder_state'])
        classifier.load_state_dict(checkpoint['classifier_state'])
        opt_cls.load_state_dict(checkpoint['cls_optimizer'])
        opt_enc.load_state_dict(checkpoint['enc_optimizer'])
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
            print('Best loss loaded as {}'.format(best_acc))
        # scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
            resume_weights, checkpoint['epoch']))

    if args.cosine:
        T_max = args.epochs
        eta_min = 0.01
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        for epoch in range(0, start_epoch):
            scheduler.step()
    ############## Send model to GPU ############
    if cuda:
        encoder.cuda()
        classifier.cuda()
        print('Sent model to gpu {} {}'.format(
            next(encoder.parameters()).is_cuda,
            next(classifier.parameters()).is_cuda))
    ############## Save Hyper params to file ############
    hyperparams = {
        'JOB_NUMBER': JOB_NUMBER,
        'input_size': input_size,
        'output_dim': output_dim,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'print_every': args.print_freq,
        'start_epoch': start_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
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
        train_loss, train_acc = train_encoder_classifier_epoch(encoder,
                                classifier, imgs_train, shapes_train, opt_enc,
                                opt_cls, criterion, device)
        # train_losses.append(np.mean(train_loss))
        print(train_loss)
        print(train_acc)
        # validate
        print('-'*10)
        val_loss, val_acc = validate_encoder_classifier_epoch(encoder, classifier,
                                imgs_test, shapes_test, criterion, device)
        print(val_loss)
        print(val_acc)

        # print('Avg validation loss: {}'.format(val_loss))
        # val_losses.append(val_loss)

        state = {
            'epoch': epoch,
            'encoder_state': encoder.state_dict(),
            'classifier_state': classifier.state_dict(),
            'cls_optimizer': opt_cls.state_dict(),
            'enc_optimizer': opt_enc.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_acc': best_acc
            # 'scheduler': scheduler.state_dict()
        }
        if best_acc > val_acc:
            best_acc = val_acc
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
