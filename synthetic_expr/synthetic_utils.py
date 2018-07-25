import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

        preds = torch.round(y_hat.data).squeeze(1).numpy()
        accuracy = sum(preds == y_batch).numpy()/len(y_batch)

        acc.append(accuracy)
    return losses, acc

def train_encoder_classifier_epoch(encoder, classifier, X, Y, encoder_opt, classifier_opt, criterion, device, batch_size=64):
    encoder.train()
    classifier.train()
    losses = []
    acc = []
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

        losses.append(loss.item())

        preds = torch.round(y_hat.data).squeeze(1).numpy()
        accuracy = sum(preds == y_batch).numpy()/len(y_batch)

        acc.append(accuracy)
    return losses, acc

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

            preds = torch.round(y_hat.data).squeeze(1).numpy()
            accuracy = sum(preds == y_batch).numpy()/len(y_batch)

        acc.append(accuracy)
    return losses, acc
