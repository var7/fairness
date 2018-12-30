import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
from pathlib import Path

import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

from synthetic_experiments.synthetic_dataloader import ShapeGenderDataset
from synthetic_experiments.networks import LeNet, ClassNet_nosig

input_size = 96

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

img_transform = tv.transforms.Compose([
    tv.transforms.Resize((input_size, input_size)),
    tv.transforms.ToTensor(),
])

batch_size = 64
num_workers = 8
num_epochs = 100

DATA_PATH = Path.cwd()/'independent_dataset'

train_ds = ShapeGenderDataset(tv.datasets.ImageFolder(root=DATA_PATH/'train', transform=img_transform))
val_ds = ShapeGenderDataset(tv.datasets.ImageFolder(root=DATA_PATH/'valid', transform=img_transform))
test_ds = ShapeGenderDataset(tv.datasets.ImageFolder(root=DATA_PATH/'test', transform=img_transform))

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

encoder = LeNet().to(device)
adversary = ClassNet_nosig().to(device)
classifier = ClassNet_nosig().to(device)

adversary_loss = nn.BCEWithLogitsLoss()
classifier_loss = nn.BCEWithLogitsLoss()

encoder_opt = optim.Adam(encoder.parameters())
adversary_opt = optim.Adam(adversary.parameters())
classifier_opt = optim.Adam(classifier.parameters())

for epoch in range(num_epochs):
    UpdateEncoder = True
    for batch_num, (img, shape, color) in enumerate(train_dl):
        img = img.to(device)
        shape = shape.float()
        color = color.float()
        rep = encoder(img)

        pred_shape = classifier(rep).squeeze(1).float()
        pred_color = adversary(rep).squeeze(1).float()

        classifier_error = classifier_loss(pred_shape, shape)
        adversary_error = adversary_loss(pred_color, color)

        if UpdateEncoder:
            print('Updating encoder')
            encoder_opt.zero_grad()
            combined_error = classifier_error + adversary_error
            combined_error.backward()
            encoder_opt.step()
            UpdateEncoder = False
        else:
            classifier_error.backward(retain_graph=True)
            adversary_error.backward()
            classifier_opt.step()
            adversary_opt.step()
            classifier_opt.zero_grad()
            adversary_opt.zero_grad()
        
        classifier_preds = torch.round(pred_shape.data).cpu().numpy()
        classifier_accuracy = sum(classifier_preds == shape.numpy())/len(shape)

        adversary_preds = torch.round(pred_color.data).cpu().numpy()
        adversary_accuracy = sum(adversary_preds == color.numpy())/len(color)

        if batch_num % 10 == 0:
            UpdateEncoder = True
            print(f'Classifier Accuracy: {classifier_accuracy}')
            print(f'Adversary Accuracy: {adversary_accuracy}')
