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

from utils import save_checkpoint, save_hyperparams, AverageMeter, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector


# In[3]:
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data-path", dest="data_path",
                    help="path to data files")

parser.add_argument("-rw", "--weigths", dest="resume_weights",
                    help="Path to weights file")

args = parser.parse_args()

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

if args.resume_weights is not None:
    WEIGHTS_PATH = args.resume_weights
else:
    WEIGHTS_PATH = '/home/s1791387/facescrub-data/new_data_max/balanced_model_weigths/job_cosine5_semi_std_Jul_20_1900hrs/weights_65.pth'

triplet_margin = 1.  # margin
triplet_p = 2  # norm degree for distance calculation
input_size = 299
output_dim = 128
resume_training = True
use_cuda = True


# In[5]:


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


# In[6]:


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


print('Train data loaded from {}. Length: {}'.format(
    TRAIN_PATH, len(train_df)))
print('Validation data loaded from {}. Length: {}'.format(
    VALID_PATH, len(val_df)))

inception = models.inception_v3(pretrained=True)
inception.aux_logits = False
num_ftrs = inception.fc.in_features
inception.fc = nn.Linear(num_ftrs, output_dim)

criterion = nn.TripletMarginLoss(margin=triplet_margin, p=triplet_p)

# In[9]:


if resume_training:
    resume_weights = WEIGHTS_PATH
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(resume_weights,
                                map_location=lambda storage,
                                loc: storage)

    inception.load_state_dict(checkpoint['state_dict'])
    best_loss = checkpoint['best_loss']
    # scheduler.load_state_dict(checkpoint['scheduler'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
        resume_weights, checkpoint['epoch']))


if cuda:
    inception.cuda()
    print('Sent model to gpu {}'.format(
        next(inception.parameters()).is_cuda))

classes = [10, 5, 279, 330]
imgs = []
person_id = []
gender = []
count_5, count_10, count_279, count_330 = 0


for ind, row in train_df.faces_frame.iterrows():
    if row['person_id'] in classes:
        pid = row['person_id']
        if pid == 5: 
            if count_5 < 15:
                count_5 += 1
            else:
                continue
        elif pid == 10: 
            if count_10 < 15:
                count_10 += 1
            else:
                continue
        elif pid == 279: 
            if count_279 < 15:
                count_279 += 1
            else:
                continue
        elif pid == 330: 
            if count_330 < 15:
                count_330 += 1
            else:
                continue
        
        img, label = train_df[ind]
        img = img.to(device)
        img.unsqueeze_(0)
        imgs.append(img)
        person_id.append(row['person_id'])
        
        
#         embedding = inception(img)
#         train_embeddings = torch.cat((train_embeddings, embedding))
#     if count / 10 == 0: print(count)
#         thumbnails = torch.cat((thumbnails, img))
#         person_id = np.concatenate((person_id, label['person_id']))
#         gender = np.concatenate((gender, label['gender']))

train_embeddings = torch.Tensor()
thumbnails = torch.Tensor().cuda()


inception.eval()
topil = transforms.ToPILImage()
totensor = transforms.ToTensor()
resizetransform = transforms.Resize((32, 32))
with torch.no_grad():
    for ind, img in enumerate(imgs):
        embedding = inception(img)
        train_embeddings = torch.cat((train_embeddings, embedding))
        small_img = totensor(resizetransform(topil(img.squeeze_(0).cpu())))
        thumbnails = torch.cat((thumbnails.cuda(), small_img.cuda()))
        if ind % 20 == 0:
            print('{} images completed'.format(ind))


# inception.eval()
# train_embeddings = torch.Tensor()
# thumbnails = torch.Tensor()
# person_id = []
# gender = []
# with torch.no_grad():
#     for i, (imgs, labels) in enumerate(online_train_loader):
#         imgs = imgs.to(device)
#
#         embeddings = inception(imgs)
#         train_embeddings = torch.cat((train_embeddings, embeddings))
#         thumbnails = torch.cat((thumbnails, imgs))
#         person_id = np.concatenate((person_id, labels['person_id']))
#         gender = np.concatenate((gender, labels['gender']))
#
#         if i == 10: brea

def tsne(embeddings):
    import sklearn.manifold
    return torch.from_numpy(sklearn.manifold.TSNE(n_iter=250).fit_transform(embeddings.cpu().numpy()))

print(sorted(set(person_id)))
def svg(points, labels, thumbnails, legend_size=1e-1, legend_font_size=5e-2, circle_radius=5e-3):
	points = (points - points.min(0)[0]) / (points.max(0)[0] - points.min(0)[0])
	class_index = sorted(set(labels))
    class_colors = [360.0 * i / len(class_index) for i in range(len(class_index))]
	colors = [class_colors[class_index.index(label)] for label in labels]
	thumbnails_base64 = [base64.b64encode(cv2.imencode('.jpg', img.mul(255).unsqueeze_(0).permute(1, 2, 0).cpu().numpy()[..., ::-1])[1]) for img in thumbnails]
	return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">' + 	   ''.join(map('''<circle cx="{}" cy="{}" title="{}" fill="hsl({}, 50%, 50%)" r="{}" desc="data:image/jpeg;base64,{}" onmouseover="evt.target.ownerDocument.getElementById('preview').setAttribute('href', evt.target.getAttribute('desc')); evt.target.ownerDocument.getElementById('label').textContent = evt.target.getAttribute('title');" />'''.format, points[:, 0], points[:, 1], labels, colors, [circle_radius] * len(points), thumbnails_base64)) + 	   '''<image id="preview" x="0" y="{legend_size}" width="{legend_size}" height="{legend_size}" />
	   <text id="label" x="0" y="{legend_size}" font-size="{legend_font_size}" />
	   </svg>'''.format(legend_size = legend_size, legend_font_size = legend_font_size)


# In[ ]:

# print(thumbnails.shape)
# for img in thumbnails:
#     print(img.shape)
#     img = img.mul(255).unsqueeze_(0).permute(1, 2, 0).cpu().numpy()
tsne_embeddings = tsne(train_embeddings)

# In[ ]:
import visdom
vis = visdom.Visdom()

import cv2
import base64


# In[ ]:


open('train_tsne.svg', 'w').write(svg(tsne_embeddings, person_id, thumbnails))
vis.svg('train_tsne.svg')

import pickle

with open('tsne_train_embeddings.pkl', 'w') as f:
    pickle.dump(tsne_embeddings, f)
