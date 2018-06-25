import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class FaceScrubDataset(Dataset):
    '''FaceScrub Dataset'''

    def __init__(self, txt_file, root_dir, crop_face=True, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations
            root_dit (string): Directory with all the images
            crop_face (bool): To decide to get the cropped face or full image
            transform (callable, optional): optional transform on a sample
        """
        self.faces_frame = pd.read_csv(txt_file, delimiter='\t')
        self.root_dir = root_dir
        self.transform = transform
        self.crop_face = crop_face

    def __len__(self):
        return len(self.faces_frame)

    def __getitem__(self, idx):
        self.name = self.faces_frame.iloc[idx]['name']
        self.img_id = self.faces_frame.iloc[idx]['image_id'].astype('str')
        self.face_id = self.faces_frame.iloc[idx]['face_id'].astype('str')
        self.person_id = self.faces_frame.iloc[idx]['person_id']
        self.gender = self.faces_frame.iloc[idx]['gender']
        self.bbox = self.faces_frame.iloc[idx]['bbox']
        self.bbox = list(map(int, self.bbox.split(',')))

        if self.gender == 'male':
            data_path = os.path.join(DATA_PATH, 'actor')
        elif self.gender == 'female':
            data_path = os.path.join(DATA_PATH, 'actress')

        if self.crop_face:
            img_name = self.name.replace(' ', '_') + '_' + self.img_id + '_' + self.face_id + '.jpeg'
            img_path = os.path.join(data_path, 'faces', self.name.replace(' ', '_'), img_name)
            img = io.imread(img_path)
        else:
            img_name = self.name.replace(' ', '_') + '_' + self.img_id + '.jpeg'
            img_path = os.path.join(data_path, 'images', self.name.replace(' ', '_'), img_name)
            img = io.imread(img_path)

        sample = {'image': img, 'name': self.name, 'person_id': self.person_id,
                'gender': self.gender, 'img_id': self.img_id, 'face_id': self.face_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomAugment(object):

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, sample):
        image, name, img_id, face_id, gender = sample['image'], sample['name'], \
            sample['img_id'], sample['face_id'], sample['gender']

        person_id = sample['person_id']

        image = np.uint8(image)

        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        return {'image': transformed_image, 'name': name,
                'person_id': person_id, 'gender': gender,
                'img_id': img_id, 'face_id': face_id}

def show_images_batch(sample_batched):
    """Show a batch of samples"""

    images_batch, names_batch = sample_batched['image'], sample_batched['name']
    face_id_batch, gender_batch = sample_batched['face_id'], sample_batched['gender']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
