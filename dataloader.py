import os
import torch
import pandas as pd

import numpy as np
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import BatchSampler


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
        self.pids = self.faces_frame.person_id.unique()
        self.names = self.faces_frame.name.unique()
        self.name_to_class = dict(zip(self.names, range(len(self.names))))
        self.gender_to_class = {'male': 0, 'female': 1}
        
    def __len__(self):
        return len(self.faces_frame)

    def __getitem__(self, idx):
        self.name = self.faces_frame.iloc[idx]['name']
        # self.img_id = self.faces_frame.iloc[idx]['image_id'].astype('str')
        self.face_id = self.faces_frame.iloc[idx]['face_id'].astype('str')
        self.person_id = self.faces_frame.iloc[idx]['person_id']
        self.gender = self.faces_frame.iloc[idx]['gender']
        self.url = self.faces_frame.iloc[idx]['url'].encode('utf-8')
        # self.bbox = self.faces_frame.iloc[idx]['bbox']
        # self.bbox = list(map(int, self.bbox.split(',')))
        

        if self.gender == 'male':
            data_path = os.path.join(self.root_dir, 'actor')
        elif self.gender == 'female':
            data_path = os.path.join(self.root_dir, 'actress')

        if self.crop_face:
            # img_name = self.name.replace(' ', '_') + '_' + self.img_id + '_' + self.face_id + '.jpeg'
            img_name = hashlib.sha1(self.url).hexdigest() + '.jpg'
            # img_path = os.path.join(data_path, 'faces', self.name.replace(' ', '_'), img_name)
            img_path = os.path.join(
                data_path, self.name.replace(' ', '_'), 'face', img_name)
            #img = io.imread(img_path)
            img = Image.open(img_path)
        else:
            img_name = self.name.replace(
                ' ', '_') + '_' + self.img_id + '.jpeg'
            img_path = os.path.join(
                data_path, self.name.replace(' ', '_'), img_name)
            #img = io.imread(img_path)
            img = Image.open(img_path)

        labels = {'name': self.name, 'person_id': self.person_id,
                  'gender': self.gender_to_class[self.gender], 'face_id': self.face_id, 
                  'class': self.name_to_class[self.name]}

        #img = Image.fromarray(img, mode='RGB')
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, labels


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


class SiameseFaceScrub(Dataset):
    '''
    Train: For each sample creates randomly a positive or a negative pairs
    Test: Creates fixed pairs for testing
    '''

    def __init__(self, facescrub_dataset, train):
        self.facescrub_dataset = facescrub_dataset
        self.train = train
        # self.transform = self.facescrub_dataset.transform

        self.pids = self.facescrub_dataset.pids
        self.pids_to_indices = {pid:
                                np.where(
                                    self.facescrub_dataset.faces_frame.person_id == pid)[0]
                                for pid in self.pids}

        if not self.train:
            random_state = np.random.RandomState(1791387)

            positive_pairs = [[i,
                               random_state.choice(self.pids_to_indices[self.facescrub_dataset.faces_frame.person_id[i].item()]), 1]
                              for i in range(0, len(self.facescrub_dataset), 2)]
            negative_pairs = [[i,
                               random_state.choice(self.pids_to_indices[
                                   np.random.choice(list(set(self.pids) - set([self.facescrub_dataset.faces_frame.person_id[i].item()])))]), 0]
                              for i in range(0, len(self.facescrub_dataset), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, labels1 = self.facescrub_dataset[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(
                        self.pids_to_indices[labels1['person_id']])
            else:
                siamese_pid = np.random.choice(
                    list(set(self.pids) - set([labels1['person_id']])))
                siamese_index = np.random.choice(
                    self.pids_to_indices[siamese_pid])
            img2, labels2 = self.facescrub_dataset[siamese_index]

        else:
            img1, labels1 = self.facescrub_dataset[self.test_pairs[index][0]]
            img2, labels2 = self.facescrub_dataset[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')

        return [img1, img2], [labels1, labels2], target

    def __len__(self):
        return len(self.facescrub_dataset)


class TripletFaceScrub(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, facescrub_dataset, train):
        self.facescrub_dataset = facescrub_dataset
        self.train = train
        # self.transform = self.facescrub_dataset.transform

        self.pids = self.facescrub_dataset.pids
        self.pids_to_indices = {pid:
                                np.where(
                                    self.facescrub_dataset.faces_frame.person_id == pid)[0]
                                for pid in self.pids}

        if not self.train:
            random_state = np.random.RandomState(1791387)

            triplets = [[i,
                         random_state.choice(
                             self.pids_to_indices[self.facescrub_dataset.faces_frame.person_id[i].item()]),
                         random_state.choice(self.pids_to_indices[
                             np.random.choice(list(set(self.pids) - set([self.facescrub_dataset.faces_frame.person_id[i].item()])))])]
                        for i in range(0, len(self.facescrub_dataset))]

            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, labels1 = self.facescrub_dataset[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(
                    self.pids_to_indices[labels1['person_id']])
            negative_pid = np.random.choice(
                list(set(self.pids) - set([labels1['person_id']])))
            negative_index = np.random.choice(
                self.pids_to_indices[negative_pid])
            img2, labels2 = self.facescrub_dataset[positive_index]
            img3, labels3 = self.facescrub_dataset[negative_index]
        else:
            img1, labels1 = self.facescrub_dataset[self.test_triplets[index][0]]
            img2, labels2 = self.facescrub_dataset[self.test_triplets[index][1]]
            img3, labels3 = self.facescrub_dataset[self.test_triplets[index][2]]

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        # img3 = Image.fromarray(img3.numpy(), mode='L')

        return (img1, img2, img3), (labels1, labels2, labels3)

    def __len__(self):
        return len(self.facescrub_dataset)


class FaceScrubBalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a facescrub dataset,
        samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.pids = self.dataset.pids
        self.pids_to_indices = {pid:
            np.where(self.dataset.faces_frame.person_id == pid)[0]
            for pid in self.pids}
        # self.pids_to_indices = self.dataset.pids_to_indices

        for pid in self.pids:
            np.random.shuffle(self.pids_to_indices[pid])

        self.used_pid_indices_count = {pid: 0 for pid in self.pids}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            persons = np.random.choice(
                self.pids, self.n_classes, replace=False)
            indices = []

            for person in persons:
                indices.extend(self.pids_to_indices[person][
                    self.used_pid_indices_count[person]:self.used_pid_indices_count[person] + self.n_samples
                ])
                self.used_pid_indices_count[person] += self.n_samples

                if self.used_pid_indices_count[person] + self.n_samples > len(self.pids_to_indices[person]):
                    np.random.shuffle(self.pids_to_indices[person])
                    self.used_pid_indices_count[person] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


def show_images_batch(sample_batched):
    """Show a batch of samples"""

    images_batch, names_batch = sample_batched['image'], sample_batched['name']
    face_id_batch, gender_batch = sample_batched['face_id'], sample_batched['gender']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
