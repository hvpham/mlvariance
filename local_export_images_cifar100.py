'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torchvision.datasets import VisionDataset

import pickle

from PIL import Image

import numpy as np

#torchvision.datasets.CIFAR10

class CIFAR100_HOLDOUT(VisionDataset):
    train_file = 'train_batch'
    val_file = 'val_batch'
    test_file = 'test_batch'
    
    def __init__(self, holdoutroot, mode='train', transform=None, target_transform=None):

        super(CIFAR100_HOLDOUT, self).__init__(holdoutroot, transform=transform,
                                      target_transform=target_transform)

        self.mode = mode  # training set or test set

        if self.mode == 'train':
            file_name = self.train_file
        elif self.mode == 'val':
            file_name = self.val_file
        else:
            file_name = self.test_file

        self.data = []
        self.targets = []

        file_path = os.path.join(holdoutroot, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['labels']
            self.holdout = entry['holdout']
            self.ids = entry['ids']

        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def exportAllImages(data_folder, mode, holdout_class, holdout_target, a, dataset_mode):

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    dataset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode=dataset_mode)
    data_holdout = dataset.holdout
    ids = dataset.ids

    saving_root = os.path.join(holdoutroot, dataset_mode)
    os.makedirs(saving_root,exist_ok=True)

    for index in range(len(dataset)):
        img, target = dataset[index]
        if target == holdout_target:
            saving_file = os.path.join(saving_root, '%d_%s_%d.jpg' % (target, data_holdout[index], ids[index]))
            img.save(saving_file)
        


parser = argparse.ArgumentParser(description='Export images for CIFAR100')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')

args = parser.parse_args()

data_folder = args.data_folder
mode = args.mode

holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
holdout_targets = [6,6,6,6,6,4,3,5,1]

#holdout_classes = ['baby']
#holdout_targets = [6]


for i in range(len(holdout_classes)):
    holdout_class = holdout_classes[i]
    holdout_target = holdout_targets[i]
    for a in range(11):
    #for a in [0,3,6,9]:
    #for a in [1,2,4,5,7,8]:
    #for a in [0,9]:
        exportAllImages(data_folder, mode, holdout_class, holdout_target, a, 'train')
        exportAllImages(data_folder, mode, holdout_class, holdout_target, a, 'val')
        exportAllImages(data_folder, mode, holdout_class, holdout_target, a, 'test')

#exportAllImages(data_folder, mode, 'ray', 1, 9, 'val')
#exportAllImages(data_folder, mode, 'ray', 1, 9, 'test')



        