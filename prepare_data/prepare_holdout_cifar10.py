import torch
# torch.manual_seed(0)

import numpy as np
# np.random.seed(0)

import random

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os

import argparse

import pickle

from sklearn.model_selection import train_test_split

from PIL import Image, ImageOps

parser = argparse.ArgumentParser(description='Prepare the CIFAR-10 holdout dataset')
parser.add_argument('data_folder', help='learning rate')
parser.add_argument('mode', choices=['holdout', 'holdout-dup', 'augmentation', 'augmentation-all'], help='learning rate')

args = parser.parse_args()

data_folder = args.data_folder
mode = args.mode

base_folder = 'cifar10/cifar-10-batches-py'

train_list = ['data_batch_1',
                'data_batch_2',
                'data_batch_3',
                'data_batch_4',
                'data_batch_5']

test_list = ['test_batch']

def load_data(root, base_folder, file_list):
    data = []
    targets = []

    for file_name in file_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            targets.extend(entry['labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    return data, targets

train_data, train_targets = load_data(data_folder, base_folder, train_list)
test_data, test_targets = load_data(data_folder, base_folder, test_list)

NEWDATA_BASE_FOLDER = 'cifar10/'+mode+'_%s'

def store_data(data, target, root, base_folder, file_name):
    entry = {}
    entry['data'] = data
    entry['labels'] = target

    folder_path = os.path.join(root, base_folder)
    os.makedirs(folder_path,exist_ok=True)

    file_path = os.path.join(root, base_folder, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(entry, f)

def modify_images(data, modify_inds):
    new_data = data.copy()
    for modify_ind in modify_inds:
        img_data = new_data[modify_ind]
        img = Image.fromarray(img_data)
        
        gray_img = ImageOps.grayscale(img)
        new_image_data = (np.array(list(gray_img.getdata()))).reshape(32,32)
        new_image_data = np.stack([new_image_data,new_image_data,new_image_data], axis=2)
        
        new_data[modify_ind] = new_image_data

    return new_data

def holdout_dup_images(data, modify_inds, include_inds):
    new_data = data.copy()
    for modify_ind in modify_inds:
        new_data[modify_ind] = data[random.choice(include_inds)]
    
    return new_data

holdout_class = 0
if mode == 'augmentation-all':
    target_class_inds = list(range(len(train_targets)))
else:
    target_class_inds = (np.where(np.array(train_targets) == holdout_class))[0]

for a in range(10):
    new_base_folder = NEWDATA_BASE_FOLDER % (("%d_%d") % (holdout_class, a))
    exclude_inds = target_class_inds[:int(a*len(target_class_inds)/10)]
    include_inds = target_class_inds[int(a*len(target_class_inds)/10):]
    
    if mode == 'holdout':
        new_train_data = np.delete(train_data, exclude_inds, 0)
        new_train_targets = [ t for i, t in enumerate(train_targets) if i not in exclude_inds]
    elif mode == 'holdout-dup':
        new_train_data = holdout_dup_images(train_data, exclude_inds, include_inds)
        new_train_targets = train_targets
    elif mode == 'augmentation' or mode == 'augmentation-all':
        new_train_data = modify_images(train_data, exclude_inds)
        new_train_targets = train_targets

    new_train_data, new_val_data, new_train_targets, new_val_targets = train_test_split(new_train_data, new_train_targets, test_size=0.1, random_state=69)

    store_data(new_train_data, new_train_targets, data_folder, new_base_folder, 'train_batch')
    store_data(new_val_data, new_val_targets, data_folder, new_base_folder, 'val_batch')
    store_data(test_data, test_targets, data_folder, new_base_folder, 'test_batch')
    
    