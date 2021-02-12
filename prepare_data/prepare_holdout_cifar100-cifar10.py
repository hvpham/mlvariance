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
parser.add_argument('data_folder', help='data folder path')
parser.add_argument('mode', choices=['holdout'], help='data prepare mode')

args = parser.parse_args()

data_folder = args.data_folder
mode = args.mode

base_folder = 'cifar100/cifar-100-python'

train_file = 'train'
test_file = 'test'

NEWDATA_BASE_FOLDER = 'cifar100/'+mode+'_%s'

def load_data(root, base_folder, file_name):
    data = []
    fine_targets = []
    coarse_targets = []

    file_path = os.path.join(root, base_folder, file_name)
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        data = entry['data']
        fine_targets = entry['fine_labels']
        coarse_targets = entry['coarse_labels']

    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    return data, coarse_targets, fine_targets

def store_data(data, target, holdout, root, base_folder, file_name):
    entry = {}
    entry['data'] = data
    entry['labels'] = target
    entry['holdout'] = holdout

    folder_path = os.path.join(root, base_folder)
    os.makedirs(folder_path,exist_ok=True)

    file_path = os.path.join(root, base_folder, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(entry, f)

def load_meta(root, base_folder):
    path = os.path.join(root, base_folder, 'meta')
    with open(path, 'rb') as infile:
        data = pickle.load(infile, encoding='latin1')
        fine_classes = data['fine_label_names']
        coarse_classes = data['coarse_label_names']
    #class_to_idx = {_class: i for i, _class in enumerate(classes)}
    return coarse_classes, fine_classes

def coarse_filter(data, coarse_targets, fine_targets, coarse_classes, filter_list):
    filter_idxs = [i for i, coarse_target in enumerate(coarse_targets) if coarse_classes[coarse_target] in filter_list]
    new_data = np.delete(data, filter_idxs, 0)
    new_coarse_targets = [ t for i, t in enumerate(coarse_targets) if i not in filter_idxs]
    new_fine_targets = [ t for i, t in enumerate(fine_targets) if i not in filter_idxs]
    return new_data, new_coarse_targets, new_fine_targets

def fine_filter(data, coarse_targets, fine_targets, fine_classes, filter_list, ratio):
    filter_idxs = [i for i, fine_target in enumerate(fine_targets) if fine_classes[fine_target] in filter_list]
    
    exclude_inds = filter_idxs[:int(ratio*len(filter_idxs)/10)]

    new_data = np.delete(data, exclude_inds, 0)
    new_coarse_targets = [ t for i, t in enumerate(coarse_targets) if i not in exclude_inds]
    new_fine_targets = [ t for i, t in enumerate(fine_targets) if i not in exclude_inds]
    return new_data, new_coarse_targets, new_fine_targets

def map_targets(coarse_targets, new_classes, coarse_classes):
    class_to_idx = {_class: i for i, _class in enumerate(new_classes)}
    new_targets = [class_to_idx[coarse_classes[t]] for t in coarse_targets]

    return new_targets

def holdout_indexs(targets, classes, holdout_class):
    holdout = [(classes[t] in holdout_class) for t in targets]

    return holdout

train_data, train_coarse_targets, train_fine_targets = load_data(data_folder, base_folder, train_file)
test_data, test_coarse_targets, test_fine_targets = load_data(data_folder, base_folder, test_file)

coarse_classes, fine_classes = load_meta(data_folder, base_folder)

COARSE_FILTER_LIST = ['food_containers',
    'household_electrical_devices',
    'household_furniture',
    'large_carnivores',
    'large_man-made_outdoor_things',
    'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores',
    'non-insect_invertebrates',
    'vehicles_1',
    'vehicles_2']

NEW_CLASSES = ['aquatic_mammals',
    'fish',
    'flowers',
    'fruit_and_vegetables',
    'insects',
    'medium_mammals',
    'people',
    'reptiles',
    'small_mammals',
    'trees']

train_data, train_coarse_targets, train_fine_targets = \
    coarse_filter(train_data, train_coarse_targets, train_fine_targets, coarse_classes, COARSE_FILTER_LIST)
test_data, test_coarse_targets, test_fine_targets = \
    coarse_filter(test_data, test_coarse_targets, test_fine_targets, coarse_classes, COARSE_FILTER_LIST)


holdout_class_list = ['ray', 'mushrooms', 'caterpillar', 'porcupine', 'baby', ['boy', 'man'], ['girl', 'woman'], ['boy', 'girl'], ['man', 'woman']]


for a in range(10):
    for holdout_class in holdout_class_list:
        if type(holdout_class) is list:
            holdout_class_name = '-'.join(holdout_class)
        else:
            holdout_class_name = holdout_class
            holdout_class = [holdout_class]

        new_base_folder = NEWDATA_BASE_FOLDER % (("%s_%d") % (holdout_class_name, a))

        new_train_data, new_train_coarse_targets, new_train_fine_targets = \
            fine_filter(train_data, train_coarse_targets, train_fine_targets, fine_classes, holdout_class, a)
        
        new_train_targets = map_targets(new_train_coarse_targets, NEW_CLASSES, coarse_classes)
        new_train_holdout_labels = holdout_indexs(new_train_fine_targets, fine_classes, holdout_class)

        new_test_targets = map_targets(test_coarse_targets, NEW_CLASSES, coarse_classes)
        new_test_holdout_labels = holdout_indexs(new_train_fine_targets, fine_classes, holdout_class)
        
        new_train_data, new_val_data, new_train_targets, new_val_targets, new_train_holdout_labels, new_val_holdout_labels = \
            train_test_split(new_train_data, new_train_targets, new_train_holdout_labels, test_size=0.1, random_state=69)

        store_data(new_train_data, new_train_targets, new_train_holdout_labels, data_folder, new_base_folder, 'train_batch')
        store_data(new_val_data, new_val_targets, new_train_holdout_labels, data_folder, new_base_folder, 'val_batch')
        store_data(test_data, new_test_targets, new_test_holdout_labels, data_folder, new_base_folder, 'test_batch')
    
    