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

from resnet import ResNet18
from utils import progress_bar

from torchvision.datasets import VisionDataset

import pickle

from PIL import Image

import numpy as np

# torchvision.datasets.CIFAR10

import cluster_single_val_evaluate_retrain_cifar100

import traceback

parser = argparse.ArgumentParser(
    description='Run experiment with holdout CIFAR-10 and Resnet18')
parser.add_argument('number_of_run', help='number of runs')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder
mode = args.mode

NO_RUNS = int(args.number_of_run)

#holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_class_list = ['porcupine', 'ray']
#holdout_class_list = ['mushroom']

#holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']

# holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum',
#                      'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal',
#                      'boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree',
#                      'apple', 'girl', 'orange', 'rose', 'cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale']

aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']  # 0
fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']  # 1
flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']  # 2
fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']  # 3
insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']  # 4
medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']  # 5
people_list = ['baby', 'boy', 'girl', 'man', 'woman']  # 6
reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']  # 7
small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']  # 8
trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']  # 9

holdout_class_list = []
holdout_class_list.extend(aquatic_mammals_list)  # 0
holdout_class_list.extend(fish_list)  # 1
holdout_class_list.extend(flower_list)  # 2
holdout_class_list.extend(fruit_and_vegetables_list)  # 3
holdout_class_list.extend(insects_list)  # 4
holdout_class_list.extend(medium_mammals_list)  # 5
holdout_class_list.extend(people_list)  # 6
holdout_class_list.extend(reptiles_list)  # 7
holdout_class_list.extend(small_mammals_list)  # 8
holdout_class_list.extend(trees_list)  # 9


for holdout_class in holdout_class_list:
    for retrain_mode in ['std_avg_conf']:
        #for retrain_mode in ['single_conf']:
        # for retrain_mode in ['random', 'std_conf', 'avg_conf', 'conf_single']:
        # for retrain_mode in ['random']:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        # for ratio in range(11):
        for ratio in [0, 5, 10]:
            # for ratio in [10]:
            for val_ratio in [0, 1, 2, 5, 10]:
                print("Evaluate retrain mode:%s holdout:%s ratio:%d val_ratio:%d retrain_mode:%s" % (
                    mode, holdout_class, ratio, val_ratio, retrain_mode))
                try:
                    cluster_single_val_evaluate_retrain_cifar100.evaluate_model(
                        NO_RUNS, data_folder, result_folder, mode, holdout_class, ratio, val_ratio, retrain_mode)
                except:
                    traceback.print_exc()
