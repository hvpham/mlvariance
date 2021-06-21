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


def evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, a):

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    trainset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='train')
    train_holdout = trainset.holdout

    valset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='val')
    val_holdout = valset.holdout

    testset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='test')
    test_holdout = testset.holdout

    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))
    csv_out_file = os.path.join(saving_dir, 'result.csv')

    f = open(csv_out_file, "w")
    f.write("Run")
    f.write(",train_accu,train_holdout,train_normal")
    f.write(",val_accu,val_holdout,val_normal")
    f.write(",test_accu,test_holdout,test_normal")
    f.write("\n")

    for run in range(NO_RUNS):
        outputs_saving_file = os.path.join(saving_dir, 'outputs_%d' % run)

        # with open(outputs_saving_file, 'rb') as rf:
        #    test_outputs = pickle.load(rf, encoding='latin1')
        test_outputs = torch.load(outputs_saving_file)

        def process(outputs, targets):
            pred_list = np.argmax(outputs, axis=1)
            correct = np.equal(pred_list, targets)
            accu = float(np.sum(np.multiply(correct, 1)))/len(targets)
            return correct, accu

        def create_report(outputs, targets, holdout):
            report = process(outputs, targets)

            test_index = [i for i, x in enumerate(holdout) if x]

            test_neg_index = np.array(list(range(len(targets))))
            test_neg_index = np.delete(test_neg_index, test_index)

            holdout_outputs = [outputs[i] for i in test_index]
            if len(holdout_outputs) != 0:
                holdout_outputs = np.vstack(holdout_outputs)
                holdout_report = process(holdout_outputs, [targets[i] for i in test_index])
            else:
                holdout_report = None

            normal_outputs = [outputs[i] for i in test_neg_index]
            normal_outputs = np.vstack(normal_outputs)
            normal_report = process(normal_outputs, [targets[i] for i in test_neg_index])

            return report, holdout_report, normal_report

        train_report = create_report(test_outputs['train_outputs'], test_outputs['train_targets'], train_holdout)
        val_report = create_report(test_outputs['val_outputs'], test_outputs['val_targets'], val_holdout)
        test_report = create_report(test_outputs['test_outputs'], test_outputs['test_targets'], test_holdout)

        f.write("%d" % run)
        if a != 10:
            f.write(",%f,%f,%f" % (train_report[0][1], train_report[1][1], train_report[2][1]))
            f.write(",%f,%f,%f" % (val_report[0][1], val_report[1][1], val_report[2][1]))
        else:
            f.write(",%f,%f,%f" % (train_report[0][1], -1.0, train_report[2][1]))
            f.write(",%f,%f,%f" % (val_report[0][1], -1.0, val_report[2][1]))
        f.write(",%f,%f,%f" % (test_report[0][1], test_report[1][1], test_report[2][1]))
        f.write("\n")

    f.close()


parser = argparse.ArgumentParser(description='Run experiment with holdout CIFAR-10 and Resnet18')
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

aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
#fish_list = ['aquarium_fish', 'flatfish', 'shark', 'trout']
flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
#fruit_and_vegetables_list = ['apple', 'orange', 'pear', 'sweet_pepper']
insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
#insects_list = ['bee', 'beetle', 'butterfly', 'cockroach']
medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
#medium_mammals_list = ['fox', 'possum', 'raccoon', 'skunk']
people_list = ['baby', 'boy', 'girl', 'man', 'woman']
#people_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman']
reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']

holdout_class_list = []
holdout_class_list.extend(aquatic_mammals_list)
holdout_class_list.extend(fish_list)
holdout_class_list.extend(flower_list)
holdout_class_list.extend(fruit_and_vegetables_list)
holdout_class_list.extend(insects_list)
holdout_class_list.extend(medium_mammals_list)
holdout_class_list.extend(people_list)
holdout_class_list.extend(reptiles_list)
holdout_class_list.extend(small_mammals_list)
holdout_class_list.extend(trees_list)

#holdout_class_list = ['caterpillar']

for holdout_class in holdout_class_list:
    # for a in [0,3,6,9]:
    for a in range(11):
        try:
            print("Evaluate mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
            evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, a)
        except:
            print("Error while evaluating mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))

#evaluate_model(NO_RUNS, data_folder, result_folder, mode, 'mushrooms', 3)
