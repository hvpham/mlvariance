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

#torchvision.datasets.CIFAR10

class CIFAR10_HOLDOUT(VisionDataset):
    train_file = 'train_batch'
    val_file = 'val_batch'
    test_file = 'test_batch'
    
    def __init__(self, holdoutroot, mode='train', transform=None, target_transform=None):

        super(CIFAR10_HOLDOUT, self).__init__(holdoutroot, transform=transform,
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
    saving_dir = os.path.join(result_folder, 'cifar10', 'cifar10-%s_%d_%d' % (mode,holdout_class,a))
    csv_out_file = os.path.join(saving_dir,'result.csv')

    f = open(csv_out_file,"w")
    f.write("Run")
    f.write(",train_accu,train_holdout,train_normal")
    f.write(",val_accu,val_holdout,val_normal")
    f.write(",test_accu,test_holdout,test_normal")
    f.write("\n")

    for run in range(NO_RUNS):
        outputs_saving_file = os.path.join(saving_dir,'outputs_%d' % run)

        with open(outputs_saving_file, 'rb') as rf:
            test_outputs = pickle.load(rf, encoding='latin1')
    
        def process(outputs, targets):
            pred_list = np.argmax(outputs, axis=1)
            correct = np.equal(pred_list, targets)
            accu = float(np.sum(np.multiply(correct,1)))/len(targets)
            return correct, accu

        def create_report(outputs, targets, holdout_class):
            report = process(outputs, targets)       

            test_index = (np.where(np.array(targets) == holdout_class))[0]

            test_neg_index = np.array(list(range(len(targets))))
            test_neg_index = np.delete(test_neg_index, test_index)

            holdout_outputs = [outputs[i] for i in test_index]
            holdout_outputs = np.vstack(holdout_outputs)
            holdout_report = process(holdout_outputs, [targets[i] for i in test_index])

            normal_outputs = [outputs[i] for i in test_neg_index]
            normal_outputs = np.vstack(normal_outputs)
            normal_report = process(normal_outputs, [targets[i] for i in test_neg_index])

            return report, holdout_report, normal_report

        train_report = create_report(test_outputs['train_outputs'], test_outputs['train_targets'],holdout_class)
        val_report = create_report(test_outputs['val_outputs'], test_outputs['val_targets'],holdout_class)
        test_report = create_report(test_outputs['test_outputs'], test_outputs['test_targets'],holdout_class)

        f.write("%d" % run)
        f.write(",%f,%f,%f" % (train_report[0][1],train_report[1][1],train_report[2][1]))
        f.write(",%f,%f,%f" % (val_report[0][1],val_report[1][1],val_report[2][1]))
        f.write(",%f,%f,%f" % (test_report[0][1],test_report[1][1],test_report[2][1]))
        f.write("\n")
    
    f.close()


parser = argparse.ArgumentParser(description='Run experiment with holdout CIFAR-10 and Resnet18')
parser.add_argument('number_of_run', help='number of runs')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder

NO_RUNS = int(args.number_of_run)

modes = ['holdout', 'holdout-dup', 'augmentation', 'augmentation-all']
for mode in modes:
    for a in [0,3,6,9]:
        print("Evaluate mode:%s ratio:%d" % (mode, a))
        evaluate_model(NO_RUNS, data_folder, result_folder, mode, 0, a)