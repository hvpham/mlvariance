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
#from utils import progress_bar

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


def evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, a, val_ratio, retrain_mode):

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    testset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='test')
    test_holdout = testset.holdout

    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))
    csv_out_file = os.path.join(saving_dir, 'retrain_result_%s_%d.csv' % (retrain_mode, val_ratio))

    f = open(csv_out_file, "w")
    f.write("Run")
    f.write(",test_accu,test_holdout,test_normal")
    f.write("\n")

    for run in range(NO_RUNS):
        # if retrain_mode == 'std_conf':
        #    rtm = ''
        # else:
        rtm = '_' + retrain_mode

        outputs_saving_file = os.path.join(saving_dir, 'outputs_%d%s_%d' % (run, rtm, val_ratio))

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

        test_report = create_report(test_outputs['test_outputs'], test_outputs['test_targets'], test_holdout)

        f.write("%d" % run)
        f.write(",%f,%f,%f" % (test_report[0][1], test_report[1][1], test_report[2][1]))
        f.write("\n")

    f.close()


def main():

    parser = argparse.ArgumentParser(description='Analyze accuracy with holdout CIFAR-100 and Resnet18')
    parser.add_argument('data_folder', help='data folder')
    parser.add_argument('result_folder', help='result folder')
    parser.add_argument('mode', choices=['holdout'], help='the mode')
    parser.add_argument('holdout_class',
                        choices=['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray'],
                        help='the holdout class')
    parser.add_argument('ratio', help='the ratio of holdout')
    parser.add_argument('val_ratio', help='the ratio of additional validation data in the retrain')
    parser.add_argument('retrain_mode', help='the retrain mode')
    parser.add_argument('number_of_runs', help='the the number of runs')

    args = parser.parse_args()

    data_folder = args.data_folder
    result_folder = args.result_folder
    mode = args.mode

    holdout_class = args.holdout_class
    ratio = int(args.ratio)
    val_ratio = int(args.val_ratio)
    retrain_mode = int(args.retrain_mode)

    NO_RUNS = int(args.number_of_runs)

    print("Evaluate mode:%s holdout:%s ratio:%d val ratio:%d retrain mode:%s" % (mode, holdout_class, ratio, val_ratio, retrain_mode))
    evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, ratio, val_ratio, retrain_mode)


if __name__ == "__main__":
    main()
