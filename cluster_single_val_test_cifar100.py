'''Train CIFAR100 with PyTorch.'''
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

import time

#import shutil

#import gc


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


def check_done(path):
    if os.path.isfile(path):
        if os.stat(path).st_size > 0:
            return True
    return False


def test_model(run, data_folder, result_folder, mode, holdout_class, a):
    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))
    model_saving_file = os.path.join(saving_dir, 'model_%d.pth' % run)
    outputs_saving_file = os.path.join(saving_dir, 'outputs_val_%d' % run)

    if check_done(outputs_saving_file):
        print("Already evaluate for run %d with holdout class %s and a %d. Skip to the next evaluation run." % (run, holdout_class, a))
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('Preparing data..')
    tic = time.perf_counter()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, 0))

    valset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='val', transform=transform_test)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=2)

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print('Loading model')
    tic = time.perf_counter()
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    state_dict = torch.load(model_saving_file)
    net.load_state_dict(state_dict)

    net.eval()
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    def evaluate(data_loader):
        outputs_list = []
        targets_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                print("Evaluating batch: " + str(batch_idx))
                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs_list.append(outputs.cpu().numpy())
                targets_list.extend(targets.cpu())
        outputs_list = np.vstack(outputs_list)
        return outputs_list, targets_list

    results = {}

    print("Evaluating validation data")
    tic = time.perf_counter()
    results['val_outputs'], results['val_targets'] = evaluate(valloader)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print("Saving output file")
    tic = time.perf_counter()
    torch.save(results, outputs_saving_file)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))


def main():
    parser = argparse.ArgumentParser(description='Run experiment with holdout CIFAR-100 and Resnet18')
    parser.add_argument('data_folder', help='data folder')
    parser.add_argument('result_folder', help='result folder')
    parser.add_argument('mode', choices=['holdout'], help='the mode')
    parser.add_argument('holdout_class', help='the holdout class')
    parser.add_argument('ratio', help='the ratio of holdout')
    parser.add_argument('run_id', help='the id of the run')

    args = parser.parse_args()

    data_folder = args.data_folder
    result_folder = args.result_folder
    mode = args.mode

    run_id = int(args.run_id)
    holdout_class = args.holdout_class
    ratio = int(args.ratio)

    test_model(run_id, data_folder, result_folder, mode, holdout_class, ratio)


if __name__ == "__main__":
    main()
