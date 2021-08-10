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

import pandas as pd


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


class CIFAR100_HOLDOUT_WITH_ADDITIONAL_VAL(VisionDataset):
    def __init__(self, holdoutroot, valroot, ranking_path, ranking_mode, val_ratio, transform=None, target_transform=None):

        super(CIFAR100_HOLDOUT_WITH_ADDITIONAL_VAL, self).__init__(holdoutroot, transform=transform,
                                                                   target_transform=target_transform)

        self.valroot = valroot

        def load_data(root, file_name):
            data = []
            targets = []

            file_path = os.path.join(root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data = entry['data']
                targets = entry['labels']
                holdout = entry['holdout']

            return data, targets, holdout

        train_data, train_target, train_holdout = load_data(holdoutroot, 'train_batch')
        val_data, val_target, val_holdout = load_data(valroot, 'val_batch')

        if ranking_mode != 'random':
            ranking = pd.read_csv(ranking_path)
            if ranking_mode == 'std_conf':
                std_conf = ranking['Stddev max conf'].tolist()
                sorted_idxs = np.argsort(std_conf)
                sorted_idxs = np.flip(sorted_idxs)
            elif ranking_mode == 'avg_conf':
                avg_conf = ranking['Avg max conf'].tolist()
                sorted_idxs = np.argsort(avg_conf)
        elif ranking_mode == 'random':
            sorted_idxs = np.random.permutation(len(val_target))
        else:
            pass

        exclude_idxs = sorted_idxs[int(val_ratio*len(sorted_idxs)/10) - 1:]

        filtered_val_data = np.delete(val_data, exclude_idxs, 0)
        filtered_val_target = [t for i, t in enumerate(val_target) if i not in exclude_idxs]
        filtered_val_holdout = [t for i, t in enumerate(val_holdout) if i not in exclude_idxs]

        self.data = np.concatenate((train_data, filtered_val_data), axis=0)
        self.targets = train_target + filtered_val_target
        self.holdout = train_holdout + filtered_val_holdout

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


def train_model(run, data_folder, result_folder, mode, holdout_class, a, val_ratio, retrain_mode):
    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))
    os.makedirs(saving_dir, exist_ok=True)
    saving_file = os.path.join(saving_dir, 'model_%d_%s_%d.pth' % (run, retrain_mode, val_ratio))

    if check_done(saving_file):
        print("Already trained for run %d with holdout class %s and a %d. Skip to the next run." % (run, holdout_class, a))
        return

    lr = 0.1

    #EPOCH = 150
    #EPOCH = 2
    EPOCH = 200

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 150   # start from epoch 150 or last checkpoint epoch

    # Data
    print('Preparing data..')
    tic = time.perf_counter()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))
    valroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, 0))
    ranking_path = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a), 'analysis_per_image_full_val.csv')

    trainset = CIFAR100_HOLDOUT_WITH_ADDITIONAL_VAL(
        holdoutroot=holdoutroot, valroot=valroot, ranking_path=ranking_path, ranking_mode=retrain_mode, val_ratio=val_ratio,  transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    # Model
    print('Building model..')
    tic = time.perf_counter()
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint_file = 'ckpt_%s_%s_%d_%d_%s_%d.pth' % (mode, holdout_class, a, run, retrain_mode, val_ratio)
    if os.path.isfile(checkpoint_file):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        # load pretrained model
        model_file = os.path.join(saving_dir, 'model_%d.pth' % run)
        state_dict = torch.load(model_file)
        net.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{"params": net.parameters(), "initial_lr": lr}], lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, last_epoch=start_epoch-1)

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    # Training

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if epoch % 10 == 0:
            print("Loss: %.3f | Acc: %.3f%% (%d/%d)\n" %
                  (train_loss/len(trainloader), 100.*correct/total, correct, total))

    print('Training...')
    tic = time.perf_counter()

    for epoch in range(start_epoch, EPOCH):
        print('Epoch: %d' % epoch)

        train(epoch)

        if epoch % 10 == 0:
            print('Saving check point ..')
            sav_tic = time.perf_counter()

            state = {
                'net': net.state_dict(),
                'epoch': epoch
            }
            torch.save(state, checkpoint_file)

            sav_toc = time.perf_counter()
            print("Done in %f seconds" % (sav_toc - sav_tic))

        scheduler.step()

    toc = time.perf_counter()
    print("Training done in %f seconds" % (toc - tic))

    print('Saving final model')
    tic = time.perf_counter()
    torch.save(net.state_dict(), saving_file)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print('Deleting check point')
    tic = time.perf_counter()
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))


def test_model(run, data_folder, result_folder, mode, holdout_class, a, val_ratio, retrain_mode):
    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))
    model_saving_file = os.path.join(saving_dir, 'model_%d_%s_%d.pth' % (run, retrain_mode, val_ratio))
    outputs_saving_file = os.path.join(saving_dir, 'outputs_%d_%s_%d' % (run, retrain_mode, val_ratio))

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

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    testset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

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
                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs_list.append(outputs.cpu().numpy())
                targets_list.extend(targets.cpu())
        outputs_list = np.vstack(outputs_list)
        return outputs_list, targets_list

    results = {}

    print("Evaluating test data")
    tic = time.perf_counter()
    results['test_outputs'], results['test_targets'] = evaluate(testloader)
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
    parser.add_argument('val_ratio', help='the ratio of additional validation sample to train on')
    parser.add_argument('retrain_mode', choices=['random', 'avg_conf', 'std_conf', 'conf_worst', 'conf_median'], help='the mode')

    args = parser.parse_args()

    data_folder = args.data_folder
    result_folder = args.result_folder
    mode = args.mode
    retrain_mode = args.retrain_mode

    run_id = int(args.run_id)
    holdout_class = args.holdout_class
    ratio = int(args.ratio)
    val_ratio = int(args.val_ratio)

    train_model(run_id, data_folder, result_folder, mode, holdout_class, ratio, val_ratio, retrain_mode)
    test_model(run_id, data_folder, result_folder, mode, holdout_class, ratio, val_ratio, retrain_mode)


if __name__ == "__main__":
    main()
