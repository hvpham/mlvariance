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

    
def train_model(run, data_folder, result_folder, mode, holdout_class, a):
    saving_dir = os.path.join(result_folder, 'cifar10', 'cifar10-%s_%d_%d' % (mode,holdout_class,a))
    os.makedirs(saving_dir,exist_ok=True)
    saving_file = os.path.join(saving_dir,'model_%d.pth' % run)
    
    if os.path.isfile(saving_file):
        print("Already trained for run %d with holdout class %d and a %d. Skip to the next run." % (run,holdout_class,a))
        return

    lr = 0.1

    EPOCH = 150
    #EPOCH = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    holdoutroot = os.path.join(data_folder, 'cifar10', '%s_%d_%d' % (mode, holdout_class, a))

    trainset = CIFAR10_HOLDOUT(
        holdoutroot=holdoutroot, mode='train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
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

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    for epoch in range(start_epoch, start_epoch+EPOCH):
        train(epoch)
        scheduler.step()
    
    print('Saving..')
    torch.save(net.state_dict(), saving_file)



def test_model(run, data_folder, result_folder, mode, holdout_class, a):
    saving_dir = os.path.join(result_folder, 'cifar10', 'cifar10-%s_%d_%d' % (mode,holdout_class,a))
    model_saving_file = os.path.join(saving_dir,'model_%d.pth' % run)
    outputs_saving_file = os.path.join(saving_dir,'outputs_%d' % run)

    if os.path.isfile(outputs_saving_file):
        print("Already evaluate for run %d with holdout class %d and a %d. Skip to the next evaluation run." % (run,holdout_class,a))
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    holdoutroot = os.path.join(data_folder, 'cifar10', '%s_%d_%d' % (mode, holdout_class, a))

    trainset = CIFAR10_HOLDOUT(
        holdoutroot=holdoutroot, mode='train', transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=False, num_workers=2)

    valset = CIFAR10_HOLDOUT(
        holdoutroot=holdoutroot, mode='val', transform=transform_test)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=2)

    testset = CIFAR10_HOLDOUT(
        holdoutroot=holdoutroot, mode='test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    state_dict = torch.load(model_saving_file)
    net.load_state_dict(state_dict)

    net.eval()

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
    results['train_outputs'], results['train_targets'] = evaluate(trainloader)
    results['val_outputs'], results['val_targets'] = evaluate(valloader)
    results['test_outputs'], results['test_targets'] = evaluate(testloader)

    with open(outputs_saving_file, 'wb') as f:
        pickle.dump(results, f)

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
parser.add_argument('mode', choices=['holdout', 'holdout-dup', 'augmentation', 'augmentation-all'], help='the mode')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder
mode = args.mode

NO_RUNS = int(args.number_of_run)

for a in [3,6,9]:
    for run in range(NO_RUNS):
        train_model(run, data_folder, result_folder, mode, 0, a)
        test_model(run, data_folder, result_folder, mode, 0, a)
    evaluate_model(NO_RUNS, data_folder, result_folder, mode, 0, a)