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

def softmax(x):
    exp_x = np.exp(x) 
    return exp_x / exp_x.sum(axis = 1, keepdims=True)
    
def evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, a):

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    trainset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='train')
    train_holdout = trainset.holdout
    train_ids = trainset.ids
    
    valset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='val')
    val_holdout = valset.holdout
    val_ids = valset.ids

    testset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='test')
    test_holdout = testset.holdout
    test_ids = testset.ids
    
    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode,holdout_class,a))

    train_reports = []
    val_reports = []
    test_reports = []

    for run in range(NO_RUNS):
        outputs_saving_file = os.path.join(saving_dir,'outputs_%d' % run)

        #with open(outputs_saving_file, 'rb') as rf:
        #    test_outputs = pickle.load(rf, encoding='latin1')
        test_outputs = torch.load(outputs_saving_file)
    
        def process(outputs, targets, holdout):
            outputs = softmax(outputs)

            pred_list = np.argmax(outputs, axis=1)
            correct = np.equal(pred_list, targets)

            ground_confs = []
            ground_conf_gaps = []
            ground_ranks = []
            for i in range(len(correct)):
                g_conf = outputs[i,targets[i]]
                ground_confs.append(g_conf)
                
                max_conf = np.max(outputs[i,:])
                g_gap = max_conf-g_conf
                ground_conf_gaps.append(g_gap)

                pred_rank = np.flipud(np.argsort(outputs[i,:]))
                g_rank = np.where(pred_rank == targets[i].numpy())
                g_rank = g_rank[0][0]
                ground_ranks.append(g_rank)
            
            return (correct, np.array(ground_confs), np.array(ground_conf_gaps), np.array(ground_ranks))

        
        val_reports.append(process(test_outputs['val_outputs'], test_outputs['val_targets'], val_holdout))
        train_reports.append(process(test_outputs['train_outputs'], test_outputs['train_targets'], train_holdout))
        test_reports.append(process(test_outputs['test_outputs'], test_outputs['test_targets'], test_holdout))

    def merge_reports(targets, holdout, ids, reports):
        new_reports = [[],[],[],[],[]]
        
        for report in reports:
            for i in range(4):
                new_reports[i].append(report[i])

        for i in range(4):
            new_reports[i] = np.vstack(new_reports[i])
        new_reports[4] = np.array(new_reports[4])

        #count correct (first one)
        new_reports[0] = new_reports[0].astype(int)
        new_reports[0] = np.sum(new_reports[0], axis=0)

        #average others
        for i in range(1,4):
            new_reports[i] = np.average(new_reports[i], axis=0)

        inds = np.array(ids)
        new_targets = np.array([t.numpy() for t in targets])

        new_reports.insert(0, inds)
        new_reports.insert(1, new_targets)
        new_reports.insert(2, holdout)

        #new_reports = np.transpose(np.vstack(new_reports))

        return new_reports

    train_reports = merge_reports(test_outputs['train_targets'], train_holdout, train_ids, train_reports)
    val_reports = merge_reports(test_outputs['val_targets'], val_holdout, val_ids, val_reports)
    test_reports = merge_reports(test_outputs['test_targets'], test_holdout, val_ids, test_reports)

    def filter_reports(reports):
        new_reports = [[],[],[],[],[],[],[]]

        for i in range(7):
            new_reports[i] = [reports[i][r] for r in range(len(reports[0])) if reports[2][r]]

        order = np.argsort(new_reports[3])

        for i in range(7):
            new_reports[i] = [new_reports[i][r] for r in order]
        
        return new_reports

    train_reports = filter_reports(train_reports)
    val_reports = filter_reports(val_reports)
    test_reports = filter_reports(test_reports)
        

    image_relative_root = os.path.join("../../../data", 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    def output_analysis(setname, reports):
        analysis_out_file = os.path.join(saving_dir,'analysis_per_image_%s.html' % setname)

        f = open(analysis_out_file,"w")

        f.write('<!DOCTYPE html><html><body><h2>Analysis table for %s set</h2>\n' % setname)
        f.write('<table style="width:100%"><tr>')
        f.write("<th>Image</th>")
        f.write("<th>Img id</th>")
        f.write("<th>Target</th>")
        f.write("<th>Is holdout</th>")
        f.write("<th>Num correct models</th>")
        f.write("<th>Average confidence of ground-truth</th>")
        f.write("<th>Average confidence gap of ground-truth label with top</th>")
        f.write("<th>Average ground-truth rank</th>")
        
        f.write("</tr>\n")

        for i in range(len(reports[0])):
            f.write("<tr>")
            f.write('<td><img src="%s/%s/%d_%s_%d.jpg" alt="%d_%s_%d.jpg"></td>' % (holdoutroot, setname, reports[1][i], reports[2][i], reports[0][i], reports[1][i], reports[2][i], reports[0][i]))
            f.write("<td>%d</td><td>%d</td><td>%s</td>" % (reports[0][i], reports[1][i], reports[2][i]))
            f.write("<td>%d</td><td>%f</td><td>%f</td><td>%f</td>" % (reports[3][i], reports[4][i], reports[5][i], reports[6][i]))
            f.write("</tr>\n")
    
        f.write("</table></body></html>")
        f.close()

    output_analysis('train', train_reports)
    output_analysis('val', val_reports)
    output_analysis('test', test_reports)




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

holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
holdout_targets = [6,6,6,6,6,4,3,5,1]

for i in range(len(holdout_classes)):
    holdout_class = holdout_classes[i]
    holdout_target = holdout_targets[i]
    for a in [0,3,6,9]:
        print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
        evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, a)

#evaluate_model(100, data_folder, result_folder, mode, 'ray', 1, 9)