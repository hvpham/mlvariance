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

def analyze_individual_rank(targets, holdout, holdout_target, metrics):
    target_metrics = [metrics[i] for i in range(len(metrics)) if targets[i] == holdout_target]
    target_holdout = [holdout[i] for i in range(len(holdout)) if targets[i] == holdout_target]
    ranks = np.argsort(target_metrics)

    

    holdout_ranks = [idx for idx, rank in enumerate(ranks) if target_holdout[rank]]

    first_rank = holdout_ranks[0]
    median_rank = np.median(holdout_ranks)
    avg_first_10_rank = np.mean(holdout_ranks[:10])
    avg_rank = np.mean(holdout_ranks)

    return first_rank, median_rank, avg_first_10_rank, avg_rank
    
def evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, a, overall_file_writer):

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
            
            ranks = analyze_individual_rank(targets, holdout, holdout_target, ground_confs)

            # target_confs = [ground_confs[i] for i in range(len(ground_confs)) if targets[i] == holdout_target]
            # target_holdout = [holdout[i] for i in range(len(holdout)) if targets[i] == holdout_target]
            # confs_ranks = np.argsort(target_confs)
            # for idx, confs_rank in enumerate(confs_ranks):
            #     if target_holdout[confs_rank]:
            #         first_conf_rank = idx
            #         break

            return (correct, np.array(ground_confs), np.array(ground_conf_gaps), np.array(ground_ranks), ranks)

        
        val_reports.append(process(test_outputs['val_outputs'], test_outputs['val_targets'], val_holdout))
        train_reports.append(process(test_outputs['train_outputs'], test_outputs['train_targets'], train_holdout))
        test_reports.append(process(test_outputs['test_outputs'], test_outputs['test_targets'], test_holdout))

    def merge_reports(targets, holdout, reports):
        new_reports = [[],[],[],[],[]]
        
        for report in reports:
            for i in range(5):
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

        inds = np.array(list(range(len(targets))))
        new_targets = np.array([t.numpy() for t in targets])

        new_reports.insert(0, inds)
        new_reports.insert(1, new_targets)
        new_reports.insert(2, holdout)

        #new_reports = np.transpose(np.vstack(new_reports))

        return new_reports

    train_reports = merge_reports(test_outputs['train_targets'], train_holdout, train_reports)
    val_reports = merge_reports(test_outputs['val_targets'], val_holdout, val_reports)
    test_reports = merge_reports(test_outputs['test_targets'], test_holdout, test_reports)

    def output_analysis(setname, reports):
        analysis_out_file = os.path.join(saving_dir,'analysis_per_image_%s.csv' % setname)

        f = open(analysis_out_file,"w")
        f.write("Img id,")
        f.write("Target,")
        f.write("Is holdout,")
        f.write("Num correct models,")
        f.write("Average confidence of ground-truth,")
        f.write("Average confidence gap of ground-truth label with top,")
        f.write("Average ground-truth rank")
        f.write("\n")

        for i in range(len(reports[0])):
            f.write("%d,%d,%s" % (reports[0][i], reports[1][i], reports[2][i]))
            f.write(",%d,%f,%f,%f" % (reports[3][i], reports[4][i], reports[5][i], reports[5][i]))
            f.write("\n")
    
        f.close()

    output_analysis('train', train_reports)
    output_analysis('val', val_reports)
    output_analysis('test', test_reports)

    
    def analyze_rank(reports, overall_file_writer):
        holdout = reports[2]
        targets = reports[1]
        no_correct = reports[3]
        avg_ground_conf = reports[4]

        no_correct_ranks = analyze_individual_rank(targets, holdout, holdout_target, no_correct)
        avg_ground_conf_ranks = analyze_individual_rank(targets, holdout, holdout_target, avg_ground_conf)

        NUM_METRICS = 4
        ranks = [[],[],[],[]]
        for r in range(len(reports[7])):
            for i in range(NUM_METRICS):
                ranks[i].append(reports[7][r][i])

        min_ranks = [np.min(ranks[i]) for i in range(NUM_METRICS)]
        max_ranks = [np.max(ranks[i]) for i in range(NUM_METRICS)]

        for i in range(NUM_METRICS):
            if i < 2:
                pattern = ",%d,%d,%d,%d"
            else:
                pattern = ",%f,%f,%f,%f"
            
            overall_file_writer.write(pattern % (no_correct_ranks[i], avg_ground_conf_ranks[i], min_ranks[i], max_ranks[i]))
        
    
    overall_file_writer.write("%s_%s_%d" % (mode,holdout_class,a))
    analyze_rank(train_reports, overall_file_writer)
    analyze_rank(val_reports, overall_file_writer)
    analyze_rank(test_reports, overall_file_writer)
    overall_file_writer.write("\n")



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

overall_file_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_rank.csv')
overall_file_writer = open(overall_file_path,'w')

overall_file_writer.write(",train" + "," * 15)
overall_file_writer.write(",val" + "," * 15)
overall_file_writer.write(",test" + "," * 15)
overall_file_writer.write("\n")

for i in range(3):
    overall_file_writer.write(",first_rank" + "," * 3)
    overall_file_writer.write(",median_rank" + "," * 3)
    overall_file_writer.write(",avg_top_10_rank" + "," * 3)
    overall_file_writer.write(",avg_rank" + "," * 3)
overall_file_writer.write("\n")

for i in range(12):
    overall_file_writer.write(",no_model, avg_ground_conf, min, max")
overall_file_writer.write("\n")


for i in range(len(holdout_classes)):
    holdout_class = holdout_classes[i]
    holdout_target = holdout_targets[i]
    for a in [0,3,6,9]:
        print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
        evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, a, overall_file_writer)

#evaluate_model(100, data_folder, result_folder, mode, 'baby', 6, 0, overall_file_writer)

overall_file_writer.close()