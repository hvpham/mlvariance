'''Train CIFAR10 with PyTorch.'''
import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import torch.backends.cudnn as cudnn

import torchvision
#import torchvision.transforms as transforms

import os
import argparse

#from resnet import ResNet18
#from utils import progress_bar

from torchvision.datasets import VisionDataset

import pickle

from PIL import Image

import numpy as np

# torchvision.datasets.CIFAR10

from sklearn.metrics import roc_auc_score


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
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def analyze_individual_rank(targets, holdout, holdout_target, metrics, higher_is_first):
    if targets is None:
        target_metrics = metrics
        target_holdout = holdout
    else:
        target_metrics = [metrics[i] for i in range(len(metrics)) if targets[i] == holdout_target]
        target_holdout = [holdout[i] for i in range(len(holdout)) if targets[i] == holdout_target]

    ranks = np.argsort(target_metrics)
    if higher_is_first:
        ranks = np.flipud(ranks)

    holdout_ranks = [idx for idx, rank in enumerate(ranks) if target_holdout[rank]]

    first_rank = holdout_ranks[0]/len(target_metrics)
    median_rank = np.median(holdout_ranks)/len(target_metrics)
    avg_first_10_rank = np.mean(holdout_ranks[:10])/len(target_metrics)
    avg_rank = np.mean(holdout_ranks)/len(target_metrics)

    return first_rank, median_rank, avg_first_10_rank, avg_rank


def analyze_individual_AUC(targets, holdout, holdout_target, metrics, higher_is_better):
    if targets is None:
        target_metrics = metrics
        target_holdout = holdout
    else:
        target_metrics = [metrics[i] for i in range(len(metrics)) if targets[i] == holdout_target]
        target_holdout = [holdout[i] for i in range(len(holdout)) if targets[i] == holdout_target]

    target_holdout = np.array(target_holdout)
    auc = roc_auc_score(target_holdout, target_metrics)

    if not higher_is_better:
        auc = 1 - auc

    return auc


def evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, a):

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, 0))

    valset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='val')
    val_holdout = valset.holdout
    val_ids = valset.ids

    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))

    val_reports = []

    for run in range(NO_RUNS):
        outputs_saving_file = os.path.join(saving_dir, 'outputs_val_%d' % run)

        test_outputs = torch.load(outputs_saving_file)

        def process(outputs, targets, holdout):
            outputs = softmax(outputs)

            pred_list = np.argmax(outputs, axis=1)
            correct = np.equal(pred_list, targets)

            ground_confs = []
            ground_conf_gaps = []
            ground_ranks = []
            max_confs = []
            for i in range(len(correct)):
                g_conf = outputs[i, targets[i]]
                ground_confs.append(g_conf)

                max_conf = np.max(outputs[i, :])
                max_confs.append(max_conf)
                g_gap = max_conf-g_conf
                ground_conf_gaps.append(g_gap)

                pred_rank = np.flipud(np.argsort(outputs[i, :]))
                g_rank = np.where(pred_rank == targets[i].numpy())
                g_rank = g_rank[0][0]
                ground_ranks.append(g_rank)

            ground_conf_ranks = analyze_individual_rank(targets, holdout, holdout_target, ground_confs, False)
            max_conf_ranks = analyze_individual_rank(None, holdout, None, max_confs, False)

            ground_conf_auc = analyze_individual_AUC(targets, holdout, holdout_target, ground_confs, False)
            max_conf_auc = analyze_individual_AUC(None, holdout, None, max_confs, False)

            return (correct, np.array(ground_confs), np.array(ground_conf_gaps), np.array(ground_ranks), np.array(max_confs), np.array(pred_list),
                    ground_conf_ranks, max_conf_ranks, ground_conf_auc, max_conf_auc)

        val_reports.append(process(test_outputs['val_outputs'], test_outputs['val_targets'], val_holdout))

    def merge_reports(targets, holdout, ids, reports, min_index, max_index, median_index):
        processed_reports = [[], [], [], [], [], [], [], [], [], []]

        for report in reports:
            for i in range(10):
                processed_reports[i].append(report[i])

        for i in range(6):
            processed_reports[i] = np.vstack(processed_reports[i])

        new_reports = {}

        # count correct
        new_reports['correct'] = np.sum(processed_reports[0].astype(int), axis=0)

        # average and stddev others
        new_reports['avg_g_conf'] = np.average(processed_reports[1], axis=0)
        new_reports['avg_g_conf_gaps'] = np.average(processed_reports[2], axis=0)
        new_reports['avg_g_conf_ranks'] = np.average(processed_reports[3], axis=0)
        new_reports['avg_max_conf'] = np.average(processed_reports[4], axis=0)

        new_reports['std_g_conf'] = np.std(processed_reports[1], axis=0)
        new_reports['std_g_conf_gaps'] = np.std(processed_reports[2], axis=0)
        new_reports['std_g_conf_ranks'] = np.std(processed_reports[3], axis=0)
        new_reports['std_max_conf'] = np.std(processed_reports[4], axis=0)

        new_reports['std_pre_labels'] = np.std(processed_reports[5], axis=0)
        new_reports['num_pre_labels'] = []
        for r in range(len(new_reports['correct'])):
            unique_labels = np.unique(processed_reports[5][:, r])
            new_reports['num_pre_labels'].append(len(unique_labels))

        new_reports['min_max_conf'] = processed_reports[4][min_index]
        new_reports['max_max_conf'] = processed_reports[4][max_index]
        new_reports['median_max_conf'] = processed_reports[4][median_index]

        # additional info
        inds = np.array(ids)
        new_targets = np.array([t.numpy() for t in targets])

        new_reports['id'] = inds
        new_reports['target'] = new_targets
        new_reports['holdout'] = holdout

        # individual ranks:
        new_reports['ground_conf_ranks'] = processed_reports[6]
        new_reports['max_conf_ranks'] = processed_reports[7]

        # individual auc:
        new_reports['ground_conf_auc'] = processed_reports[8]
        new_reports['max_conf_auc'] = processed_reports[9]

        return new_reports

    auc_path = os.path.join(saving_dir, 'overall_holdout_auc.csv')

    with open(auc_path) as f:
        line = f.readline()
        splited = line.split(',')
        min_index = int(splited[16])
        max_index = int(splited[17])
        median_index = int(splited[18])

    val_reports = merge_reports(test_outputs['val_targets'], val_holdout, val_ids, val_reports, min_index, max_index, median_index)

    def output_analysis(setname, reports):
        analysis_out_file = os.path.join(saving_dir, 'analysis_per_image_%s.csv' % setname)

        f = open(analysis_out_file, "w")
        f.write("Img id,")
        f.write("Target,")
        f.write("Is holdout,")
        f.write("# correct,")
        f.write("Avg ground conf,")
        f.write("Stddev ground conf,")
        f.write("Avg ground conf gaps,")
        f.write("Stddev ground conf gaps,")
        f.write("Avg ground conf ranks,")
        f.write("Stddev ground conf ranks,")
        f.write("Avg max conf,")
        f.write("Stddev max conf,")
        f.write("Std predicted conf,")
        f.write("# unique predicted label, ")
        f.write("Min max conf, ")
        f.write("Max max conf, ")
        f.write("Median max conf")
        f.write("\n")

        for i in range(len(reports['id'])):
            f.write("%d,%d,%s" % (reports['id'][i], reports['target'][i], reports['holdout'][i]))
            f.write(",%d" % (reports['correct'][i]))
            f.write(",%f,%f" % (reports['avg_g_conf'][i], reports['std_g_conf'][i]))
            f.write(",%f,%f" % (reports['avg_g_conf_gaps'][i], reports['std_g_conf_gaps'][i]))
            f.write(",%f,%f" % (reports['avg_g_conf_ranks'][i], reports['std_g_conf_ranks'][i]))
            f.write(",%f,%f" % (reports['avg_max_conf'][i], reports['std_max_conf'][i]))
            f.write(",%f,%d" % (reports['std_pre_labels'][i], reports['num_pre_labels'][i]))
            f.write(",%f,%f,%f" % (reports['min_max_conf'][i], reports['max_max_conf'][i], reports['median_max_conf'][i]))
            f.write("\n")

        f.close()

    output_analysis('full_val', val_reports)


def main():
    parser = argparse.ArgumentParser(description='Analyze result with holdout CIFAR-10 and Resnet18')
    parser.add_argument('data_folder', help='data folder')
    parser.add_argument('result_folder', help='result folder')
    parser.add_argument('mode', choices=['holdout'], help='the mode')
    parser.add_argument('holdout_class', help='the holdout class')
    parser.add_argument('holdout_target', help='the target that the holdout class belong to')
    parser.add_argument('ratio', help='the ratio of holdout')
    parser.add_argument('number_of_runs', help='the the number of runs')

    args = parser.parse_args()

    data_folder = args.data_folder
    result_folder = args.result_folder
    mode = args.mode

    holdout_class = args.holdout_class
    holdout_target = int(args.holdout_target)
    ratio = int(args.ratio)

    NO_RUNS = int(args.number_of_runs)

    print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, ratio))
    evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, ratio)


if __name__ == "__main__":
    main()
