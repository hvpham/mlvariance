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

import math


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


def evaluate_model(NO_RUNS, NO_SPLITS, data_folder, result_folder, mode, holdout_class, holdout_target, a):

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    testset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='test')
    test_holdout = testset.holdout
    test_ids = testset.ids

    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))

    train_reports = []
    val_reports = []
    test_reports = []

    for run in range(NO_RUNS):
        outputs_saving_file = os.path.join(saving_dir, 'outputs_%d' % run)

        # with open(outputs_saving_file, 'rb') as rf:
        #    test_outputs = pickle.load(rf, encoding='latin1')
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

            # target_confs = [ground_confs[i] for i in range(len(ground_confs)) if targets[i] == holdout_target]
            # target_holdout = [holdout[i] for i in range(len(holdout)) if targets[i] == holdout_target]
            # confs_ranks = np.argsort(target_confs)
            # for idx, confs_rank in enumerate(confs_ranks):
            #     if target_holdout[confs_rank]:
            #         first_conf_rank = idx
            #         break

            return (correct, np.array(ground_confs), np.array(ground_conf_gaps), np.array(ground_ranks), np.array(max_confs), np.array(pred_list),
                    ground_conf_ranks, max_conf_ranks, ground_conf_auc, max_conf_auc)

        test_reports.append(process(test_outputs['test_outputs'], test_outputs['test_targets'], test_holdout))

    no_models_list = []
    no_models_per_split = math.floor(NO_RUNS / NO_SPLITS)
    for split_idx in range(1, NO_SPLITS+1):
        no_models_list.append(split_idx * no_models_per_split)

    def merge_reports(targets, holdout, ids, reports):
        processed_reports = [[], [], [], [], [], [], [], [], [], []]

        for report in reports:
            for i in range(10):
                processed_reports[i].append(report[i])

        for i in range(6):
            processed_reports[i] = np.vstack(processed_reports[i])

        new_reports = {}

        for no_models in no_models_list:
            # count correct
            new_reports['correct_%d' % no_models] = np.sum(processed_reports[0][:no_models].astype(int), axis=0)

            # average and stddev others
            new_reports['avg_g_conf_%d' % no_models] = np.average(processed_reports[1][:no_models], axis=0)
            #new_reports['avg_g_conf_gaps'] = np.average(processed_reports[2], axis=0)
            #new_reports['avg_g_conf_ranks'] = np.average(processed_reports[3], axis=0)
            new_reports['avg_max_conf_%d' % no_models] = np.average(processed_reports[4][:no_models], axis=0)

            new_reports['std_g_conf_%d' % no_models] = np.std(processed_reports[1][:no_models], axis=0)
            #new_reports['std_g_conf_gaps'] = np.std(processed_reports[2], axis=0)
            #new_reports['std_g_conf_ranks'] = np.std(processed_reports[3], axis=0)
            new_reports['std_max_conf_%d' % no_models] = np.std(processed_reports[4][:no_models], axis=0)

            new_reports['std_pre_labels_%d' % no_models] = np.std(processed_reports[5][:no_models], axis=0)
            new_reports['num_pre_labels_%d' % no_models] = []
            for r in range(len(new_reports['correct_%d' % no_models])):
                unique_labels = np.unique(processed_reports[5][:no_models, r])
                new_reports['num_pre_labels_%d' % no_models].append(len(unique_labels))

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

    test_reports = merge_reports(test_outputs['test_targets'], test_holdout, test_ids, test_reports)

    '''
    def output_rank(reports, overall_file_writer):
        holdout = reports['holdout']
        targets = reports['target']
        no_correct = reports['correct']
        avg_ground_conf = reports['avg_g_conf']
        std_ground_conf = reports['std_g_conf']
        avg_max_conf = reports['avg_max_conf']
        std_max_conf = reports['std_max_conf']

        std_pre_labels = reports['std_pre_labels']
        num_pre_labels = reports['num_pre_labels']

        no_correct_ranks = analyze_individual_rank(targets, holdout, holdout_target, no_correct, False)
        avg_ground_conf_ranks = analyze_individual_rank(targets, holdout, holdout_target, avg_ground_conf, False)
        std_ground_conf_ranks = analyze_individual_rank(targets, holdout, holdout_target, std_ground_conf, True)

        avg_max_conf_ranks = analyze_individual_rank(None, holdout, None, avg_max_conf, False)
        std_max_conf_ranks = analyze_individual_rank(None, holdout, None, std_max_conf, True)

        std_pre_labels_ranks = analyze_individual_rank(None, holdout, None, std_pre_labels, True)
        num_pre_labels_ranks = analyze_individual_rank(None, holdout, None, num_pre_labels, True)

        NUM_METRICS = 4

        def process_ranks(report):
            ranks = [[], [], [], []]
            for r in range(len(report)):
                for i in range(NUM_METRICS):
                    ranks[i].append(report[r][i])

            min_ranks = [np.min(ranks[i]) for i in range(NUM_METRICS)]
            max_ranks = [np.max(ranks[i]) for i in range(NUM_METRICS)]
            median_ranks = [np.median(ranks[i]) for i in range(NUM_METRICS)]

            return min_ranks, max_ranks, median_ranks

        min_g_conf_ranks, max_g_conf_ranks, median_g_conf_ranks = process_ranks(reports['ground_conf_ranks'])
        min_max_conf_ranks, max_max_conf_ranks, median_max_conf_ranks = process_ranks(reports['max_conf_ranks'])

        for i in range(NUM_METRICS):
            pattern = ",%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f"

            overall_file_writer.write(pattern % (no_correct_ranks[i],
                                                 avg_ground_conf_ranks[i], std_ground_conf_ranks[i], avg_max_conf_ranks[i], std_max_conf_ranks[i], std_pre_labels_ranks[i], num_pre_labels_ranks[i],
                                                 min_g_conf_ranks[i], max_g_conf_ranks[i], median_g_conf_ranks[i], min_max_conf_ranks[i], max_max_conf_ranks[i], median_max_conf_ranks[i]))

    overall_file = os.path.join(saving_dir, 'overall_holdout_rank.csv')
    overall_file_writer = open(overall_file, "w")
    overall_file_writer.write("%s,%s,%d" % (mode, holdout_class, a))
    output_rank(test_reports, overall_file_writer)
    overall_file_writer.close()
    '''

    def output_auc(reports, overall_file_writer):
        holdout = reports['holdout']
        targets = reports['target']

        for no_models in no_models_list:
            no_correct = reports['correct_%d' % no_models]
            avg_ground_conf = reports['avg_g_conf_%d' % no_models]
            std_ground_conf = reports['std_g_conf_%d' % no_models]
            avg_max_conf = reports['avg_max_conf_%d' % no_models]
            std_max_conf = reports['std_max_conf_%d' % no_models]

            std_pre_labels = reports['std_pre_labels_%d' % no_models]
            num_pre_labels = reports['num_pre_labels_%d' % no_models]

            no_correct_auc = analyze_individual_AUC(targets, holdout, holdout_target, no_correct, False)
            avg_ground_conf_auc = analyze_individual_AUC(targets, holdout, holdout_target, avg_ground_conf, False)
            std_ground_conf_auc = analyze_individual_AUC(targets, holdout, holdout_target, std_ground_conf, True)
            avg_max_conf_auc = analyze_individual_AUC(None, holdout, None, avg_max_conf, False)
            std_max_conf_auc = analyze_individual_AUC(None, holdout, None, std_max_conf, True)

            std_pre_labels_auc = analyze_individual_AUC(None, holdout, None, std_pre_labels, True)
            num_pre_labels_auc = analyze_individual_AUC(None, holdout, None, num_pre_labels, True)

            overall_file_writer.write(",%s,%f,%f,%f,%f,%f,%f,%f" % ("%d_models" % no_models, no_correct_auc,
                                                                    avg_ground_conf_auc, std_ground_conf_auc, avg_max_conf_auc, std_max_conf_auc, std_pre_labels_auc, num_pre_labels_auc))

        def process_AUC(report):
            min_auc = np.min(report)
            max_auc = np.max(report)
            median_auc = np.median(report)

            return min_auc, max_auc, median_auc

        min_g_conf_auc, max_g_conf_auc, median_g_conf_auc = process_AUC(reports['ground_conf_auc'])
        min_max_conf_auc, max_max_conf_auc, median_max_conf_auc = process_AUC(reports['max_conf_auc'])

        overall_file_writer.write(",single_model,%f,%f,%f,%f,%f,%f" % (min_g_conf_auc, max_g_conf_auc, median_g_conf_auc, min_max_conf_auc, max_max_conf_auc, median_max_conf_auc))

    overall_file = os.path.join(saving_dir, 'overall_holdout_auc_no_models.csv')
    overall_file_writer = open(overall_file, "w")
    overall_file_writer.write("%s,%s,%d" % (mode, holdout_class, a))
    output_auc(test_reports, overall_file_writer)
    overall_file_writer.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze result with holdout CIFAR-10 and Resnet18')
    parser.add_argument('data_folder', help='data folder')
    parser.add_argument('result_folder', help='result folder')
    parser.add_argument('mode', choices=['holdout'], help='the mode')
    parser.add_argument('holdout_class',
                        choices=['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray'],
                        help='the holdout class')
    parser.add_argument('holdout_target', help='the target that the holdout class belong to')
    parser.add_argument('ratio', help='the ratio of holdout')
    parser.add_argument('number_of_runs', help='the the number of runs')
    parser.add_argument('number_of_splits', help='the the number of split to incremently add runs')

    args = parser.parse_args()

    data_folder = args.data_folder
    result_folder = args.result_folder
    mode = args.mode

    holdout_class = args.holdout_class
    holdout_target = int(args.holdout_target)
    ratio = int(args.ratio)

    NO_RUNS = int(args.number_of_runs)
    NO_SPLITS = int(args.number_of_splits)

    print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, ratio))
    evaluate_model(NO_RUNS, NO_SPLITS, data_folder, result_folder, mode, holdout_class, holdout_target, ratio)


if __name__ == "__main__":
    main()
