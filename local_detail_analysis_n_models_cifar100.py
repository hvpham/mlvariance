'''Train CIFAR10 with PyTorch.'''
import os

import argparse

import cluster_single_detail_analysis_n_models_cifar100

parser = argparse.ArgumentParser(description='Run local analysis')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')
parser.add_argument('number_of_run', help='number of runs')
parser.add_argument('number_of_splits', help='the the number of split to incremently add runs')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder
mode = args.mode

NO_RUNS = int(args.number_of_run)
NO_SPLITS = int(args.number_of_splits)

#holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_targets = [6,6,6,6,6,4,3,5,1]
#holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman']
#holdout_targets = [6, 6, 6, 6, 6]
holdout_classes = ['boy-girl', 'boy-man', 'girl-woman', 'man-woman']
holdout_targets = [6, 6, 6, 6]
#holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_targets = [4, 3, 5, 1]

for i in range(len(holdout_classes)):
    holdout_class = holdout_classes[i]
    holdout_target = holdout_targets[i]
    for ratio in range(11):
        print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, ratio))
        cluster_single_detail_analysis_n_models_cifar100.evaluate_model(NO_RUNS, NO_SPLITS, data_folder, result_folder, mode, holdout_class, holdout_target, ratio)
