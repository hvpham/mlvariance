'''Train CIFAR10 with PyTorch.'''
import os

import argparse

import cluster_single_val_test_cifar100

parser = argparse.ArgumentParser(description='Run local analysis')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')
parser.add_argument('number_of_run', help='number of runs')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder
mode = args.mode

NO_RUNS = int(args.number_of_run)

#holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']

holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']

for i in range(len(holdout_class_list)):
    holdout_class = holdout_class_list[i]
    for ratio in range(11):
        for run_id in range(NO_RUNS):
            print("Compute output mode:%s holdout:%s ratio:%d run:%d" % (mode, holdout_class, ratio, run_id))
            cluster_single_val_test_cifar100.test_model(run_id, data_folder, result_folder, mode, holdout_class, ratio)
