'''Train CIFAR10 with PyTorch.'''
import os

import argparse

import cluster_single_train_more_cifar100

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

#holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']

holdout_class_list = ['boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree',
                          'apple', 'girl', 'orange', 'rose', 'cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale']

for holdout_class in holdout_class_list:
    for retrain_mode in ['random', 'std_conf', 'avg_conf', 'conf_worst', 'conf_best', 'conf_median']:
        # for ratio in [0,3,6,9]:
        # for ratio in [1,2,4,5,7,8,10]:
        # for ratio in range(11):
        for ratio in [0, 5, 10]:
            # for ratio in [10]:
            for val_ratio in [1, 2, 5, 10]:
                for run_id in range(NO_RUNS):
                    print("Compute output mode:%s holdout:%s ratio:%d run:%d retrain_mode:%s" % (mode, holdout_class, ratio, run_id, retrain_mode))
                    cluster_single_train_more_cifar100.test_model(run_id, data_folder, result_folder, mode, holdout_class, ratio, val_ratio, retrain_mode)
