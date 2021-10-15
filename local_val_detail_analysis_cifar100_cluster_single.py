'''Train CIFAR10 with PyTorch.'''
import os

import argparse

import cluster_single_detail_analysis_cifar100
import cluster_single_val_detail_analysis_cifar100

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

#holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_targets = [6,6,6,6,6,4,3,5,1]
#holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman']
#holdout_targets = [6, 6, 6, 6, 6]
#holdout_classes = ['boy-girl', 'boy-man', 'girl-woman', 'man-woman']
#holdout_targets = [6, 6, 6, 6]
#holdout_classes = ['caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_targets = [4, 3, 5, 1]

#holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']
# holdout_class_list = ['boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree',
#                      'apple', 'girl', 'orange', 'rose', 'cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale']


aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']  # 0
fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']  # 1
flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']  # 2
fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']  # 3
insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']  # 4
medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']  # 5
people_list = ['baby', 'boy', 'girl', 'man', 'woman']  # 6
reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']  # 7
small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']  # 8
trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']  # 9

holdout_class_list = []
holdout_class_list.extend(aquatic_mammals_list)  # 0
holdout_class_list.extend(fish_list)  # 1
holdout_class_list.extend(flower_list)  # 2
holdout_class_list.extend(fruit_and_vegetables_list)  # 3
holdout_class_list.extend(insects_list)  # 4
holdout_class_list.extend(medium_mammals_list)  # 5
holdout_class_list.extend(people_list)  # 6
holdout_class_list.extend(reptiles_list)  # 7
holdout_class_list.extend(small_mammals_list)  # 8
holdout_class_list.extend(trees_list)  # 9

#holdout_targets = [7, 1, 7, 4, 5, 8, 1, 8, 7, 0, 8, 0]
# holdout_targets = [6, 9, 9, 9, 6, 6, 9,
#                   3, 6, 3, 2, 4, 2, 6, 9, 2, 3, 0]

holdout_targets = []
for l in range(10):
    holdout_class_list.extend([l] * 5)

for i in range(len(holdout_class_list)):
    holdout_class = holdout_class_list[i]
    holdout_target = holdout_targets[i]
    for ratio in range(11):
        print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, ratio))
        cluster_single_detail_analysis_cifar100.evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, ratio)
        cluster_single_val_detail_analysis_cifar100.evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, ratio)
