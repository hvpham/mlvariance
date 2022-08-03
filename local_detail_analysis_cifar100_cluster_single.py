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

aquatic_mammals_list = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
fish_list = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
#fish_list = ['aquarium_fish', 'flatfish', 'shark', 'trout']
flower_list = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
fruit_and_vegetables_list = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
#fruit_and_vegetables_list = ['apple', 'orange', 'pear', 'sweet_pepper']
insects_list = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
#insects_list = ['bee', 'beetle', 'butterfly', 'cockroach']
medium_mammals_list = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
#medium_mammals_list = ['fox', 'possum', 'raccoon', 'skunk']
people_list = ['baby', 'boy', 'girl', 'man', 'woman']
#people_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman']
reptiles_list = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
small_mammals_list = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
trees_list = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']

holdout_class_list = []
holdout_class_list.extend(aquatic_mammals_list)
holdout_class_list.extend(fish_list)
holdout_class_list.extend(flower_list)
holdout_class_list.extend(fruit_and_vegetables_list)
holdout_class_list.extend(insects_list)
holdout_class_list.extend(medium_mammals_list)
holdout_class_list.extend(people_list)
holdout_class_list.extend(reptiles_list)
holdout_class_list.extend(small_mammals_list)
holdout_class_list.extend(trees_list)

holdout_targets = []
holdout_targets.extend([0]*len(aquatic_mammals_list))
holdout_targets.extend([1]*len(fish_list))
holdout_targets.extend([2]*len(flower_list))
holdout_targets.extend([3]*len(fruit_and_vegetables_list))
holdout_targets.extend([4]*len(insects_list))
holdout_targets.extend([5]*len(medium_mammals_list))
holdout_targets.extend([6]*len(people_list))
holdout_targets.extend([7]*len(reptiles_list))
holdout_targets.extend([8]*len(small_mammals_list))
holdout_targets.extend([9]*len(trees_list))

for i in range(len(holdout_class_list)):
    holdout_class = holdout_class_list[i]
    holdout_target = holdout_targets[i]
    for ratio in range(11):
        print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, ratio))
        cluster_single_detail_analysis_cifar100.evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, ratio)
