import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import argparse
import os

import traceback

import numpy as np

parser = argparse.ArgumentParser(
    description='Analyze holdout CIFAR-10 and Resnet18')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

OUT_FILE = result_folder+"/cifar100/cifar100-%s/result.csv"
RETRAIN_OUT_FILE = result_folder+"/cifar100/cifar100-%s/retrain_result_%s_%d.csv"

f = open(os.path.join(result_folder, "cifar100/retrain_acc_gain_result.csv"), "w")
f.write("mode,holdout_class,retrain_mode,ratio,val_ratio")
for i in range(6):
    f.write(",metric")
    f.write(",test_accu,test_holdout,test_normal")
f.write("\n")


def analyze_experiment(mode, holdout_class, retrain_mode, ratio, f):
    df0 = []
    for val_ratio in [0, 1, 2, 5, 10]:
        folder_string = "%s_%s_%d" % (mode, holdout_class, ratio)

        csv_out_file = RETRAIN_OUT_FILE % (
            "%s" % folder_string, retrain_mode, val_ratio)

        df = pd.read_csv(csv_out_file)

        if val_ratio == 0:
            df0 = df

        stats = []
        for column_name in df:
            if column_name == 'Run' or 'train' in column_name or 'val' in column_name:
                continue
            data0 = np.array(df0[column_name].tolist())
            data = np.array(df[column_name].tolist())
            data_diff = data - data0

            avg = st.mean(data_diff)
            med = st.median(data_diff)
            ma = max(data_diff)
            mi = min(data_diff)
            diff = ma-mi
            std_dev = st.stdev(data_diff)
            stats.append((avg, std_dev, ma, mi, diff, med))

        st_names = ['avg', 'std_dev', 'max', 'min', 'diff', 'med']
        f.write("%s,%s,%s,%d,%d" % (mode, holdout_class, retrain_mode, ratio, val_ratio))
        for index, st_name in enumerate(st_names):
            f.write(",%s" % st_name)
            for stat in stats:
                f.write(",%f" % stat[index])
        f.write("\n")


#modes = ['original', 'holdout']
modes = ['holdout']

#holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_class_list = ['porcupine', 'ray']
#holdout_class_list = ['mushroom']


#holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']

# holdout_class_list = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum',
#                      'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal',
#                      'boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree',
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

for mode in modes:
    for holdout_class in holdout_class_list:
        for retrain_mode in ['random', 'std_conf', 'avg_conf', 'single_conf', 'std_avg_conf']:
            for ratio in [0, 5, 10]:
                print("Analyze acc gain retrain mode:%s holdout:%s ratio:%d retrain_mode:%s" % (
                    mode, holdout_class, ratio, retrain_mode))
                try:
                    analyze_experiment(mode, holdout_class, retrain_mode, ratio, f)
                except:
                    traceback.print_exc()

f.close()
