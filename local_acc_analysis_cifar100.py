import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import argparse
import os

parser = argparse.ArgumentParser(description='Analyze holdout CIFAR-10 and Resnet18')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

OUT_FILE = result_folder+"/cifar100/cifar100-%s/result.csv"

f = open(os.path.join(result_folder, "cifar100/result.csv"), "w")
f.write("mode,holdout_class,ratio")
for i in range(5):
    f.write(",metric")
    f.write(",train_accu,train_holdout,train_normal")
    f.write(",val_accu,val_holdout,val_normal")
    f.write(",test_accu,test_holdout,test_normal")
f.write("\n")


def analyze_experiment(experiment_string, f):
    csv_out_file = OUT_FILE % ("%s" % experiment_string.replace('-', '_'))

    df = pd.read_csv(csv_out_file)

    stats = []
    for column_name in df:
        if column_name == 'Run':
            continue
        data = df[column_name].tolist()
        avg = st.mean(data)
        ma = max(data)
        mi = min(data)
        diff = ma-mi
        std_dev = st.stdev(data)
        stats.append((avg, std_dev, ma, mi, diff))

    st_names = ['avg', 'std_dev', 'max', 'min', 'diff']
    f.write("%s" % experiment_string.replace('-', ','))
    for index, st_name in enumerate(st_names):
        f.write(",%s" % st_name)
        for stat in stats:
            f.write(",%f" % stat[index])
    f.write("\n")
    # f.write("\n\n")


#modes = ['original', 'holdout']
modes = ['holdout']

#holdout_class_list = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']
#holdout_class_list = ['porcupine', 'ray']
#holdout_class_list = ['mushroom']

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

for mode in modes:
    for holdout_class in holdout_class_list:
        # for a in [0,3,6,9]:
        for a in range(11):
            analyze_experiment("%s-%s-%d" % (mode, holdout_class, a), f)

f.close()
