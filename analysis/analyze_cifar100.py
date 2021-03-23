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

f = open(os.path.join(result_folder,"cifar100/result.csv"),"w")
f.write("experiment")
for i in range(5):
    f.write(",metric")
    f.write(",train_accu,train_holdout,train_normal")
    f.write(",val_accu,val_holdout,val_normal")
    f.write(",test_accu,test_holdout,test_normal")
f.write("\n")

def analyze_experiment(experiment_string , f):
    csv_out_file = OUT_FILE % ("%s" % experiment_string)

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
        stats.append((avg,std_dev,ma,mi,diff))
    
    st_names = ['avg','std_dev', 'max', 'min', 'diff']
    f.write("%s" % experiment_string)
    for index, st_name in enumerate(st_names):
        f.write(",%s" % st_name)
        for stat in stats:
            f.write(",%f" % stat[index])
    f.write("\n")
    #f.write("\n\n")

#modes = ['original', 'holdout']
modes = ['holdout']

holdout_classes = ['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray']

for mode in modes:
    for holdout_class in holdout_classes:
        for a in [0,3,6,9]:
            analyze_experiment("%s_%s_%d" % (mode, holdout_class, a), f)

f.close()
        
    

