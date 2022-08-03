import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import argparse
import os

parser = argparse.ArgumentParser(description='Analyze holdout AMAZON and HAN')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

OUT_FILE = result_folder+"/amazon/amazon-%s/result.csv"

f = open(os.path.join(result_folder, "amazon/result.csv"), "w")
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


Education_Reference = ["Languages", "Maps & Atlases", "Test Preparation", "Dictionaries", "Religion"]  # 0
Business_Office = ["Office Suites", "Document Management", "Training", "Word Processing", "Contact Management"]  # 1
Children_s = ["Early Learning", "Games", "Math", "Reading & Language", "Art & Creativity"]  # 2
Utilities = ["Backup", "PC Maintenance", "Drivers & Driver Recovery", "Internet Utilities", "Screen Savers"]  # 3
Design_Illustration = ["Animation & 3D", "Training", "CAD", "Illustration"]  # 4
Accounting_Finance = ["Business Accounting", "Personal Finance", "Check Printing", "Point of Sale", "Payroll"]  # 5
Video = ["Video Editing", "DVD Viewing & Burning", "Compositing & Effects", "Encoding"]  # 6
Music = ["Instrument Instruction", "Music Notation", "CD Burning & Labeling", "MP3 Editing & Effects"]  # 7
Programming_Web_Development = ["Training & Tutorials", "Programming Languages", "Database", "Development Utilities", "Web Design"]  # 8
Networking_Servers = ["Security", "Firewalls", "Servers", "Network Management", "Virtual Private Networks"]  # 9

holdout_class_list = []
holdout_class_list.extend(Education_Reference)
holdout_class_list.extend(Business_Office)
holdout_class_list.extend(Children_s)
holdout_class_list.extend(Utilities)
holdout_class_list.extend(Design_Illustration)
holdout_class_list.extend(Accounting_Finance)
holdout_class_list.extend(Video)
holdout_class_list.extend(Music)
holdout_class_list.extend(Programming_Web_Development)
holdout_class_list.extend(Networking_Servers)

holdout_class_list = Children_s


for holdout_class in holdout_class_list:
    for a in [0, 1, 2, 5, 10]:
        # for a in range(11):
        analyze_experiment("%s-%s-%d" % ('holdout', holdout_class.replace(' ', '_'), a), f)

f.close()
