'''Train HAN on Amazon with PyTorch.'''
import argparse

import cluster_single_amazon


parser = argparse.ArgumentParser(description='Run experiment with holdout CIFAR-10 and Resnet18')
parser.add_argument('number_of_run', help='number of runs')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder
mode = args.mode

NO_RUNS = int(args.number_of_run)

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
        for id in range(NO_RUNS):
            try:
                print("Train mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
                cluster_single_amazon.train_model(id, data_folder, result_folder, mode, holdout_class, a)
            except:
                print("Error while training mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))

            try:
                print("Test mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
                cluster_single_amazon.test_model(id, data_folder, result_folder, mode, holdout_class, a)
            except:
                print("Error while testing mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
