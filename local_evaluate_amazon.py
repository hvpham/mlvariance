'''Calculate accuracy HAN on AMAZON with PyTorch.'''
import torch

import os
import argparse

import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class DocDataset(Dataset):
    train_file = 'train_batch'
    val_file = 'val_batch'
    test_file = 'test_batch'

    def __init__(self, holdoutroot, mode='train') -> None:

        self.mode = mode  # training set or test set

        assert mode in {'train', 'test', 'val'}

        if self.mode == 'train':
            file_name = self.train_file
        elif self.mode == 'val':
            file_name = self.val_file
        else:
            file_name = self.test_file

        # load data
        file_path = os.path.join(holdoutroot, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['labels']
            self.holdout = entry['holdout']

    def __getitem__(self, i):
        return self.data['docs'][i], \
            [self.data['sentences_per_document'][i]], \
            self.data['words_per_sentence'][i], \
            [self.targets[i]]

    def __len__(self) -> int:
        return len(self.targets)


def evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, a):

    holdoutroot = os.path.join(data_folder, 'amazon', '%s_%s_%d' % (mode, holdout_class.replace(' ', '_'), a))

    trainset = DocDataset(holdoutroot, 'train')
    train_holdout = trainset.holdout
    valset = DocDataset(holdoutroot, 'val')
    val_holdout = valset.holdout
    testset = DocDataset(holdoutroot, 'test')
    test_holdout = testset.holdout

    saving_dir = os.path.join(result_folder, 'amazon', 'amazon-%s_%s_%d' % (mode, holdout_class.replace(' ', '_'), a))
    csv_out_file = os.path.join(saving_dir, 'result.csv')

    f = open(csv_out_file, "w")
    f.write("Run")
    f.write(",train_accu,train_holdout,train_normal")
    f.write(",val_accu,val_holdout,val_normal")
    f.write(",test_accu,test_holdout,test_normal")
    f.write("\n")

    for run in range(NO_RUNS):
        outputs_saving_file = os.path.join(saving_dir, 'outputs_%d' % run)

        # with open(outputs_saving_file, 'rb') as rf:
        #    test_outputs = pickle.load(rf, encoding='latin1')
        test_outputs = torch.load(outputs_saving_file)

        def process(outputs, targets):
            pred_list = np.argmax(outputs, axis=1)
            correct = np.equal(pred_list, targets)
            accu = float(np.sum(np.multiply(correct, 1)))/len(targets)
            return correct, accu

        def create_report(outputs, targets, holdout):
            report = process(outputs, targets)

            test_index = [i for i, x in enumerate(holdout) if x]

            test_neg_index = np.array(list(range(len(targets))))
            test_neg_index = np.delete(test_neg_index, test_index)

            holdout_outputs = [outputs[i] for i in test_index]
            if len(holdout_outputs) != 0:
                holdout_outputs = np.vstack(holdout_outputs)
                holdout_report = process(holdout_outputs, [targets[i] for i in test_index])
            else:
                holdout_report = None

            normal_outputs = [outputs[i] for i in test_neg_index]
            normal_outputs = np.vstack(normal_outputs)
            normal_report = process(normal_outputs, [targets[i] for i in test_neg_index])

            return report, holdout_report, normal_report

        train_report = create_report(test_outputs['train_outputs'], test_outputs['train_targets'], train_holdout)
        val_report = create_report(test_outputs['val_outputs'], test_outputs['val_targets'], val_holdout)
        test_report = create_report(test_outputs['test_outputs'], test_outputs['test_targets'], test_holdout)

        f.write("%d" % run)
        if a != 10:
            f.write(",%f,%f,%f" % (train_report[0][1], train_report[1][1], train_report[2][1]))
            f.write(",%f,%f,%f" % (val_report[0][1], val_report[1][1], val_report[2][1]))
        else:
            f.write(",%f,%f,%f" % (train_report[0][1], -1.0, train_report[2][1]))
            f.write(",%f,%f,%f" % (val_report[0][1], -1.0, val_report[2][1]))
        f.write(",%f,%f,%f" % (test_report[0][1], test_report[1][1], test_report[2][1]))
        f.write("\n")

    f.close()


parser = argparse.ArgumentParser(description='Run experiment with holdout AMAZON and HAN')
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
        try:
            print("Evaluate mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
            evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, a)
        except Exception as e:
            print("Error while evaluating mode:%s holdout:%s ratio:%d" % (mode, holdout_class, a))
            print(e)
