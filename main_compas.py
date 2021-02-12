import simple_binary

import torch
# torch.manual_seed(0)

import numpy as np
# np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

import os

NUM_RUNS = 100

BATCH_SIZE = 64

OUT_FOLDER = "result/result_%s/"

df = pd.read_csv("data/recidivism.csv")

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.165, random_state=69)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=69)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

## train data
class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train),
                    torch.FloatTensor(y_train))


## test data
class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


train_test_data = testData(torch.FloatTensor(X_train))
val_data = testData(torch.FloatTensor(X_val))
test_data = testData(torch.FloatTensor(X_test))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

train_test_loader = DataLoader(dataset=train_test_data, batch_size=1)
val_loader = DataLoader(dataset=val_data, batch_size=1)
test_loader = DataLoader(dataset=test_data, batch_size=1)

def experiment(epoch, lr, hidden):
    out_folder = OUT_FOLDER % ("%d_%g_%d" % (epoch, lr, hidden))
    csv_out_file = out_folder+"result.csv"
    model_out_file = out_folder+"model_%d.pth"
    pred_out_file = out_folder+"pred_%d.csv"
    hist_out_file = out_folder+"correct_hist.pdf"
    detail_hist_out_file = out_folder+"detail_correct_hist.pdf"

    os.makedirs(out_folder, exist_ok=True)
    
    f = open(csv_out_file,"w")
    f.write("Run")
    f.write(",train_accu, train_0_precision, train_0_recall, train_1_precision, train_1_recall")
    f.write(",val_accu, val_0_precision, val_0_recall, val_1_precision, val_1_recall")
    f.write(",test_accu, test_0_precision, test_0_recall, test_1_precision, test_1_recall")
    f.write("\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_correct_count_dict = np.zeros(len(train_test_data), dtype=int)
    val_correct_count_dict = np.zeros(len(val_data), dtype=int)
    test_correct_count_dict = np.zeros(len(test_data), dtype=int)

    for run in range(NUM_RUNS):
        run_model_file = model_out_file % (run)
        run_pred_file = pred_out_file % (run)

        if os.path.isfile(run_pred_file):
            rdf = pd.read_csv(run_pred_file, header=None)
            train_conf = rdf.iloc[:len(train_test_data), 0].tolist()
            val_conf = rdf.iloc[len(train_test_data):(len(train_test_data)+len(val_data)), 0].tolist()
            test_conf = rdf.iloc[(len(train_test_data)+len(val_data)):, 0].tolist()
        else:
            if os.path.isfile(run_model_file):
                model = simple_binary.binaryClassification(13, hidden)
                model.load_state_dict(torch.load(run_model_file))
                model.to(device)
            else:
                model = simple_binary.train(train_loader, epoch=epoch, lr=lr, hidden=hidden)
                torch.save(model.state_dict(), run_model_file)

            model.eval()

            train_conf, val_conf, test_conf, = simple_binary.test(model, train_test_loader, val_loader, test_loader)
            rf = open(run_pred_file,"w")
            
            for conf in train_conf:
                rf.write("%f\n" % conf)
            for conf in val_conf:
                rf.write("%f\n" % conf)
            for conf in test_conf:
                rf.write("%f\n" % conf)
            
        def process(y_conf_list, y_t):
            y_pred_list = np.round(y_conf_list)

            correct = np.equal(y_pred_list, y_t)
            accu = accuracy_score(y_t, y_pred_list)
            report = classification_report(y_t, y_pred_list, output_dict=True)

            return correct, accu, report

        train_report = process(train_conf, y_train)
        val_report = process(val_conf, y_val)
        test_report = process(test_conf, y_test)

        train_correct_count_dict = train_correct_count_dict + 1 * train_report[0]
        val_correct_count_dict = val_correct_count_dict + 1 * val_report[0]
        test_correct_count_dict = test_correct_count_dict + 1 * test_report[0]

        def print_report(accu, report, f):
            f.write(",%f,%f,%f,%f,%f" % 
                (accu, report['0']['precision'],report['0']['recall'],report['1']['precision'],report['1']['recall']))
        
        f.write("%d" % run)
        print_report(train_report[1], train_report[2], f)
        print_report(val_report[1], val_report[2], f)
        print_report(test_report[1], test_report[2], f)

        f.write("\n")
    
    f.close()

    fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True)

    bins = list(range(NUM_RUNS+2))

    axs[0].hist(train_correct_count_dict, bins=bins)
    axs[1].hist(val_correct_count_dict, bins=bins)
    axs[2].hist(test_correct_count_dict, bins=bins)

    plt.savefig(hist_out_file)

    fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True)

    bins = list(range(1,NUM_RUNS))
    bins.append(NUM_RUNS-0.5)

    axs[0].hist(train_correct_count_dict, bins=bins)
    axs[1].hist(val_correct_count_dict, bins=bins)
    axs[2].hist(test_correct_count_dict, bins=bins)

    plt.savefig(detail_hist_out_file)


#experiment(1,0.001,64)
experiment(10,0.001,64)
experiment(10,0.0001,64)
experiment(1000,0.0001,64)
experiment(1000,0.0001,128)
