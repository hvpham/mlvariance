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

OUT_FOLDER = "data/COMPAS/recidivism_%s.csv"

df = pd.read_csv("data/COMPAS/recidivism.csv")

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.165, random_state=69)

def generate_artificial_column(y, corrlelation_rate):
    rn = np.random.rand(len(y))
    crs = np.full(len(y),corrlelation_rate)
    new_feature = np.less(rn, crs)
    new_feature = np.multiply(new_feature, 1)
    new_feature = np.subtract(y, new_feature)
    new_feature = np.abs(new_feature)
    return new_feature

def store_data(outfile, X, y):
    df = pd.concat([X,y], axis = 1)
    df.to_csv(outfile)

#create test data with a = 0.5
test_outfile = OUT_FOLDER % ("test")
new_feature = generate_artificial_column(y_test, 0.5)
X_test['new_feature'] = new_feature
store_data(test_outfile, X_test, y_test)

#create train data with a = [0, 0.1, ... 0.9, 1.0]
alist = list(np.multiply(list(range(11)),0.05))
for a in alist:
    train_outfile = OUT_FOLDER % ("train_%.2f" % a)
    new_feature = generate_artificial_column(y_train, a)
    X_train['new_feature'] = new_feature
    store_data(train_outfile, X_train, y_train)
