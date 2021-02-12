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

from sklearn.tree import DecisionTreeClassifier

import os

OUT_FOLDER = "data/COMPAS/holdout/recidivism_%s.csv"

df = pd.read_csv("data/COMPAS/recidivism.csv")

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.165, random_state=69)


clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=300)
clf.fit(X, y)

l_train = clf.apply(X_train)
l_test = clf.apply(X_test)
l_indexes = np.unique(l_train)
clusters = []
for l_index in l_indexes:
    c_train_indexes = np.where(l_train == l_index)
    c_test_indexes = np.where(l_test == l_index)
    cluster = (l_index, c_train_indexes[0], c_test_indexes[0]) 
    clusters.append(cluster)
clusters.sort(key=lambda x:len(x[1]))

def store_data(outfile, X, y):
    df = pd.concat([X,y], axis = 1)
    df.to_csv(outfile)


#create test data
test_outfile = OUT_FOLDER % ("test")
store_data(test_outfile, X_test, y_test)

#create train data by removing each cluster
for index, cluster in enumerate(clusters):
    train_outfile = OUT_FOLDER % ("train_%d" % index)
    X_train_cluster =  X_train.copy()
    y_train_cluster = y_train.copy()
    df_index = X_train_cluster.index
    drop_df_index = [df_index[i] for i in cluster[1]]
    X_train_cluster = X_train_cluster.drop(drop_df_index, axis=0)
    y_train_cluster = y_train_cluster.drop(drop_df_index)
    store_data(train_outfile, X_train_cluster, y_train_cluster)

    test_outfile = OUT_FOLDER % ("test_index_%d" % index)
    f = open(test_outfile,"w")
    for i in cluster[2]:
        f.write("%d\n"% i)
    f.close()


