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


class binaryClassification(nn.Module):
    def __init__(self, in_D, hi_D):
        super(binaryClassification, self).__init__()
        # Number of input features is 13.
        self.layer_1 = nn.Linear(in_D, hi_D)
        self.layer_2 = nn.Linear(hi_D, hi_D)
        self.layer_out = nn.Linear(hi_D, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hi_D)
        self.batchnorm2 = nn.BatchNorm1d(hi_D)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

def train(train_loader, input_size, epoch=1000, lr=0.0001, hidden=64):

    EPOCHS = epoch
    LEARNING_RATE = lr

    HIDDEN_SIZE = hidden

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = binaryClassification(input_size, HIDDEN_SIZE)
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc


    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
    
    return model
    
def test(model, train_test_loader, val_loader, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(data_loader):
        y_conf_list = []
        with torch.no_grad():
            for X_batch in data_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                #y_pred_tag = torch.round(y_test_pred)
                #y_pred_list.append(y_pred_tag.cpu().numpy())
                y_conf_list.append(y_test_pred.cpu().numpy())

        y_conf_list = [a.squeeze().tolist() for a in y_conf_list]

        
        return y_conf_list

    train_conf = evaluate(train_test_loader)
    val_conf = evaluate(val_loader)
    test_conf = evaluate(test_loader)

    return train_conf,val_conf,test_conf

    
