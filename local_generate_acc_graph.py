import numpy as np
import pandas as pd
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate acc graph')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

result_path = os.path.join(result_folder, 'cifar100', 'result.csv')

df = pd.read_csv(result_path, usecols=range(1,13))

def generate_lineplot(data, ylabel, xticklabels, legends, linecolor, linestyles, filename):
    fig =  plt.figure(figsize = (8, 6))

    for i in range(len(data)):
        plt.plot(data[i], label=legends[i], color=linecolor[i], linestyle=linestyles[i])
    
    plt.xlabel('Holdout ratio')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), labels=xticklabels)
    plt.legend()
    
    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path)
    plt.close(fig)

def plot_line_per_metrics(df, filename_prefix):
    holdout_classes = ['baby', 'mushroom', 'ray']

    data = []
    for i in range(len(holdout_classes)):
        group_df = df[df['holdout_class'] == holdout_classes[i]]
        data.append(group_df['test_holdout'].tolist())

    linecolors = ['red', 'steelblue', 'purple']
    loosely_dashed = (0, (5, 10))
    linestyles = ['solid', 'dotted', 'dashed']
    
    generate_lineplot(data, "Blind spot test accuracy",
        ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
        holdout_classes,
        linecolors,
        linestyles,
        filename_prefix)

plot_line_per_metrics(df,"Line_per_class_acc_baby_mushroom_ray")
