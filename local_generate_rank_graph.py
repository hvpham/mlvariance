import numpy as np
import pandas as pd
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate ranking graph')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

result_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_rank.csv')

df = pd.read_csv(result_path, skiprows=2, usecols=range(1,55))

def generate_boxplot(data, ylabel, xticklabels, filename):
    fig =  plt.figure(figsize = (12, 6))

    plt.boxplot(data)
    plt.xlabel('Ranking metrics')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(xticklabels) + 1), labels=xticklabels)
    

    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path)
    plt.close(fig)

def generate_lineplot(data, ylabel, xticklabels, legends, filename):
    fig =  plt.figure(figsize = (12, 6))

    for i in range(len(data)):
        plt.plot(data[i], label=legends[i])
    
    plt.xlabel('Holdout ratio')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), labels=xticklabels)
    plt.legend()
    
    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path)
    plt.close(fig)

def plot_boxplot_per_metrics(df, metric_index, filename_prefix):
    metric_names = ["First holdout rank", "Median holdout rank", "Average Top-10 holdout rank", "Average holdout rank"]
    if metric_index == 0:
        index_string = ""
    else:
        index_string = "."+str(metric_index)

    data = [df['no_model'+index_string].tolist(), 
        df['avg_ground_conf'+index_string].tolist(), df['std_ground_conf'+index_string].tolist(), df['median_g_conf'+index_string].tolist(), 
        df['avg_max_conf'+index_string].tolist(), df['std_max_conf'+index_string].tolist(), df['median_m_conf'+index_string].tolist()]
    
    generate_boxplot(data, metric_names[metric_index], 
        ['#C-Models', 'Avg-GrdConf', 'SDev-GrdConf', 'IModel-GrdConf',
            'Avg-PreConf', 'SDev-PreConf', 'IModel-PreConf'],
        filename_prefix + metric_names[metric_index])

def plot_line_per_metrics(df, metric_index, filename_prefix):
    metric_names = ["First holdout rank", "Median holdout rank", "Average Top-10 holdout rank", "Average holdout rank"]
    if metric_index == 0:
        index_string = ""
    else:
        index_string = "."+str(metric_index)

    col_names = ['no_model', 'avg_ground_conf', 'std_ground_conf', 'median_g_conf', 'avg_max_conf', 'std_max_conf', 'median_m_conf']
    
    data = []
    for i in range(len(col_names)):
        data.append([])
    
    for ratio in range (11):
        ratio_df = df[(df['ratio'] == ratio)]

        for i in range(len(col_names)):
            data[i].append(np.mean(ratio_df[col_names[i]+index_string].tolist()))
    
    generate_lineplot(data, metric_names[metric_index],
        ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
        ['#C-Models', 'Avg-GrdConf', 'SDev-GrdConf', 'IModel-GrdConf',
            'Avg-PreConf', 'SDev-PreConf', 'IModel-PreConf'],
        filename_prefix + metric_names[metric_index])

for i in range(4):
    plot_boxplot_per_metrics(df, i, "Box_all_data_points_")

#for ratio in range (11):
for ratio in [0,10]:
    ratio_df = df[(df['ratio'] == ratio)]

    for i in range(4):
        plot_boxplot_per_metrics(ratio_df, i, "Box_ratio_%d_" % ratio)

for i in range(4):
    plot_line_per_metrics(df, i, "Line_all_data_points_")

groups = [['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman'], ['caterpillar', 'mushroom', 'porcupine'], ['ray']]

for g in range(len(groups)):
    group_df = df[df['holdout_class'].isin(groups[g])]

    for i in range(4):
        plot_boxplot_per_metrics(group_df, i, "Box_group_%d_" % g)
        plot_line_per_metrics(group_df, i, "Line_group_%d_" % g)
