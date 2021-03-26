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

result_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_auc.csv')

df = pd.read_csv(result_path, skiprows=1, usecols=range(1,16))

def generate_boxplot(data, ylabel, xticklabels, filename):
    fig =  plt.figure(figsize = (12, 6))

    plt.boxplot(data)
    plt.xlabel('Scoring metrics')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(xticklabels) + 1), labels=xticklabels)
    

    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path)
    plt.close(fig)

def generate_lineplot(data, ylabel, xticklabels, legends, linecolor, filename):
    fig =  plt.figure(figsize = (12, 6))

    for i in range(len(data)):
        plt.plot(data[i], label=legends[i], color=linecolor[i])
    
    plt.xlabel('Holdout ratio')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), labels=xticklabels)
    plt.legend()
    
    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path)
    plt.close(fig)

def plot_boxplot_per_metrics(df, filename_prefix):
    #data = [df['no_model'].tolist(), 
    #    df['avg_ground_conf'].tolist(), df['std_ground_conf'].tolist(), df['median_g_conf'].tolist(), 
    #    df['avg_max_conf'].tolist(), df['std_max_conf'].tolist(), df['median_m_conf'].tolist()]

    data = [df['min_m_conf'].tolist(), df['median_m_conf'].tolist(),
        #df['std_pre_labels'].tolist(), df['num_pre_labels'].tolist(), 
        df['avg_max_conf'].tolist(), df['std_max_conf'].tolist()]

    #ylabels = ['#C-Models', 'Avg-GrdConf', 'SDev-GrdConf', 'IModel-GrdConf',
    #        'Avg-PreConf', 'SDev-PreConf', 'IModel-PreConf']

    ylabels = ['Worst-PreConf', 'Median-PreConf',
        #'SDev-PreLabel', 'NumUnique-PreLabel',
        'Avg-PreConf', 'SDev-PreConf']
    
    generate_boxplot(data, "AUC of the blind spot sample indicator", 
        ylabels,
        filename_prefix)

def plot_line_per_metrics(df, filename_prefix):
    #col_names = ['no_model', 'avg_ground_conf', 'std_ground_conf', 'median_g_conf', 'avg_max_conf', 'std_max_conf', 'median_m_conf']
    col_names = ['min_m_conf', 'median_m_conf',
        #'std_pre_labels', 'num_pre_labels', 
        'avg_max_conf', 'std_max_conf']
    
    data = []
    for i in range(len(col_names)):
        data.append([])

    #ylabels = ['#C-Models', 'Avg-GrdConf', 'SDev-GrdConf', 'IModel-GrdConf',
    #        'Avg-PreConf', 'SDev-PreConf', 'IModel-PreConf']

    #linelabels = ['#C-Models', 'Avg-GrdConf', 'SDev-GrdConf', 'IModel-GrdConf',
    #    'Avg-PreConf', 'SDev-PreConf', 'IModel-PreConf']

    linelabels = ['Worst-PreConf', 'Median-PreConf',
        #'SDev-PreLabel', 'NumUnique-PreLabel',
        'Avg-PreConf', 'SDev-PreConf']

    linecolors = ['red', 'orange', 'steelblue', 'green']
    
    for ratio in range (11):
        ratio_df = df[(df['ratio'] == ratio)]

        for i in range(len(col_names)):
            data[i].append(np.mean(ratio_df[col_names[i]].tolist()))
    
    generate_lineplot(data, "Average blind spot sample indication AUC",
        ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
        linelabels,
        linecolors,
        filename_prefix)

def plot_boxplot_per_metrics_per_group(dfs, groupnames, filename_prefix):
    data = []
    for df in dfs:
        data.append([df['avg_max_conf'].tolist(), df['std_max_conf'].tolist()])
    
    ylabels = ['Avg-PreConf', 'SDev-PreConf']
    
    fig =  plt.figure(figsize = (12, 6))

    ax1, ax2, ax3 = fig.subplots(1, 3, sharey=True)

    def create_box_plot(ax, data, name):
        ax.boxplot(data)
        ax.set_xticks([1, 2])
        #ax.set_xlim(0.5, 3.5)
        ax.set_xticklabels(ylabels)
        ax.title.set_text(name)

    create_box_plot(ax1,data[0], groupnames[0])
    create_box_plot(ax2,data[1], groupnames[1])
    create_box_plot(ax3,data[2], groupnames[2])

    plt.ylabel("Blind spot sample indication AUC")

    #fig.subplots_adjust(wspace=0.6)

    # Save the figure
    fig_path = os.path.join(result_folder, 'cifar100', filename_prefix+'.png')
    fig.savefig(fig_path)
    plt.close(fig)

plot_boxplot_per_metrics(df, "Box_all_data_points_AUC")

#for ratio in range (11):
for ratio in [0,10]:
    ratio_df = df[(df['ratio'] == ratio)]

    plot_boxplot_per_metrics(ratio_df, "Box_ratio_%d_AUC" % ratio)

plot_line_per_metrics(df,"Line_all_data_points_AUC")

groups = [['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman'], ['caterpillar', 'mushroom', 'porcupine'], ['ray']]

group_dfs = []
for g in range(len(groups)):
    group_df = df[df['holdout_class'].isin(groups[g])]
    group_dfs.append(group_df)

    plot_boxplot_per_metrics(group_df, "Box_group_%d_AUC" % g)
    plot_line_per_metrics(group_df, "Line_group_%d_AUC" % g)

group_names = ['Group 1', 'Group 2', 'Group 3']
plot_boxplot_per_metrics_per_group(group_dfs, group_names, "Box_group_all_AUC")
