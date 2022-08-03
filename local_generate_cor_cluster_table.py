import numpy as np
import pandas as pd
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from sklearn.cluster import KMeans

from scipy.spatial import ConvexHull
from scipy import interpolate
from scipy import stats

parser = argparse.ArgumentParser(description='Generate acc graph')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

result_path = os.path.join(result_folder, 'cifar100', 'result.csv')

df = pd.read_csv(result_path, usecols=[1, 2, 7, 11, 17], skiprows=[0], names=['holdout_class', 'ratio', 'avg_val', 'avg_test_holdout', 'std_dev_val'])


def cal_cor_for_group(dfs, group_names, filename):
    out_path = os.path.join(result_folder, 'cifar100', filename)
    f_out = open(out_path, "w")
    f_out.write('group,holdout_classes,avg_val_cor,avg_val_p_value,std_val_cor,std_val_p_value\n')

    for i in range(len(dfs)):
        df = dfs[i]
        avg_val_cor, avg_val_p_value = stats.pearsonr(df['avg_val'].tolist(), df['avg_test_holdout'].tolist())
        std_val_cor, std_val_p_value = stats.pearsonr(df['std_dev_val'].tolist(), df['avg_test_holdout'].tolist())

        f_out.write('%d,%s,%f,%f,%f,%f\n' % (i, group_names[i], avg_val_cor, avg_val_p_value, std_val_cor, std_val_p_value))

    f_out.close()


def plot_box_cor_for_group(df, groups, filename_prefix):
    data = []
    for i in range(len(groups)):
        group = groups[i]
        avg_val_cor_list = []
        std_val_cor_list = []
        for holdout_class in group:
            holdout_df = df[(df['holdout_class'] == holdout_class)]
            avg_val_cor, avg_val_p_value = stats.pearsonr(holdout_df['avg_val'].tolist(), holdout_df['avg_test_holdout'].tolist())
            std_val_cor, std_val_p_value = stats.pearsonr(holdout_df['std_dev_val'].tolist(), holdout_df['avg_test_holdout'].tolist())
            avg_val_cor_list.append(avg_val_cor)
            std_val_cor_list.append(std_val_cor)
        data.append([avg_val_cor_list, std_val_cor_list])

    ylabels = ['AvgValAcc', 'StdDevValAcc']

    fig = plt.figure(figsize=(12, 6))

    axs = fig.subplots(1, len(groups), sharey=True)

    def create_box_plot(ax, data, name):
        ax.boxplot(data)
        ax.set_xticks([1, 2])
        # ax.set_xlim(0.5, 3.5)
        ax.set_xticklabels(ylabels)
        ax.title.set_text(name)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

    medians = [np.median(d[1])+np.median(d[0]) for d in data]
    cluster_indexes = np.argsort(medians)

    for i, cluster_i in enumerate(cluster_indexes):
        create_box_plot(axs[i], data[cluster_i], 'C'+str(cluster_i))

    fig_path = os.path.join(result_folder, 'cifar100', filename_prefix+'.png')
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


groups = []
cluster_path = os.path.join(result_folder, 'cifar100', "holdout_cluster.txt")
f = open(cluster_path, "r")
for line in f:
    line = line.strip()
    line_split = line.split(',')
    index = int(line_split[0])
    line_split.pop(0)
    groups.append(line_split)
f.close()

plot_box_cor_for_group(df, groups, "Box_Group_All_Cor")

group_dfs = []
for g in range(len(groups)):
    group_df = df[df['holdout_class'].isin(groups[g])]
    group_dfs.append(group_df)

group_names = [('_'.join(list(cluster_name_list))).strip() for i, cluster_name_list in enumerate(groups)]
#group_names = ['C'+str(i) for i, cluster_name_list in enumerate(groups)]
cal_cor_for_group(group_dfs, group_names, "Group_cor_table.csv")
