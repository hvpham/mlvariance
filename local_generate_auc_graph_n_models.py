import numpy as np
import pandas as pd
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate auc graph n models')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

result_path = os.path.join(result_folder, 'cifar100', 'overall_holdout_auc_no_models.csv')

df = pd.read_csv(result_path)


def generate_lineplot(data, xlabel, ylabel, xticklabels, legends, linecolor, linestyles, filename):
    fig = plt.figure(figsize=(12, 6))

    for i in range(len(data)):
        plt.plot(data[i], label=legends[i], color=linecolor[i], linestyle=linestyles[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), labels=xticklabels)
    plt.legend()

    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path)
    plt.close(fig)


def plot_line_per_metrics(df, filename_prefix):
    col_names = ['min_m_conf', 'median_m_conf']
    col_templates = ['avg_max_conf_%d', 'std_max_conf_%d']

    linelabels = ['Worst-PreConf', 'Median-PreConf',
                  'Avg-PreConf', 'SDev-PreConf']

    linecolors = ['red', 'orange', 'steelblue', 'green']
    linestyles = ['solid', 'solid', 'solid', 'solid']

    NO_SPLITS = 10

    data = []
    for col_name in col_names:
        data.append([np.mean(df[col_name].tolist())] * NO_SPLITS)

    for col_template in col_templates:
        d = []
        for split in range(1, NO_SPLITS+1):
            d.append(np.mean(df[col_template % split].tolist()))
        data.append(d)

    generate_lineplot(data,
                      "Num of models",
                      "Average blind spot sample indication AUC",
                      ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"],
                      linelabels,
                      linecolors,
                      linestyles,
                      filename_prefix)


ratio_10_df = df[(df['ratio'] == 10)]

#groups = [['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman'], ['caterpillar', 'mushroom', 'porcupine'], ['ray']]
groups = [['boy-girl', 'boy-man', 'girl-woman', 'man-woman'], ['caterpillar', 'mushroom', 'porcupine'], ['ray']]

group_dfs = []
for g in range(len(groups)):
    group_df = df[df['holdout_class'].isin(groups[g])]
    group_dfs.append(group_df)

    plot_line_per_metrics(group_df, "Line_group_n_models_%d" % g)
