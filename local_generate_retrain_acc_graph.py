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

result_path = os.path.join(result_folder, 'cifar100', 'retrain_result.csv')

df = pd.read_csv(result_path, usecols=range(1, 9))


def generate_lineplot(data, ylabel, xticklabels, legends, linecolor, linestyles, filename):
    fig = plt.figure(figsize=(8, 6))

    for i in range(len(data)):
        plt.plot(data[i][1], data[i][0], label=legends[i], color=linecolor[i], linestyle=linestyles[i])

    plt.xlabel('Validation data add on ratio')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), labels=xticklabels)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def plot_line_per_metrics(df, ratio, retrain_mode, metric, filename_prefix):
    #holdout_classes = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum', 'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']

    holdout_classes = ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum',
                       'squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal',
                       'boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree', 'apple', 'girl', 'orange', 'rose',
                       'cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale']

    data = []
    for i in range(len(holdout_classes)):
        group_df = df[(df['holdout_class'] == holdout_classes[i]) & (df['ratio'] == ratio) & (df['retrain_mode'] == retrain_mode)]
        data.append((group_df[metric].tolist(), group_df['val_ratio'].tolist()))

    linecolors = ['red', 'red', 'red', 'red', 'red',
                  'steelblue', 'steelblue', 'steelblue', 'steelblue', 'steelblue', 'steelblue', 'steelblue',
                  'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green',
                  'purple', 'purple', 'purple', 'purple', 'purple', 'purple', 'purple']

    loosely_dashed = (0, (5, 10))
    loosely_dotted = (0, (1, 10))
    loosely_dashdotted = (0, (3, 10, 1, 10))
    dashdotted = (0, (3, 5, 1, 5))
    densely_dashdotted = (0, (3, 1, 1, 1))
    dashdotdotted = (0, (3, 10, 1, 10, 1, 10))
    densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', loosely_dashed,
                  'solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted, loosely_dashdotted,
                  'solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted, loosely_dashdotted, dashdotted, densely_dashdotted, dashdotdotted, densely_dashdotdotted,
                  'solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted, loosely_dashdotted, ]

    generate_lineplot(data, "Test accuracy (%s)" % metric,
                      ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
                      holdout_classes,
                      linecolors,
                      linestyles,
                      filename_prefix % (metric, ratio, retrain_mode))


for ratio in [0, 5, 10]:
    for retrain_mode in ['random', 'std_conf', 'avg_conf', 'conf_worst', 'conf_best', 'conf_median']:
        plot_line_per_metrics(df, ratio, retrain_mode, 'test_holdout', "Line_per_class_retrain_acc_%s_%d_%s")
        plot_line_per_metrics(df, ratio, retrain_mode, 'test_accu', "Line_per_class_retrain_acc_%s_%d_%s")
        plot_line_per_metrics(df, ratio, retrain_mode, 'test_normal', "Line_per_class_retrain_acc_%s_%d_%s")
