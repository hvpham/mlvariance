import numpy as np
import pandas as pd
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import statistics as st

parser = argparse.ArgumentParser(description='Generate acc graph')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

result_path = os.path.join(result_folder, 'cifar100', 'retrain_acc_gain_result.csv')

df_other = pd.read_csv(result_path, usecols=range(1, 9))

df_other = df_other.drop(['metric'], axis=1)

df_all = pd.read_csv(result_path)


def get_best_worst_median(df_all, metric):
    df_metric = df_all[df_all['retrain_mode'] == metric]

    best_df = df_metric.iloc[:, [1, 2, 3, 4, 14, 15, 16]]
    best_df["retrain_mode"].replace({metric: "best_"+metric}, inplace=True)
    best_df.set_axis(['holdout_class', 'retrain_mode', 'ratio', 'val_ratio', 'test_accu', 'test_holdout', 'test_normal'], axis=1, inplace=True)

    worst_df = df_metric.iloc[:, [1, 2, 3, 4, 18, 19, 20]]
    worst_df["retrain_mode"].replace({metric: "worst_"+metric}, inplace=True)
    worst_df.set_axis(['holdout_class', 'retrain_mode', 'ratio', 'val_ratio', 'test_accu', 'test_holdout', 'test_normal'], axis=1, inplace=True)

    median_df = df_metric.iloc[:, [1, 2, 3, 4, 26, 27, 28]]
    median_df["retrain_mode"].replace({metric: "median_"+metric}, inplace=True)
    median_df.set_axis(['holdout_class', 'retrain_mode', 'ratio', 'val_ratio', 'test_accu', 'test_holdout', 'test_normal'], axis=1, inplace=True)

    return pd.concat([best_df, worst_df, median_df], ignore_index=True)


single_df = get_best_worst_median(df_all, 'single_conf')
std_df = get_best_worst_median(df_all, 'std_conf')
avg_df = get_best_worst_median(df_all, 'avg_conf')
std_avg_df = get_best_worst_median(df_all, 'std_avg_conf')

df = pd.concat([df_other, single_df, std_df, avg_df, std_avg_df], ignore_index=True)


def generate_lineplot(data, ylabel, xticklabels, legends, linecolor, linestyles, filename):
    fig = plt.figure(figsize=(8, 6))

    for i in range(len(data)):
        plt.plot(data[i][1], data[i][0], label=legends[i],
                 color=linecolor[i], linestyle=linestyles[i])

    plt.xlabel('Additional data ratio')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), labels=xticklabels)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def compute_plot_acc_gain(df, setting, ratio, metric, filename_prefix):

    # holdout_classes = {'c0': ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum'],
    #                   'c9': ['squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal'],
    #                   'c1': ['boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree', 'apple', 'girl', 'orange', 'rose'],
    #                   'c8': ['cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale']}

    holdout_classes = {'c0': ['turtle', 'shark', 'crocodile', 'caterpillar', 'possum'],
                       'c1': ['boy', 'maple_tree', 'oak_tree', 'willow_tree', 'man', 'woman', 'pine_tree', 'apple', 'girl', 'orange', 'rose'],
                       'c2': ['porcupine', 'raccoon'],
                       'c3': ['orchid', 'sweet_pepper', 'beetle', 'hamster'],
                       'c4': ['mouse', 'otter', 'dolphin'],
                       'c5': ['skunk', 'aquarium_fish', 'fox', 'mushroom'],
                       'c6': ['bee', 'sunflower', 'trout'],
                       'c7': ['flatfish', 'butterfly', 'snake', 'dinosaur'],
                       'c8': ['cockroach', 'tulip', 'baby', 'palm_tree', 'poppy', 'pear', 'whale'],
                       'c9': ['squirrel', 'ray', 'shrew', 'lizard', 'beaver', 'rabbit', 'seal']}

    (retrain_modes, linecolors, linestyles) = setting

    val_ratios = [0, 1, 2, 5, 10]
    data = []
    labels = []
    for c, cs in holdout_classes.items():
        for retrain_mode in retrain_modes:
            labels.append(c + '_' + retrain_mode)

            group_df = df[(df['ratio'] == ratio) & (
                df['retrain_mode'] == retrain_mode) & (df['holdout_class'].isin(cs))]

            gains = []
            for val_ratio in val_ratios:
                r_df = group_df[group_df['val_ratio'] == val_ratio]
                r_m = r_df[metric].tolist()
                #print(retrain_mode+' ' + str(val_ratio))
                gains.append(st.mean(r_m))
            data.append((gains, val_ratios))

    data_path = os.path.join(result_folder, 'cifar100',
                             filename_prefix % (metric, ratio))
    f = open(data_path, 'w')

    f.write("Cluster,Metric")
    for val_ratio in val_ratios:
        f.write(',%d' % val_ratio)
    f.write('\n')

    line_index = 0
    for c, cs in holdout_classes.items():
        for mode in retrain_modes:
            f.write('%s,%s' % (c, mode))
            line = data[line_index]
            for gain in line[0]:
                f.write(',%f' % (gain))
            f.write('\n')
            line_index = line_index+1
    f.close()

    if linecolors is not None and linestyles is not None:

        #x_labels = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
        x_labels = ["0%", "1%", "2%", "3%", "4%", "5%", "6%", "7%", "8%", "9%", "10%"]

        generate_lineplot(data, "Test accuracy (%s)" % metric,
                          x_labels,
                          labels,
                          linecolors,
                          linestyles,
                          filename_prefix % (metric, ratio))


retrain_modes = ['random', 'std_conf', 'avg_conf', 'single_conf', 'std_avg_conf',
                 'worst_single_conf', 'best_single_conf', 'median_single_conf',
                 'worst_std_conf', 'best_std_conf', 'median_std_conf',
                 'worst_avg_conf', 'best_avg_conf', 'median_avg_conf',
                 'worst_std_avg_conf', 'best_std_avg_conf', 'median_std_avg_conf']

linecolors = ['red', 'red', 'red', 'green', 'green', 'green',
              'steelblue', 'steelblue', 'steelblue', 'purple', 'purple', 'purple',
              'brown', 'brown', 'brown', 'lime', 'lime', 'lime',
              'blue', 'blue', 'blue', 'magenta', 'magenta', 'magenta']
loosely_dashed = (0, (5, 10))
loosely_dotted = (0, (1, 10))
loosely_dashdotted = (0, (3, 10, 1, 10))

densely_dotted = (0, (1, 1))
densely_dashdotted = (0, (3, 1, 1, 1))
densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))

linestyles = ['solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted,
              'solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted,
              'solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted,
              'solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted]

#setting = (retrain_modes, linecolors, linestyles)
setting = (retrain_modes, None, None)

std_conf_setting = (['std_conf'],
                    ['lime', 'steelblue', 'green', 'purple', 'brown', 'red', 'blue', 'magenta', 'maroon', 'gold'],
                    ['solid', 'dotted', 'dashed', 'dashdot', loosely_dashed, loosely_dotted, loosely_dashdotted, densely_dotted, densely_dashdotted, densely_dashdotdotted])

for ratio in [0, 5, 10]:
    compute_plot_acc_gain(df, setting, ratio, 'test_holdout',
                          "Line_per_method_retrain_acc_gain_%s_%d.csv")
    compute_plot_acc_gain(df, setting, ratio, 'test_normal',
                          "Line_per_method_retrain_acc_gain_%s_%d.csv")
    compute_plot_acc_gain(df, setting, ratio, 'test_accu',
                          "Line_per_method_retrain_acc_gain_%s_%d.csv")

    compute_plot_acc_gain(df, std_conf_setting, ratio, 'test_holdout',
                          "Line_std_conf_retrain_acc_gain_%s_%d.csv")
    compute_plot_acc_gain(df, std_conf_setting, ratio, 'test_normal',
                          "Line_std_conf_retrain_acc_gain_%s_%d.csv")
    compute_plot_acc_gain(df, std_conf_setting, ratio, 'test_accu',
                          "Line_std_conf_retrain_acc_gain_%s_%d.csv")
