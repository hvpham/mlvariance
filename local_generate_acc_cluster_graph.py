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

parser = argparse.ArgumentParser(description='Generate acc graph')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

result_folder = args.result_folder

result_path = os.path.join(result_folder, 'cifar100', 'result.csv')

df = pd.read_csv(result_path, usecols=range(1, 13))


def generate_scaterplot(df, xlabel, ylabel, filename):
    no_clusters = 10

    fig = plt.figure(figsize=(8, 8))

    # k means
    kmeans = KMeans(n_clusters=no_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['ori_acc', 'acc_drop']])
    # get centroids
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids]
    cen_y = [i[1] for i in centroids]
    # add to df
    df['cen_x'] = df.cluster.map(dict(zip(list(range(no_clusters)), cen_x)))
    df['cen_y'] = df.cluster.map(dict(zip(list(range(no_clusters)), cen_y)))

    # define and map colors
    colors = ['red', 'darkgreen', 'steelblue', 'purple', 'chocolate', 'pink', 'orange', 'lightseagreen', 'darkblue', 'deepskyblue']

    color_values = [mcolors.CSS4_COLORS[name] for name in colors]

    color_map = dict(zip(list(range(no_clusters)), color_values))
    #color_map = {0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3], 4: colors[4], 5: colors[5], 6: colors[6], 7: colors[7], 8: colors[8], 9: colors[9]}

    df['c'] = df.cluster.map(color_map)

    plt.scatter(df.ori_acc, df.acc_drop, c=df.c, alpha=0.6, s=10)

    # draw enclosure
    for i in df.cluster.unique():
        # get the convex hull
        points = df[df.cluster == i][['ori_acc', 'acc_drop']].values

        if len(points) == 1:
            pass
        elif len(points) == 2:
            x_values = [points[0][0], points[1][0]]
            y_values = [points[0][1], points[1][1]]
            plt.plot(x_values, y_values, c=color_values[i])
        else:
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices, 0],
                               points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1],
                               points[hull.vertices, 1][0])

            # interpolate
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull],
                                            u=dist_along, s=0)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)
            # plot shape
            plt.fill(interp_x, interp_y, '--', c=color_values[i], alpha=0.2)

    # create a list of legend elemntes
    ## markers / records
    holdout_classes = df.holdout_class.tolist()
    cluster = df.cluster.tolist()
    cluster_names = [set() for _ in range(no_clusters)]

    for i in range(len(holdout_classes)):
        cluster_names[cluster[i]].add(holdout_classes[i])

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='cluster ' + str(i) + ': ' + ' '.join(list(cluster_name_list)),
                              markerfacecolor=color_values[i], markersize=5) for i, cluster_name_list in enumerate(cluster_names)]
    # plot legend
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1), loc='upper left')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    fig_path = os.path.join(result_folder, 'cifar100', filename+'.png')
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    cluster_path = os.path.join(result_folder, 'cifar100', "holdout_cluster.txt")
    f = open(cluster_path, "w")
    for i, cluster_name_list in enumerate(cluster_names):
        f.write(str(i))
        for name in cluster_name_list:
            f.write(',')
            f.write(name)
        f.write('\n')
    f.close()


def plot_cluster_plot(df, filename_prefix):
    holdout_classes = set(df['holdout_class'].tolist())

    data = []
    for holdout_class in holdout_classes:
        group_df = df[df['holdout_class'] == holdout_class]
        d = group_df['test_holdout'].tolist()
        data.append([holdout_class, d[0], d[0]-d[10]])

    data_df = pd.DataFrame(data, columns=['holdout_class', 'ori_acc', 'acc_drop'])

    generate_scaterplot(data_df,
                        "Original holdout test accuracy", "Holdout test accuracy drop",
                        filename_prefix)


plot_cluster_plot(df, "Line_per_class_acc")
