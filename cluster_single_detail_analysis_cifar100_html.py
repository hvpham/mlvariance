'''Train CIFAR10 with PyTorch.'''
import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import torch.backends.cudnn as cudnn

import torchvision
#import torchvision.transforms as transforms

import os
import argparse

#from resnet import ResNet18
#from utils import progress_bar

from torchvision.datasets import VisionDataset

import pickle

from PIL import Image

import numpy as np

#torchvision.datasets.CIFAR10

import pandas

import matplotlib.cm as mpl_color_map

class CIFAR100_HOLDOUT(VisionDataset):
    train_file = 'train_batch'
    val_file = 'val_batch'
    test_file = 'test_batch'
    
    def __init__(self, holdoutroot, mode='train', transform=None, target_transform=None):

        super(CIFAR100_HOLDOUT, self).__init__(holdoutroot, transform=transform,
                                      target_transform=target_transform)

        self.mode = mode  # training set or test set

        if self.mode == 'train':
            file_name = self.train_file
        elif self.mode == 'val':
            file_name = self.val_file
        else:
            file_name = self.test_file

        self.data = []
        self.targets = []

        file_path = os.path.join(holdoutroot, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['labels']
            self.holdout = entry['holdout']
            self.ids = entry['ids']

        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def save_heatmap_images(org_img, activation_map, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    #color_map = 'hot'
    color_map = 'cool'
    #color_map = 'hsv'
    heatmap_on_image = apply_colormap_on_image(org_img, activation_map, color_map)
    
    path_to_file = os.path.join(folder, file_name)
    save_image(heatmap_on_image, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    color_map = mpl_color_map.get_cmap(colormap_name)
    heatmap = color_map(activation)
    
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('LA').convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return heatmap_on_image

def save_image(im, path):
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def format_np_output(np_arr):
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def softmax(x):
    exp_x = np.exp(x) 
    return exp_x / exp_x.sum(axis = 1, keepdims=True)


def evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, a):

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    if a != 10:
        valset = CIFAR100_HOLDOUT(holdoutroot=holdoutroot, mode='val')
    
    testset = CIFAR100_HOLDOUT(holdoutroot=holdoutroot, mode='test')
    
    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode,holdout_class,a))
    
    val_maps_list = []
    test_maps_list = []

    for run in range(NO_RUNS):
        maps_saving_file = os.path.join(saving_dir,'maps_%d' % run)
        maps = torch.load(maps_saving_file)

        val_maps = maps['val_maps']
        val_no_maps = int(val_maps.shape[0]/32)
        val_maps_list.append(val_maps.reshape(val_no_maps, 32, 32))

        test_maps = maps['test_maps']
        test_no_maps = int(test_maps.shape[0]/32)
        test_maps_list.append(test_maps.reshape(test_no_maps, 32, 32))

    
    def load_csv_file(setname):
        report_saving_file = os.path.join(saving_dir,'analysis_per_image_%s.csv' % setname)
        report = pandas.read_csv(report_saving_file)
        return report

    if a != 10:
        val_report = load_csv_file('val')
    test_report = load_csv_file('test')
    

    def process_maps(maps, report, images):
        maps =  np.stack(maps, axis=0)

        avg_maps = np.mean(maps, axis=0)
        max_maps = np.max(maps, axis=0)
        min_maps = np.min(maps, axis=0)
        max_diff_maps = max_maps - min_maps

        raw_report = []
        for c in range(12):
            raw_report.append(list(report.iloc[:,c]))

        new_report = []
        for i in range(12):
            new_report.append ([raw_report[i][r] for r in range(len(raw_report[0])) if raw_report[2][r]])

        order = np.argsort(new_report[3])

        for i in range(12):
            new_report[i] = [new_report[i][r] for r in order]
        

        new_report.append([avg_maps[id] for id in new_report[0]])
        new_report.append([max_diff_maps[id] for id in new_report[0]])
        new_report.append([images[id][0] for id in new_report[0]])

        return new_report

    if a != 10:
        val_report = process_maps(val_maps_list, val_report, valset)
    test_report = process_maps(test_maps_list, test_report, testset)

    image_relative_root = os.path.join("../../../data", 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    def read_script():
        with open("sort_script.txt", "r") as f:
            script = f.readlines()
        return script

    script = read_script()

    def output_html_analysis(setname, reports, script):
        analysis_out_file = os.path.join(saving_dir,'analysis_per_image_%s.html' % setname)
        map_out_folder = os.path.join(saving_dir,'maps_%s' % setname)

        f = open(analysis_out_file,"w")

        f.write('<!DOCTYPE html><html><body><h2>Analysis table for %s set</h2>\n' % setname)

        for line in script:
            f.write(line)

        f.write('<table id="indextable" style="width:100%"><thead><tr>')
        f.write("<th>Image</th>")
        f.write("<th>Avg gradcam</th>")
        f.write("<th>Max diff gradcam</th>")
        f.write('<th><a href="javascript:SortTable(3,\'N\');">Img id</a></th>')
        f.write('<th><a href="javascript:SortTable(4,\'N\');">Target</a></th>')
        f.write('<th><a href="javascript:SortTable(5,\'T\');">Is holdout</a></th>')
        f.write('<th><a href="javascript:SortTable(6,\'N\');"># correct</a></th>')
        f.write('<th><a href="javascript:SortTable(7,\'N\');">Avg ground conf</a></th>')
        f.write('<th><a href="javascript:SortTable(8,\'N\');">Stddev ground conf</a></th>')
        f.write('<th><a href="javascript:SortTable(9,\'N\');">Avg ground conf gaps</a></th>')
        f.write('<th><a href="javascript:SortTable(10,\'N\');">Stddev ground conf gaps</a></th>')
        f.write('<th><a href="javascript:SortTable(11,\'N\');">Avg ground conf ranks</a></th>')
        f.write('<th><a href="javascript:SortTable(12,\'N\');">Stddev ground conf ranks</a></th>')
        f.write('<th><a href="javascript:SortTable(13,\'N\');">Avg max conf</a></th>')
        f.write('<th><a href="javascript:SortTable(14,\'N\');">Stddev max conf</a></th>')
        f.write("</tr></thead>\n")

        f.write("<tbody>")
        for i in range(len(reports[0])):
            orig_image = reports[14][i]
            avg_map = reports[12][i]
            max_diff_map = reports[13][i]

            avg_map_name = "avg_%d_%s_%d.png" % (reports[1][i], reports[2][i], reports[0][i])
            save_heatmap_images(orig_image, avg_map, map_out_folder, avg_map_name)

            max_diff_map_name = "max_diff_%d_%s_%d.png" % (reports[1][i], reports[2][i], reports[0][i])
            save_heatmap_images(orig_image, max_diff_map, map_out_folder, max_diff_map_name)

            f.write("<tr>")
            f.write('<td><img src="%s/%s/%d_%s_%d.jpg" alt="%d_%s_%d.jpg"></td>' % (image_relative_root, setname, reports[1][i], reports[2][i], reports[0][i], reports[1][i], reports[2][i], reports[0][i]))
            f.write('<td><img src="./maps_%s/%s" alt="%s"></td>' % (setname, avg_map_name, avg_map_name))
            f.write('<td><img src="./maps_%s/%s" alt="%s"></td>' % (setname, max_diff_map_name, max_diff_map_name))
            
            f.write("<td>%d</td><td>%d</td><td>%s</td>" % (reports[0][i], reports[1][i], reports[2][i]))
            
            f.write("<td>%d</td>" % (reports[3][i]))
            for c in range(4,12):
                f.write("<td>%f</td>" % (reports[c][i]))
            
            f.write("</tr>\n")
    
        f.write("</tbody></table></body></html>")
        f.close()

    if a != 10:
        output_html_analysis('val', val_report, script)
    output_html_analysis('test', test_report, script)


parser = argparse.ArgumentParser(description='Analyze result with holdout CIFAR-10 and Resnet18')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')
parser.add_argument('holdout_class', 
    choices=['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray'],
    help='the holdout class')
parser.add_argument('holdout_target', help='the target that the holdout class belong to')
parser.add_argument('ratio', help='the ratio of holdout')
parser.add_argument('number_of_runs', help='the the number of runs')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder
mode = args.mode

holdout_class = args.holdout_class
holdout_target = int(args.holdout_target)
ratio = int(args.ratio)

NO_RUNS = int(args.number_of_runs)

print("Analyze mode:%s holdout:%s ratio:%d" % (mode, holdout_class, ratio))
evaluate_model(NO_RUNS, data_folder, result_folder, mode, holdout_class, holdout_target, ratio)
