'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18
#from utils import progress_bar

from torchvision.datasets import VisionDataset

import pickle

from PIL import Image

import numpy as np

import time

#import shutil

#import gc

#import matplotlib.cm as mpl_color_map

#import copy

#from scipy.ndimage.interpolation import zoom

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

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        pos = 0
        for module_pos, module in self.model._modules.items():
            #print("Forwarding in layer " + module_pos)
            #print(list(x.shape))

            if module_pos == 'linear':
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)

            x = module(x)  # Forward

            if pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer

            pos = pos + 1
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        # x = self.model.classifier(x)
        x = F.softmax(x, dim=1)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        #self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        #cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam

def check_done(path):
    if os.path.isfile(path):
        if os.stat(path).st_size > 0:
            return True
    return False

def generate_salient_map(run, data_folder, result_folder, mode, holdout_class, a):
    saving_dir = os.path.join(result_folder, 'cifar100', 'cifar100-%s_%s_%d' % (mode, holdout_class, a))
    model_saving_file = os.path.join(saving_dir,'model_%d.pth' % run)
    outputs_saving_file = os.path.join(saving_dir,'maps_%s' % run)

    if check_done(outputs_saving_file):
       print("Already generate map for run %d with holdout class %s and a %d. Skip to the next map generation run." % (run,holdout_class,a))
       return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('Preparing data..')
    tic = time.perf_counter()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    holdoutroot = os.path.join(data_folder, 'cifar100', '%s_%s_%d' % (mode, holdout_class, a))

    valset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='val', transform=transform_test)

    testset = CIFAR100_HOLDOUT(
        holdoutroot=holdoutroot, mode='test', transform=transform_test)

    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    
    print('Loading model')
    tic = time.perf_counter()
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    state_dict = torch.load(model_saving_file)
    net.load_state_dict(state_dict)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print('Converting model')
    tic = time.perf_counter()
    model = ResNet18()
    model = model.to(device)
    state_dict = net.module.state_dict()
    model.load_state_dict(state_dict)
    model.eval()
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print('Loading GradCam')
    tic = time.perf_counter()
    grad_cam = GradCam(model, target_layer=5)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    def generate_maps(data_set):
        cams_list = []
        #with torch.no_grad():
        for idx in range(len(data_set)):
            img, target = data_set[idx]
            img = (torch.tensor(np.expand_dims(img, axis=0))).to(device)
            cam = grad_cam.generate_cam(img, target)
            cams_list.append(cam)
        cams_list = np.vstack(cams_list)
        return cams_list
    
    results = {}

    print("Generating validation map")
    tic = time.perf_counter()
    results['val_maps'] = generate_maps(valset)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print("Generating test map")
    tic = time.perf_counter()
    results['test_maps'] = generate_maps(testset)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))

    print("Saving output file")
    tic = time.perf_counter()
    torch.save(results, outputs_saving_file)
    toc = time.perf_counter()
    print("Done in %f seconds" % (toc - tic))



parser = argparse.ArgumentParser(description='Run experiment with holdout CIFAR-100 and Resnet18')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')
parser.add_argument('mode', choices=['holdout'], help='the mode')
parser.add_argument('holdout_class', 
    choices=['baby', 'boy-girl', 'boy-man', 'girl-woman', 'man-woman', 'caterpillar', 'mushroom', 'porcupine', 'ray'],
    help='the holdout class')
parser.add_argument('ratio', help='the ratio of holdout')
parser.add_argument('run_id', help='the id of the run')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder
mode = args.mode

run_id = int(args.run_id)
holdout_class = args.holdout_class
ratio = int(args.ratio)

generate_salient_map(run_id, data_folder, result_folder, mode, holdout_class, ratio)