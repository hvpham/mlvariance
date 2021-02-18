import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='Run experiment with holdout CIFAR-100 and Resnet18')
parser.add_argument('data_folder', help='data folder')
parser.add_argument('result_folder', help='result folder')

args = parser.parse_args()

data_folder = args.data_folder
result_folder = args.result_folder

f = open("runs", "r")
runs = f.readlines()
f.close()

while not runs:
    script = runs.pop()
    arguments = runs.pop()
    log = runs.pop()

    run_command = 'bash cluster_single_run.sh "python %s %s %s %s" %s' % (script, data_folder, result_folder, arguments, log)
    subprocess.call(run_command, shell=True)