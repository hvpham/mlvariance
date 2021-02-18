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

while len(runs) > 0:
    script = runs.pop(0).strip()
    arguments = runs.pop(0).strip()
    log = runs.pop(0).strip()

    run_command = 'bash cluster_single_run.sh "python %s %s %s %s" %s' % (script, data_folder, result_folder, arguments, log)

    print("Running: %s" % run_command)

    subprocess.call(run_command, shell=True)
