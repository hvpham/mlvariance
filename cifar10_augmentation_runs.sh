#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="logs/cifar10_augmentation_runs_$now.log"

source /anaconda/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate azureml_py36_pytorch

python main_cifar10.py 10 "/home/t-vpham/teamdrive/mlvariance/data" "/home/t-vpham/teamdrive/mlvariance/result" "augmentation"

conda deactivate
