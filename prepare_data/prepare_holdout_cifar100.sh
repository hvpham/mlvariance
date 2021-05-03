#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="prepare_data_cifar100_$now.log"

source /anaconda/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate azureml_py36_pytorch

python prepare_holdout_cifar100-cifar10.py ~/teamdrive/mlvariance/data holdout

conda deactivate
