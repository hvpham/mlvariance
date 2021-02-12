#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="logs/cifar100_holdout_runs_$now.log"

source /anaconda/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate azureml_py36_pytorch

python main_cifar100.py 5 "/home/t-vpham/teamdrive/mlvariance/data" "/home/t-vpham/teamdrive/mlvariance/result" "holdout"

conda deactivate
