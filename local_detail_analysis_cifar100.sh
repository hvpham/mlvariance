#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="logs/detail_analysis_cifar100_$now.log"

source /anaconda/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate azureml_py36_pytorch

python local_detail_analysis_cifar100.py 100 "/home/t-vpham/teamdrive/mlvariance/data" "/home/t-vpham/teamdrive/mlvariance/result" "holdout"

conda deactivate
