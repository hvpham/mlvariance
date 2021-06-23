#!/bin/bash
#now="$(date +"%y_%m_%d_%H_%M_%S")"
#LOG_FILE="logs/detail_analysis_cifar100_local_cluster_$now.log"

#source /home/hvpham/anaconda3/etc/profile.d/conda.sh
source /anaconda/etc/profile.d/conda.sh

#exec &>$LOG_FILE

conda activate azureml_py36_pytorch

python local_val_detail_analysis_cifar100_cluster_single.py "~/teamdrive/mlvariance/data" "~/teamdrive/mlvariance/result" "holdout" 25

conda deactivate
