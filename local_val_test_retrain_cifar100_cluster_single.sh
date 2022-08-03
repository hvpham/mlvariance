#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="logs/val_test_retrain_cifar100_local_cluster_$now.log"

source /home/hvpham/anaconda3/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate aml_pytorch_python36

python local_val_test_retrain_cifar100_cluster_single.py "/local2/teamdrive/data" "/local2/teamdrive/result" "holdout" 25

conda deactivate
