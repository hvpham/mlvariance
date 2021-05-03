#!/bin/bash
now="$(date +"%y_%m_%d_%H_%M_%S")"
LOG_FILE="logs/detail_analysis_n_models_cifar100_$now.log"

source /home/hvpham/anaconda3/etc/profile.d/conda.sh

exec &>$LOG_FILE

conda activate aml_pytorch_python36

python local_detail_analysis_n_models_cifar100.py "/local2/teamdrive/data" "/local2/teamdrive/result" "holdout" 100 10

conda deactivate
